#!/usr/bin/env python3
"""
convert_chain_to_multiturn.py

Turn a free‑form chain‑of‑thought string (with interleaved (x,y) coordinates)
into dialogue‑style multi‑turn data suitable for tool‑use training.
"""

import argparse, io, json, re, textwrap
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageDraw           # Pillow ≥9.x
import datasets
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os
import multiprocessing
from functools import partial
from qwen_vl_utils import fetch_image

random.seed(42)

# ────────────────────────────────────────────────────────────────────────────────
# Regex helpers
# ────────────────────────────────────────────────────────────────────────────────
COORD_RE = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
SENT_END_RE = re.compile(r"[.!?](?:\s|$|\n)")          # simple sentence boundary

def extract_coordinates(chain_text: str, scale_factor_width: float = 1.0, scale_factor_height: float = 1.0) -> List[Tuple[float, float, int, int]]:
    """Return [(x,y,start_idx,end_idx), …] in textual order."""
    
    if scale_factor_width != 1.0 or scale_factor_height != 1.0:
        def scale_coord(match):
            x = float(match.group(1))
            y = float(match.group(2))
            scaled_x = x * scale_factor_width
            scaled_y = y * scale_factor_height
            return f"({int(scaled_x)}, {int(scaled_y)})"
        
        chain_text = COORD_RE.sub(scale_coord, chain_text)
    
    coords = []
    for m in COORD_RE.finditer(chain_text):
        coords.append((float(m.group(1)), float(m.group(2)), m.start(), m.end()))
    
    return coords, chain_text

def extract_final_answer(chain_text: str) -> Tuple[int, int] | str | None:
    # Try to match coordinate pattern
    m = re.search(r"<answer>\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)\s*</answer>",chain_text, re.DOTALL)
    if m:
        return int(float(m.group(1))), int(float(m.group(2))), True
    
    # Try to match text pattern
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", chain_text, re.DOTALL)
    if m:
        return m.group(1).strip(), False
    
    return None

def _get_point_crop(image: Image.Image, point: List[int], offset: int = 50, crop_size: int = 512, draw_dot: bool = True) -> Image.Image:
    """
    Get a crop of the image centered on the point with a side length of 2*offset.
    Also draws a dot at the point location within the crop.
    """
    x, y = point
    width, height = image.size

    if x > width or y > height or x < 0 or y < 0:
        raise ValueError(f"Point ({x}, {y}) is outside of image dimensions ({width}, {height})")

    # Ensure crop boundaries are within image dimensions
    left = max(0, x - offset)
    top = max(0, y - offset)
    right = min(width, x + offset)
    bottom = min(height, y + offset)
    
    # Ensure that right > left and bottom > top
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    
    # Create the crop
    crop = image.crop((left, top, right, bottom))
    
    # Draw a dot at the point location (relative to the crop)
    if draw_dot:
        draw = ImageDraw.Draw(crop)
        dot_x = x - left
        dot_y = y - top
        radius = 7
        draw.ellipse(
            [(dot_x - radius, dot_y - radius), (dot_x + radius, dot_y + radius)],
            fill="red",
            outline="white",
            width=2
        )

    # Super-resolution to crop_size x crop_size
    crop = fetch_image({"image": crop, "resized_width": crop_size, "resized_height": crop_size})

    return crop

def img_to_bytes_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ────────────────────────────────────────────────────────────────────────────────
# Core conversion routine
# ────────────────────────────────────────────────────────────────────────────────

def _next_sentence_end(txt: str, start_idx: int) -> int:
    """Return index just *after* the next sentence‑terminating punctuation."""
    m = SENT_END_RE.search(txt, pos=start_idx)
    return (m.end() if m else len(txt))

def convert_chain_to_dialogue(chain_text: str,
                              img_path: Path,
                              description: str,
                              max_turns: int = 5,
                              crop_resize: int = 512,
                              offset: int = 50,
                              shuffle_prob: float = 0.5,
                              sys_prompt: str = None,
                              draw_dot: bool = True,
                              minimum_side_length: int = None,
                              min_pixels: int = None,
                              max_pixels: int = None):
    img = Image.open(img_path).convert("RGB")

    scale_factor_width = 1.0
    scale_factor_height = 1.0
    if minimum_side_length:
        if img.width < minimum_side_length or img.height < minimum_side_length:
            # resize the image to the minimum side length while maintaining aspect ratio
            width, height = img.size
            if width < height:
                new_width = minimum_side_length
                new_height = int(height * (minimum_side_length / width))
                scale_factor_width = new_width / width
                scale_factor_height = new_height / height
            elif width > height:
                new_height = minimum_side_length
                new_width = int(width * (minimum_side_length / height))
                scale_factor_width = new_width / width
                scale_factor_height = new_height / height
            else:
                new_width = minimum_side_length
                new_height = minimum_side_length
                scale_factor_width = new_width / width
                scale_factor_height = new_height / height

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    elif min_pixels and max_pixels:

        width, height = img.size

        img = fetch_image({"image": img, "min_pixels": min_pixels, "max_pixels": max_pixels})

        width_new, height_new = img.size

        scale_factor_width = width_new / width
        scale_factor_height = height_new / height

    coord_matches, chain_text = extract_coordinates(chain_text, scale_factor_width, scale_factor_height)
    final_answer, is_coord = extract_final_answer(chain_text)
    if final_answer is None:
        # print(f"No final answer found in chain text: {chain_text}")
        return None
        
    cursor = 0
    turns, seen_coords = [], set()
    accumulated_reasoning, tool_call_count = "", 0

    for (x, y, start, end) in coord_matches:
        # # Sentence that *contains* this coordinate -----------------------------
        # sent_end = _next_sentence_end(chain_text, end)
        # prefix   = chain_text[cursor:start]           # up to coord (no coord)
        # suffix   = chain_text[end:sent_end]           # after coord → sentence end
        # reasoning_chunk = (prefix + suffix).strip()
        # cursor = sent_end                             # advance past the sentence

        # Sentence that *contains* this coordinate -----------------------------
        sent_end = _next_sentence_end(chain_text, end)
        prefix   = chain_text[cursor:start]           # up to coord (no coord)
        suffix   = chain_text[end:sent_end]           # after coord → sentence end
        reasoning_chunk = (prefix + chain_text[start:end] + suffix).strip()
        cursor = sent_end

        if reasoning_chunk:
            accumulated_reasoning = (accumulated_reasoning + "\n\n" + reasoning_chunk
                                      if accumulated_reasoning else reasoning_chunk)

        coord_key = (int(x), int(y))
        if coord_key in seen_coords or tool_call_count >= max_turns:
            continue

        # Build assistant turn -------------------------------------------------
        seen_coords.add(coord_key)
        tool_call_count += 1

        accumulated_reasoning = accumulated_reasoning.replace("<think>", "").replace("</think>", "").strip()
        think_block = f"<think> {accumulated_reasoning} </think>\n"
        accumulated_reasoning = ""                    # reset

        tool_json = {"name": "search_coordinate",
                     "arguments": {"coordinate": [int(x), int(y)]}}
        action_block = "<tool_call>\n" + json.dumps(tool_json) + "\n</tool_call>"

        turns.append({"role": "assistant", "content": think_block + action_block})

        # Build user observation (image crop) ----------------------------------
        try:
            # crop_img = crop_with_dot(img, (int(x), int(y)), box_size=crop_px)
            crop_img = _get_point_crop(img, [int(x), int(y)], offset=offset, crop_size=crop_resize, draw_dot=draw_dot)
        except Exception:
            # print(f"Error cropping image at ({x}, {y})")
            return None
        
        if draw_dot:
            turns.append({
                "role": "user",
                "content": "<observation>\nHere is the crop of the image centered on the coordinate, with a red dot at the coordinate location:\n<image>\n</observation>",
                "_img_bytes": img_to_bytes_png(crop_img)
            })
        else:
            turns.append({
                "role": "user",
                "content": "<observation>\nHere is the crop of the image centered on the coordinate:\n<image>\n</observation>",
                "_img_bytes": img_to_bytes_png(crop_img)
            })

    # Append any leftover reasoning before final answer ------------------------
    if cursor < len(chain_text):
        trailing = chain_text[cursor:].strip()
        trailing = trailing.replace("<think>", "").replace("</think>", "").strip()
        if trailing:
            accumulated_reasoning += ("\n\n" + trailing if accumulated_reasoning else trailing)

    # final_think  = f"<think> {accumulated_reasoning or 'Based on all information, here is my answer.'} </think>\n"
    final_think = "<think> Based on all the information I've gathered, I'll now provide my final answer. </think>\n"
    if is_coord:
        answer_block = f"<answer> ({final_answer[0]}, {final_answer[1]}) </answer>"
    else:
        answer_block = f"<answer> {final_answer} </answer>"
    turns.append({"role": "assistant", "content": final_think + answer_block})

    # Assemble message list ----------------------------------------------------
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user",   "content": f"{description}"}
    ] + [{k: v for k, v in t.items() if k != "_img_bytes"} for t in turns]

    image_list = [{"bytes": img_to_bytes_png(img)}] + [
        {"bytes": t["_img_bytes"]} for t in turns if "_img_bytes" in t
    ]

    # With some probability, reposition the first tool call and observation
    if len(messages) > 5 and random.random() < shuffle_prob:  # Need at least 2 tool calls to shuffle
        # Only consider turns with tool calls and their observations
        valid_indices = []
        for i in range(4, len(messages) - 2, 2):  # Start after first tool call, end before final answer
            if "tool_call" in messages[i]["content"] and messages[i+1]["role"] == "user":
                valid_indices.append(i)
        
        if valid_indices:
            # Extract the first tool call (index 2) and observation (index 3)
            first_tool_call = messages[2]
            first_observation = messages[3]
            
            # Remove them from their original position
            messages = messages[:2] + messages[4:]
            
            # Pick a random valid insertion point (after another observation)
            insert_idx = random.choice(valid_indices) - 2  # Adjust for removal of first pair
            
            # Insert at the new position
            messages = messages[:insert_idx] + [first_tool_call, first_observation] + messages[insert_idx:]
            
            # Update image list order to match
            # First find the index of the observation image in the original list
            obs_img_idx = 1  # First observation image is at index 1
            obs_img = image_list[obs_img_idx]
            
            # Remove and reinsert at appropriate position
            image_list = image_list[:obs_img_idx] + image_list[obs_img_idx+1:]
            new_img_idx = 1 + (insert_idx - 2) // 2  # Calculate new image position
            image_list.insert(new_img_idx, obs_img)

    return {"messages": messages, "images": image_list}

def process_paths_batch(batch_data):
    """Process a batch of rows to save images and create path references."""
    batch_idx, batch, images_folder = batch_data
    rows_with_paths = []
    
    for i, row in enumerate(batch):
        idx = batch_idx * len(batch) + i  # Calculate global index
        row_with_paths = {"messages": []}
        
        # Save images and update references
        image_paths = []
        for img_idx, img in enumerate(row["images"]):
            img_bytes = img["bytes"]
            img_filename = f"{idx}_{img_idx}.png"
            img_path = f"{images_folder}/{img_filename}"
            
            # Save the image
            os.makedirs(os.path.dirname(img_path), exist_ok=True)  # Ensure directory exists
            with open(img_path, "wb") as f:
                f.write(img_bytes)
            
            # Add path to list
            image_paths.append(img_path)
        
        # Update messages to reference image paths instead of bytes
        for msg in row["messages"]:
            msg_copy = msg.copy()
            if "_img_bytes" in msg:
                # Remove the bytes field
                del msg_copy["_img_bytes"]
            row_with_paths["messages"].append(msg_copy)
        
        # Add image paths
        row_with_paths["images"] = image_paths
        rows_with_paths.append(row_with_paths)
    
    return rows_with_paths

def process_chain_batch(chains, max_turns, crop_resize, offset, shuffle_prob, sys_prompt, draw_dot, minimum_side_length, min_pixels, max_pixels):
    """Process a batch of chains in a worker process."""
    results = []
    for chain in chains:
        chain_text = chain["messages"][-1]["content"]
        description = chain["messages"][1]["content"]
        img_path = Path(chain["images"][0])

        dlg = convert_chain_to_dialogue(chain_text, img_path,
                                        description, max_turns, crop_resize, offset, shuffle_prob, sys_prompt, draw_dot, minimum_side_length, min_pixels, max_pixels)
        if dlg:
            results.append(dlg)
    return results

def main():
    '''
    Please run build_reasoning_chains_from_mcts.py first to get the single turn rollouts. Then specify path to single turn rollouts as mcts_chain_path.
    '''

    # Adjust this as needed
    mcts_chain_path = "path/to/rollouts" # TODO: change this to the path to the single turn rollouts
    prompt_type = "web_grounding" # "web_grounding", "spatial", "web_action", "vstar"
    images_folder = "/path/to/images" # path to save training cropped images
    output_parquet_paths = "/path/to/output/parquet_w_image_paths" # path to save the parquet file with image paths

    system_prompt_web_grounding = textwrap.dedent("""You are an assistant tasked with identifying precise (x,y) coordinates of a described element in an image.\nYour task involves multiple turns of reasoning, each with EXACTLY one <think> step and one action:\n- At each turn, first clearly reason about ONE area or element in the image enclosed in <think> </think> tags.\n- After reasoning, either:\n  a) Output a search action formatted precisely as:\n     <tool_call>\n     {\"name\": \"search_coordinate\", \"arguments\": {\"coordinate\": [x, y]}}\n     </tool_call>\n  b) If confident you've found the correct location, output your final answer enclosed in <answer> (x, y) </answer> tags.\n- Only answer if you are confident about the answer. If you are not confident, output a search action. You should not always end after one turn.\n- You should not repeat the same coordinates in a tool call more than once. Coordinates must be unique across tool calls, including values that are the same or nearly identical (e.g., differing by only a few pixels).""")

    sys_prompt_qa = textwrap.dedent("""You are a helpful assistant tasked with answering a question about an image. You should systematically reason through the problem step by step by checking and verifying relevant image regions, while grounding reasoning steps to specific (x, y) points in the image:\n- At each turn, first clearly reason about ONE area or element in the image enclosed in <think> </think> tags.\n- After reasoning, either:\n  a) Zoom-in on a specific region to see it better by outputting a search action formatted precisely as:\n     <tool_call>\n     {\"name\": \"search_coordinate\", \"arguments\": {\"coordinate\": [x, y]}}\n     </tool_call>\n  b) If confident you've found the correct location, output your final answer enclosed in <answer> {final answer} </answer> tags.\n- Only answer if you are confident about the answer. If you are not confident, output a search action. You should not always end after one turn.\n- You should not repeat the same coordinates in a tool call more than once. Coordinates must be unique across tool calls, including values that are the same or nearly identical (e.g., differing by only a few pixels).\n- If unclear, infer based on likely context or purpose.\n- Verify each step by examining multiple possible solutions before selecting a final answer.""")

    # defaults
    max_turns = 5
    offset = 182
    crop_resize = 672
    shuffle_prob = 0.2
    minimum_side_length = None
    min_pixels = 2700000
    max_pixels = 2850000
    draw_dot = False

    if prompt_type == "web_grounding":
        sys_prompt = system_prompt_web_grounding
        draw_dot = True
        crop_resize = 512
        offset = 100
    elif prompt_type == "spatial":
        sys_prompt = sys_prompt_qa
    elif prompt_type == "web_action":
        sys_prompt = sys_prompt_qa
    elif prompt_type == "vstar":
        sys_prompt = sys_prompt_qa
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    
    # Number of processes to use (leave some cores free)
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Loading data from {mcts_chain_path}...")
    with open(mcts_chain_path, "r", encoding="utf-8") as f:
        mcts_chain = json.load(f)
    
    print(f"Processing {len(mcts_chain)} chains using {num_processes} processes...")
    
    # Split chains into roughly equal batches
    batch_size = len(mcts_chain) // num_processes
    batches = [mcts_chain[i:i+batch_size] for i in range(0, len(mcts_chain), batch_size)]
    
    # Create a partial function with the fixed arguments
    process_fn = partial(process_chain_batch, max_turns=max_turns, 
                         crop_resize=crop_resize, offset=offset, shuffle_prob=shuffle_prob, sys_prompt=sys_prompt, draw_dot=draw_dot, minimum_side_length=minimum_side_length, min_pixels=min_pixels, max_pixels=max_pixels)
    
    # Process batches in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        batch_results = list(tqdm(pool.imap(process_fn, batches), 
                                  total=len(batches), 
                                  desc="Processing batches"))
    
    # Flatten results from all batches
    rows = [item for sublist in batch_results for item in sublist]
    print(f"Successfully processed {len(rows)} chains")
    
    # # Save to parquet
    # datasets.Dataset.from_pandas(pd.DataFrame(rows)).to_parquet(output_parquet)
    # print(f"Saved results to {output_parquet}")

    # Create a version with image paths instead of bytes - now with multiprocessing
    print(f"Creating path-based version with multiprocessing...")
    
    # Ensure images directory exists
    os.makedirs(images_folder, exist_ok=True)
    
    # Split rows into batches for parallel processing
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    batch_size = max(1, len(rows) // num_processes)
    batches = [(i, rows[i*batch_size:(i+1)*batch_size], images_folder) 
               for i in range(num_processes)]
    if num_processes * batch_size < len(rows):
        batches.append((num_processes, rows[num_processes*batch_size:], images_folder))
    
    # Process batches in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(process_paths_batch, batches), 
            total=len(batches), 
            desc="Processing image batches"
        ))
    
    # Flatten results from all batches
    rows_with_paths = [item for sublist in batch_results for item in sublist]
    
    # Save to parquet with image paths
    datasets.Dataset.from_pandas(pd.DataFrame(rows_with_paths)).to_parquet(output_parquet_paths)
    print(f"Saved results with image paths to {output_parquet_paths}")

if __name__ == "__main__":
    main()

