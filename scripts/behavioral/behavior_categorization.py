#!/usr/bin/env python
import argparse, json, os, re, random, shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont           # Pillow ≥9.x
import cv2
import numpy as np
import math
from tqdm import tqdm
import os
import json
import argparse
import re
from openai import AzureOpenAI
import random
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from tqdm import tqdm

client = AzureOpenAI(
    api_key = os.getenv("AZURE_OPENAI_KEY"),  
    api_version = "2023-05-15",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

random.seed(18)

BLUE = "#00BFFF"
GREEN = "#00C000"
CIRCLE_R = 8            # small green dot
OUTLINE_R = 100         # blue annotation circle

BLUE = "#00BFFF"
GREEN = "#00C000"
MARKER_R = 15            # radius of the filled circle marker
CIRCLE_R = 8             # small green dot (unused in new version)
OUTLINE_R = 100          # blue annotation circle (unused in new version)


COORD_RE = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")

def natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# keep your existing constants
GREEN = "#00C000"
MARKER_R = 15
COORD_RE = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)

def cluster_points(points, cluster_thresh=None):
    """Group raw (x,y) points if they're closer than cluster_thresh."""
    if cluster_thresh is None:
        cluster_thresh = MARKER_R * 2
    clusters = []  # each is {"center":(cx,cy), "members":[(x,y),...]}
    for x,y in points:
        placed = False
        for cl in clusters:
            cx, cy = cl["center"]
            if math.hypot(x-cx, y-cy) <= cluster_thresh:
                cl["members"].append((x,y))
                # update center by running average
                n = len(cl["members"])
                cl["center"] = ((cx*(n-1) + x)/n, (cy*(n-1) + y)/n)
                placed = True
                break
        if not placed:
            clusters.append({"center":(x,y), "members":[(x,y)]})
    return clusters

def draw_points(img_path: Path, points, max_side=None):
    """
    Returns: (pil_image, point_to_cluster_map)
    where point_to_cluster_map[(x,y)] = cluster_id
    """
    # --- load + optional resize ---
    img = Image.open(img_path).convert("RGB")
    if max_side and max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((int(img.width*scale), int(img.height*scale)))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # cluster raw points
    clusters = cluster_points(points)
    # build lookup: each original point → its cluster index
    point_to_cluster = {}
    for cid, cl in enumerate(clusters, start=1):
        for member in cl["members"]:
            point_to_cluster[member] = cid

    green_bgr = hex_to_bgr(GREEN)

    # draw each cluster once
    for cid, cl in enumerate(clusters, start=1):
        cx, cy = cl["center"]
        # filled circle
        cv2.circle(
            img,
            (int(cx), int(cy)),
            MARKER_R,
            green_bgr,
            thickness=-1
        )
        # label = single cluster id
        label = str(cid)
        (tw, th), _ = cv2.getTextSize(
            label,
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            thickness=2
        )
        tx = int(cx - tw/2)
        ty = int(cy + th/2)
        cv2.putText(
            img,
            label,
            (tx, ty),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.2,
            color=(255,255,255),
            thickness=2,
            lineType=cv2.LINE_AA
        )

    # back to PIL
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return pil, point_to_cluster

def replace_coords(text: str, point_to_cluster: dict):
    """
    Replace every '(x, y)' with '(Point #k)' where k is the cluster ID
    for that exact coordinate.
    """
    def repl(m):
        x, y = float(m.group(1)), float(m.group(2))
        cid = point_to_cluster.get((x,y))
        if cid is None:
            return m.group(0)
        return f"(Point #{cid})"
    return COORD_RE.sub(repl, text)

def create_analysis_prompts(reasoning: str):
        prompts = [
            # 1. Visual Verification steps 
            f"""Here is a chain-of-reasoning that a Language Model generated while analyzing an image. The chain-of-reasoning output from the model is: 
            '''\n{reasoning}\n'''
Evaluate whether the chain-of-reasoning contains any visual verification steps. A visual verification step is when the model confirms or checks something it sees in the image. Examples include: "I can see that the object is not a cat, but a dog", "The text confirms this visual aspect is correct", "I can verify this is indeed red", or "Looking at the image, I can confirm...". Count both explicit mentions of image regions and implicit verifications.
Count all instances where the model verifies information from the image and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any visual verification steps, please provide a count of 0 as <count>0</count>.""",

            # 2. Backtracking 
            f"""Here is a chain-of-reasoning that a Language Model generated while analyzing an image. The chain-of-reasoning output from the model is: 
            '''\n{reasoning}\n'''
Evaluate whether the chain-of-reasoning contains any backtracking behavior, where the model changes its interpretation or corrects itself. Examples include: "At first I thought X, but looking more carefully I see it's actually Y", "I initially interpreted this as a circle, but it's actually an oval", "On second thought...", "Actually, I notice that...", or "Let me correct my earlier observation...".
Count all instances where the model revises its understanding and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backtracking behavior, please provide a count of 0 as <count>0</count>.""",

            # 3. Subgoal setting 
            f"""Here is a chain-of-reasoning that a Language Model generated while analyzing an image. The chain-of-reasoning output from the model is: 
            '''\n{reasoning}\n'''
Evaluate whether the chain-of-reasoning contains any visual subgoal setting, where the model breaks down the image analysis into smaller steps or focuses on different parts of the image in sequence. Examples include: "First, I'll examine this part, then I'll look at that object", "Let me check each element one by one", "I need to identify what's in this area", or any structured approach to analyzing different parts of the image.
Count all instances where the model sets up a plan or approach for analyzing the image and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any visual subgoal setting, please provide a count of 0 as <count>0</count>.""",

            # 4. Visual regions explored
            f"""Here is a chain-of-reasoning that a Language Model generated while analyzing an image. The chain-of-reasoning output from the model is: 
            '''\n{reasoning}\n'''
Count how many distinct visual regions or elements the model explicitly mentions examining in the image. Examples include: "I can see a dog in the corner", "There's text at the top of the image", "The object in the center appears to be...", "Looking at the left side...", or any reference to a specific part or element of the image that the model is analyzing.
Count all distinct visual regions or elements mentioned and provide the count between the tags <count> </count>. If the chain-of-reasoning does not mention any specific visual regions, please provide a count of 0 as <count>0</count>.""",
        ]

        return prompts

# ----------  roll‑out loading  ----------
def load_correct_rollouts(root: Path, out_img_dir: Path,
                          model_tag: str, max_side=None):
    """
    Return dict[id] = { question, reasoning, answer, img_path }
    Only rollouts with judge_score == 1 are kept.
    """
    
    results = {}
    count = 0
    paths = list(root.rglob("*.jsonl"))
    random.shuffle(paths)
    for jsonl_path in tqdm(paths):
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)

                rid = r["id"]
                if rid in results:                      # first perfect wins
                    continue

                if not r["final_answer"]:
                    # print(f"Skipping {r['image']} because it has no final answer.")
                    # continue
                    r["final_answer"] = ""

                if "<think>" in r["final_answer"]:
                    # get content between <think> and </think>
                    try:
                        reasoning = re.search(r"<think>(.*?)</think>", r["final_answer"], re.DOTALL).group(1)
                        answer = re.search(r"<answer>(.*?)</answer>", r["final_answer"], re.DOTALL).group(1)
                        r["final_answer"] = answer
                        r["thoughts"] = [reasoning]
                    except:
                        r["final_answer"] = ""
                        r["thoughts"] = [""]

                if len(r["thoughts"]) == 0:
                    # print(f"Skipping {r['image']} because it has no thoughts.")
                    # continue
                    r["thoughts"] = [""]

                reasoning = r["thoughts"][0] if isinstance(r["thoughts"], list) else r["thoughts"]
                all_coords = [(float(x), float(y)) for x, y in COORD_RE.findall(reasoning)]

                # Prepare image
                orig_img = Path(r["image"])

                results[rid] = dict(
                    question = r.get("question", ""),
                    reasoning = reasoning,
                    answer    = r.get("final_answer", ""),
                    judge_score = r["judge_score"],
                    orig_img_path = str(orig_img),
                    # behavior_counts = behavior_counts,
                    unique_coords = list(set(all_coords))
                )

                count += 1
                # if count > 10:
                #     return results
                
    return results

def process_item(rid, c, c_idx, idx, behavior_types_to_run=None):
    """Process a single item with API calls in a separate process"""
    rec1 = c[rid]
    behavior_counts = {}
    prompts = create_analysis_prompts(rec1["reasoning"])

    behavior_types = ["Verification", "Backtracking", "Subgoal Setting", "Visual Regions Explored"]
    
    # If behavior_types_to_run is specified, only run those behaviors
    if behavior_types_to_run:
        behaviors_to_process = [(i, b) for i, b in enumerate(behavior_types) if b in behavior_types_to_run]
    else:
        behaviors_to_process = list(enumerate(behavior_types))
    
    # Create a new client in each process to avoid issues with sharing the client
    client = AzureOpenAI(
        api_key = os.getenv("AZURE_OPENAI_KEY"),  
        api_version = "2023-05-15",
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    
    # for behavior, prompt in zip(["Verification", "Backtracking", "Subgoal Setting", "Visual Regions Explored"], prompts):
    for i, behavior in behaviors_to_process:

        try:
            messages = [
                {"role": "user", "content": [{"type": "text", "text": f"{prompts[i]}"}]},
            ]

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=256,
                temperature=0.0,
                top_p=1.0,
            )

            count = re.search(r"<count>(.*?)</count>", completion.choices[0].message.content)
            if count:
                count = int(count.group(1))
            else:
                count = 0
            behavior_counts[behavior] = count
        except Exception as e:
            print(f"Error processing item {rid} for behavior {behavior}: {e}")
            behavior_counts[behavior] = 0

    return {
        "c_idx": c_idx,
        "entry": {
            "Question"  : rec1["question"],
            "img_path" : rec1["orig_img_path"],
            "reasoning": rec1["reasoning"],
            "Answer"   : rec1["answer"],
            "Verification" : behavior_counts.get("Verification", 0),
            "Backtracking" : behavior_counts.get("Backtracking", 0),
            "Subgoal Setting" : behavior_counts.get("Subgoal Setting", 0),
            "Visual Regions Explored" : behavior_counts.get("Visual Regions Explored", 0),
            "Unique Coords" : rec1["unique_coords"],
            "Correctness" : rec1["judge_score"],
            "id"        : rid,
            "index"     : idx
        }
    }
def process_item_wrapper(args):
    """Wrapper function to unpack arguments for process_item"""
    rid, c, c_idx, idx = args
    return process_item(rid, c, c_idx, idx)

def calculate_and_print_metrics(out_root, model_tags, model_dirs):
    """
    Calculate and print summary metrics for each condition
    """
    print("\n=== Summary Metrics ===")
    
    # Save the mapping of model tags to directories
    with open(out_root / "model_mapping.json", "w") as mapping_file:
        json.dump({tag: str(path) for tag, path in zip(model_tags, model_dirs)}, mapping_file, indent=2)
    
    for tag_idx, tag in enumerate(model_tags):
        data_path = out_root / f"data_{tag}.jsonl"
        metrics = {
            "Verification": [],
            "Backtracking": [],
            "Subgoal Setting": [],
            "Visual Regions Explored": [],
            "Unique Coords": [],
            "Correctness": []
        }
        
        with open(data_path) as f:
            for line in f:
                entry = json.loads(line)
                metrics["Verification"].append(entry["Verification"])
                metrics["Backtracking"].append(entry["Backtracking"])
                metrics["Subgoal Setting"].append(entry["Subgoal Setting"])
                metrics["Visual Regions Explored"].append(entry["Visual Regions Explored"])
                metrics["Unique Coords"].append(len(entry["Unique Coords"]))
                metrics["Correctness"].append(entry["Correctness"])

        print(f"\nCondition: {tag} - {model_dirs[tag_idx]}")
        print(f"  Verification steps: {sum(metrics['Verification'])}/{len(metrics['Verification'])} entries, "
              f"avg: {sum(metrics['Verification'])/len(metrics['Verification']):.2f}")
        print(f"  Backtracking instances: {sum(metrics['Backtracking'])}/{len(metrics['Backtracking'])} entries, "
              f"avg: {sum(metrics['Backtracking'])/len(metrics['Backtracking']):.2f}")
        print(f"  Subgoal setting: {sum(metrics['Subgoal Setting'])}/{len(metrics['Subgoal Setting'])} entries, "
              f"avg: {sum(metrics['Subgoal Setting'])/len(metrics['Subgoal Setting']):.2f}")
        print(f"  Unique coordinates: total: {sum(metrics['Unique Coords'])}, "
              f"avg: {sum(metrics['Unique Coords'])/len(metrics['Unique Coords']):.2f} per entry")
        print(f"  Visual regions explored: total: {sum(metrics['Visual Regions Explored'])}, "
              f"avg: {sum(metrics['Visual Regions Explored'])/len(metrics['Visual Regions Explored']):.2f} per entry")
        print(f"  Correctness: total: {sum(metrics['Correctness'])}, "
              f"avg: {sum(metrics['Correctness'])/len(metrics['Correctness']):.2f} per entry")
        # Export metrics to CSV
        with open(out_root / f"metrics_{tag}.json", "w") as f_out:
            json.dump({
                "model_dir": str(model_dirs[tag_idx]),
                "verification_avg": sum(metrics['Verification'])/len(metrics['Verification']),
                "backtracking_avg": sum(metrics['Backtracking'])/len(metrics['Backtracking']),
                "subgoal_setting_avg": sum(metrics['Subgoal Setting'])/len(metrics['Subgoal Setting']),
                "unique_coords_avg": sum(metrics['Unique Coords'])/len(metrics['Unique Coords']),
                "visual_regions_explored_avg": sum(metrics['Visual Regions Explored'])/len(metrics['Visual Regions Explored']),
                "verification_total": sum(metrics['Verification']),
                "backtracking_total": sum(metrics['Backtracking']),
                "subgoal_setting_total": sum(metrics['Subgoal Setting']),
                "unique_coords_total": sum(metrics['Unique Coords']),
                "sample_count": len(metrics['Verification']),
                "correctness_total": sum(metrics['Correctness']),
                "correctness_avg": sum(metrics['Correctness'])/len(metrics['Correctness'])
            }, f_out, indent=2)

# ----------  main  ----------
def main():

    # SAT-2 VAL
    dir1 = ""
    dir2 = ""
    dir3 = ""
    out = "data/prelim/behavior_categorization"

    model_names = {
            Path(dir1): "ViGoRL",
            Path(dir2): "Naive GRPO",
            Path(dir3): "ViGoRL no grounding"
        }

    behavior_types_to_run = ["Verification", "Backtracking", "Subgoal Setting", "Visual Regions Explored"]

    model_dirs = [Path(dir1), Path(dir2), Path(dir3)]
    
    max_side = 1260
    max_samples = 300 #100
    
    out_root   = Path(out)
    img_dir    = out_root / "images"
    out_root.mkdir(parents=True, exist_ok=True)
        
    print("\n=== Model Directories ===")
    for i, dir_path in enumerate(model_dirs, 1):
        print(f"m{i}: {dir_path} - {model_names.get(dir_path, '')}")
    
    # Save this mapping to a file for future reference
    with open(out_root / "model_info.json", "w") as f:
        json.dump({
            f"m{i+1}": {
                "path": str(path),
                "description": model_names.get(path, "")
            } for i, path in enumerate(model_dirs)
        }, f, indent=2)

    # Load roll‑outs from both models
    conditions = []
    for dir_path in model_dirs:
        conditions.append(load_correct_rollouts(dir_path, img_dir, f"m{len(conditions)+1}",
                                max_side=max_side))

    common_ids = sorted(conditions[0])
    for c in conditions[1:]:
        common_ids = sorted(set(common_ids) & set(c))
    common_ids = list(common_ids)
    random.shuffle(common_ids)          # random task order

    out_path = out_root / "data.jsonl"

    # Create lists to store entries for each condition
    entries = [[] for _ in range(len(conditions))]
    
    # Create a list of all tasks to process
    tasks = []
    for idx, rid in enumerate(common_ids[:max_samples]):
        for c_idx, c in enumerate(conditions):
            # Store the index for later use
            tasks.append((rid, c, c_idx, idx, behavior_types_to_run))
    
    results = []
    with tqdm(total=len(tasks), desc="Processing items") as pbar:
        # Use fewer workers to reduce load on API
        with ProcessPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_item, *task) for task in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing task: {e}")
                pbar.update(1)
    
    # Organize results back into the entries list
    for result in results:
        c_idx = result["c_idx"]
        entry = result["entry"]
        entries[c_idx].append(entry)

    # Define model tags to match the order of conditions
    model_tags = ["m1", "m2", "m3"]
    
    for c_idx, c in enumerate(conditions):
        with (out_root / f"data_{model_tags[c_idx]}.jsonl").open("w") as fout:
            for entry in entries[c_idx]:
                fout.write(json.dumps(entry) + "\n")

    print(f"✓ Wrote {len(common_ids)} paired samples to {out_path}")

    calculate_and_print_metrics(out_root, model_tags, model_dirs)

if __name__ == "__main__":
    main()
