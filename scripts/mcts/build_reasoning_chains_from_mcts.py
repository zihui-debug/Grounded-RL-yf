import json
from typing import List, Dict, Any
import os
import random
from PIL import Image
from io import BytesIO
import base64
import wandb
import ast
from PIL import ImageDraw
import re
random.seed(42)

# ---------------------------------------------------------------------------
# 1) Helpers to build reasoning chains from your MCTS data
# ---------------------------------------------------------------------------
def process_text_chain(chain: List[str]) -> (str, str):
    """
    1. Removes first line of chain if it contains the word "<image>"
    2. Removes <think>, </think>, <answer>, </answer>
    3. Joins the chain together
    4. Returns (joined_chain, final_answer)
    """

    if chain and (chain[0].startswith("<image>") or chain[0].endswith("<image>")):
        chain = chain[1:]

    final_answer = chain[-1]
    final_answer = final_answer.replace("<answer>", "").replace("</answer>", "").strip()
    chain = chain[:-1]

    # Remove any <think>, </think>, etc. from the lines
    cleaned = []
    for line in chain:
        line = line.replace("<think>", "").replace("</think>", "")
        line = line.replace("<answer>", "").replace("</answer>", "")
        cleaned.append(line.strip())

    joined_chain = " ".join(cleaned)
    return joined_chain, final_answer


def build_reasoning_chains_from_rollouts(
    node_data: Dict[str, Any],
    backtrack_message: str = "Wait, this seems off. Let's try something else.",
    thought_start_tag: str = "<think>",
    thought_end_tag: str = "</think>",
    answer_start_tag: str = "<answer>",
    answer_end_tag: str = "</answer>",
) -> List[str]:
    """
    Return all possible reasoning chains from this node (recursively) as strings.
    Each chain includes wrong attempts (if any) plus a backtrack message, then a correct attempt.
    """
    rollouts = node_data.get("rollouts", [])
    correct_rollouts = []
    wrong_rollouts = []
    for r in rollouts:
        if r["reward"] >= 1.0:
            correct_rollouts.append(r)
        else:
            wrong_rollouts.append(r)

    child_nodes = node_data.get("children", [])
    is_terminal = node_data.get("is_terminal", False)

    all_chains = []

    # 1) Build chains from ephemeral rollouts at this node
    #    (Wrong -> backtrack -> Correct) and also purely Correct
    for wrong_r in wrong_rollouts:
        wrong_chain, _ = process_text_chain(wrong_r["ephemeral_texts"])
        # Insert a backtrack line after the wrong chain
        wrong_chain += f"\n{backtrack_message}"
        if correct_rollouts:
            for correct_r in correct_rollouts:
                correct_chain, correct_ans = process_text_chain(correct_r["ephemeral_texts"])
                combined_chain = wrong_chain + "\n" + correct_chain
                # Format it with <think> ... </think> plus <answer> ... </answer>:
                combined_chain = (
                    f"{thought_start_tag}\n{combined_chain}\n{thought_end_tag}\n"
                    f"{answer_start_tag} {correct_ans} {answer_end_tag}"
                )
                all_chains.append(combined_chain)

    for correct_r in correct_rollouts:
        chain_text, final_ans = process_text_chain(correct_r["ephemeral_texts"])
        chain_text = (
            f"{thought_start_tag}\n{chain_text}\n{thought_end_tag}\n"
            f"{answer_start_tag} {final_ans} {answer_end_tag}"
        )
        all_chains.append(chain_text)

    # 2) Recurse into children if not terminal
    if not is_terminal:
        for child in child_nodes:
            child_chains = build_reasoning_chains_from_rollouts(child, backtrack_message)
            all_chains.extend(child_chains)

    return all_chains


def build_all_reasoning_for_sample(sample_json: Dict[str, Any]) -> (List[str], float):
    """
    Build all possible reasoning chains from the top-level 'tree' in sample_json.
    Returns (list_of_chains, root_node_value).
    """
    root_node = sample_json["tree"]
    root_node_value = root_node.get("value", 0.0)
    chains = build_reasoning_chains_from_rollouts(root_node)
    # Deduplicate if needed:
    unique_chains = list(set(chains))
    return unique_chains, root_node_value

# ---------------------------------------------------------------------------
# 2) Helper to extract intermediate points from <think> text
# ---------------------------------------------------------------------------
def extract_think_points(chain_text: str):
    """
    Finds all <think>...</think> sections in chain_text, then extracts
    every coordinate (x, y) from those sections in order.
    Returns a list of (x, y, index).
    """
    # Find all <think> ... </think> blocks (could be multiple if there's a backtrack)
    think_blocks = re.findall(r"<think>(.*?)</think>", chain_text, flags=re.DOTALL)
    all_points = []
    point_idx = 1

    for block in think_blocks:
        # Find coords of form (123, 456) or ( 123 , 456 )
        coords = re.findall(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", block)
        for (x_str, y_str) in coords:
            try:
                x = float(x_str)
                y = float(y_str)
                all_points.append((x, y, point_idx))
                point_idx += 1
            except:
                pass  # if there's a parse error, skip

    return all_points

# ---------------------------------------------------------------------------
# 2) Main script
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # Adjust this as needed
    data_str = "path/to/rollouts" # TODO: change this to the path to the rollouts
    log_to_wandb = True
    prompt_type = "web_grounding" # "web_grounding", "spatial", "web_action"

    system_prompt_web_grounding = "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant systematically reasons through the problem step by step, verifying each step and grounding every step to a specific point in the image.\n\nAll reasoning processes must be enclosed within a single set of '<think>' tags, with each reasoning step explicitly referencing a coordinate:\n\n<think>\n[Reasoning text with grounded points inline] (x1, y1). [Further reasoning] (x2, y2), [Final refinement] (x3, y3).\n</think>\n\nThe final answer should be enclosed in '<answer>' tags in the format:\n<answer> (xf, yf) </answer>\n\nYour task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.\n- Aim to point to the center or a representative point within the described area/element/object as accurately as possible.\n- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.\n- The final output should be the single most precise coordinate for the requested element.\n- The Assistant should verify each step and check multiple possible solutions before selecting the final answer."

    system_prompt_spatial="A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant systematically reasons through the problem step by step by checking and verifying possible solutions and image regions, while grounding reasoning steps to specific objects and their relationships in the image using (x,y) coordinates. There may be one image or two images concatenated together, in which case the Assistant must compare the spatial relationships between the two images.\n\nAll reasoning processes must be enclosed within a single set of '<think>' tags, and reasoning steps must include specific reference coordinates:\n\nFor example, <think>\n{Reasoning text}. {Further reasoning text} {more reasoning} \n</think>\n\nThe final answer should be enclosed in '<answer>' tags in the format:\n<answer> {text of selected answer choice} </answer>\n\nThe Assistant must help the user identify the correct answer choice from the options provided.\n-Your answer should be the **exact text** of the selected answer option, without additional explanations or reasoning or the option text. For example, if the answer is A. right , your response should just be <answer>right</answer> (not <answer>A. right</answer>).\n-If the correct answer is unclear, select the most relevant option based on the spatial relationships and dynamics within the image.\n- The Assistant should verify each step and check multiple possible solutions before selecting the final answer."

    system_prompt_web_action="""You are a helpful Assistant tasked with navigating a web browser. These tasks will be accomplished through the use of specific actions you can issue. Your task is to choose the action that makes the most progress towards an objective. You should systematically reason through the problem step by step by checking and verifying possible actions and webpage regions, while grounding reasoning steps to specific (x, y) points in the image:\nEach reasoning step must be enclosed within '<think>' tags and reference exactly one specific coordinate (x, y):\n<think>\n[Reasoning text with grounded points inline] (x_1, y_1). [Further reasoning] (x_2, y_2), ..., [Final reasoning] (x_n, y_n).\n</think>\nWhen ready to provide the final answer, enclose it within '<answer>' tags:\n<answer> {action} </answer>\n- Each reasoning step must explicitly describe and evaluate the region’s relevance to the objective and proposing an action.\n- Never repeat coordinates from previous steps.\n- Look at diverse webpage regions to figure out which action should be taken.\n- Verify your selection by examining multiple possible solutions.\n\n**Inputs**\nHere's the information you'll have:\n1. OBJECTIVE: This is the task you are trying to complete.\n2. The web page screenshot: This is a screenshot of the current webpage you are on, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.\n3. PREVIOUS ACTIONS: This is the actions that you have performed prior to getting to the current page, but instead of the button id, the button text of the actions taken on the previously navigated pages are provided.\n\n**Action Space**\nYou can take the following actions:\n1. ```click [id]```: This action clicks on an element with a specific id on the webpage.\n2. ```type [id] [content]```: Use this to type the content into the field with id. By default, typing the content simulates pressing the "Enter" key afterward to submit the text.\n3. ```scroll [down]```: Scroll the page down.\n4. ```go_back```: Navigate to the previously viewed page.\n5. ```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If no answer is required, output empty brackets.\n\n**Guidelines**\nTo be successful, it is very important to follow the following rules:\n2. Generate the final action in the correct format. For example, '<answer> click [1234] </answer>'.\n3. Issue the stop action (i.e. stop [answer]) when you think you have achieved the objective. Don't generate anything after stop.\n4. In your final answer, you should only output a single action and should never output a prediction involving taking multiple actions."""

    system_prompt_qa="""You are an assistant answering a visual question by reasoning through image regions. You must systematically examine and verify relevant regions of the image, grounding each reasoning step to a specific (x, y) coordinate.

All reasoning steps must be enclosed within '<think>' tags and each step must start with an absolute (x, y) coordinate, followed by a description and evaluation of the corresponding image region. 
When confident in the answer, provide it inside '<answer>' tags:

<think>\n[Reasoning text with grounded points inline] (x_1, y_1). [Further reasoning] (x_2, y_2), ..., [Final reasoning] (x_n, y_n).\n</think>\nWhen ready to provide the final answer, enclose it within '<answer>' tags:\n<answer> {final answer} </answer>

Instructions:
- Always begin a reasoning step with an (x, y) coordinate.
- Coordinates must be absolute image points formatted as integers: (x, y).
- Regions refer to spatially distinct parts of the image: quadrants (e.g., top-left), discrete objects (e.g., bottle), or structural zones (e.g., background).
- Explore diverse, even less likely, regions early on to ensure broad coverage.
- Reason about a region's relevance to the question and—if visible—its relation to prior steps.
- Aim to choose accurate, representative coordinates within each region."""

    if prompt_type == "web_grounding":
        system_prompt = system_prompt_web_grounding
    elif prompt_type == "spatial":
        system_prompt = system_prompt_spatial
    elif prompt_type == "web_action":
        system_prompt = system_prompt_web_action
    elif prompt_type == "vstar":
        # NOTE: V* single turn not tested yet
        system_prompt = system_prompt_qa
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")

    # What fraction or number of val samples do we want:
    val_size = 0.05  # 10% for val, for example
    draw_points = True
    max_samples_per_file = 10000

    # -----------------------------------------------------------------------

    # 1) Gather all .jsonl files from data_str (if it's a folder)
    if os.path.isdir(data_str):
        data_files = [os.path.join(data_str, f) for f in os.listdir(data_str) if f.endswith(".jsonl")]
    else:
        data_files = [data_str]

    # We'll store:
    # - all textual chains to write to a .txt file
    # - for each chain, we create a separate SFT entry
    all_chains_text = []
    sft_entries_train = []
    sft_entries_val = []
    all_images_processed = set()
    chain_global_id = 0
    correct_count = 0

    # split data_files into two lists: train and val based on the val_size
    train_files = data_files[:int(len(data_files) * (1 - val_size))]
    val_files = data_files[int(len(data_files) * (1 - val_size)):]

    for i, data_file in enumerate(data_files):
        with open(data_file, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line:
                continue
            data_json = json.loads(line)

        error = data_json.get("error", "")
        if error:
            print(f"error processing {data_file}: {error}")
            continue

        question = data_json.get("question", "")
        image = data_json.get("image", "")
        system_prompt_MCTS = data_json.get("system_prompt", "")
        true_answer = data_json.get("true_answer", "")
        # tree, etc...
        chains, root_node_value = build_all_reasoning_for_sample(data_json)
        if root_node_value > 0:
            correct_count += 1

        if len(chains) > max_samples_per_file:
            chains = random.sample(chains, max_samples_per_file)

        # For each chain, we treat it as a separate SFT training example
        for c in chains:
            # We'll store in a text file
            all_chains_text.append(c)

            # check if more than 1 "<image>" in question
            if c.count("<image>") > 0:
                # remove all "<image>" tags
                c = c.replace("<image>", "").replace("</image>", "")

            # SFT-style record:
            #   "messages": [
            #       {"role": "system", "content": system_prompt},
            #       {"role": "user",   "content": question},
            #       {"role": "assistant", "content": c}
            #   ]
            #   "images": [image]
            chain_entry = {
                "id": f"{i}_{chain_global_id}",
                "metadata": {},
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": c
                    }
                ],
                "images": [image],
                "gt_answer": true_answer,
            }
            if data_file in train_files:
                sft_entries_train.append(chain_entry)
                if image not in all_images_processed:
                    all_images_processed.add(image)
            else:
                sft_entries_val.append(chain_entry)
                if image not in all_images_processed:
                    all_images_processed.add(image)
            chain_global_id += 1
        
    total_samples = len(data_files)
    if total_samples == 0:
        print("No samples found. Exiting.")
        exit()

    accuracy = correct_count / total_samples
    print(f"Processed {total_samples} input JSONL files.")
    print(f"Root node correctness ratio: {accuracy:.3f} ({correct_count}/{total_samples})")

    # 2) Write out all chains to a text file
    base_dir = os.path.dirname(data_files[0]) if data_files else os.path.dirname(data_str)
    out_dir = os.path.join(base_dir, "reasoning_chains")
    os.makedirs(out_dir, exist_ok=True)

    txt_path = os.path.join(out_dir, "reasoning_chains.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        sep = "\n\n----------------------\n\n----------------------\n\n"
        for ch in all_chains_text:
            f.write(ch + sep)
    print(f"Wrote {len(all_chains_text)} total chains to {txt_path}")

    # 3) Train/Val split at the chain level has been done during processing

    train_json_path = os.path.join(out_dir, "reasoning_chains_train.json")
    val_json_path = os.path.join(out_dir, "reasoning_chains_val.json")

    with open(train_json_path, "w", encoding="utf-8") as f:
        json.dump(sft_entries_train, f, indent=2)
    print(f"Saved train SFT data to: {train_json_path}  (count={len(sft_entries_train)})")

    with open(val_json_path, "w", encoding="utf-8") as f:
        json.dump(sft_entries_val, f, indent=2)
    print(f"Saved val SFT data to: {val_json_path}  (count={len(sft_entries_val)})")

    # 4) (Optional) Log to W&B
    if log_to_wandb:
        wandb.init(project="vlm-search", name="mcts_reasoning_chains_sft")
        max_samples = 50
        # Shuffle the data
        sft_entries = sft_entries_train + sft_entries_val
        random_indices = random.sample(range(len(sft_entries)), min(max_samples, len(sft_entries)))
        html_content = "<html><body>"
        for idx in random_indices:
            entry = sft_entries[idx]
            question = entry["messages"][1]["content"]
            chain_text = entry["messages"][2]["content"]
            image_path = entry["images"][0]
            true_answer = entry["gt_answer"]
            # Extract final answer from <answer> tags, if present
            final_answer = ""
            if "<answer>" in chain_text and "</answer>" in chain_text:
                try:
                    final_answer = chain_text.split("<answer>")[1].split("</answer>")[0].strip()
                except:
                    pass

            img_html = ""
            if image_path and os.path.exists(image_path):
                with Image.open(image_path) as pil_img:
                    if draw_points:
                        # Prepare to draw on the image
                        draw = ImageDraw.Draw(pil_img)
                        circle_radius = 7

                        # 1) Draw intermediate points from <think> text
                        intermediate_points = extract_think_points(chain_text)
                        for (x, y, point_num) in intermediate_points:
                            draw.ellipse(
                                (x - circle_radius, y - circle_radius,
                                 x + circle_radius, y + circle_radius),
                                fill="green",
                            )
                            draw.text((x + 15, y), str(point_num), fill="green")

                        # 2) If final answer parse is valid, draw it in red
                        try:
                            pred_ans = ast.literal_eval(final_answer)
                            if (
                                isinstance(pred_ans, (list, tuple)) and
                                len(pred_ans) == 2
                            ):
                                px, py = float(pred_ans[0]), float(pred_ans[1])
                                draw.ellipse(
                                    (px - circle_radius, py - circle_radius,
                                     px + circle_radius, py + circle_radius),
                                    fill="red",
                                )
                                draw.text((px + 15, py), "PRED", fill="red")
                        except:
                            pass

                        # 3) If ground truth is provided, draw it in blue
                        try:
                            gt_ans = ast.literal_eval(true_answer)
                            if (
                                isinstance(gt_ans, (list, tuple)) and
                                len(gt_ans) == 2
                            ):
                                gx, gy = float(gt_ans[0]), float(gt_ans[1])
                                draw.ellipse(
                                    (gx - circle_radius, gy - circle_radius,
                                     gx + circle_radius, gy + circle_radius),
                                    fill="blue",
                                )
                                draw.text((gx + 15, gy), "GT", fill="blue")
                        except:
                            pass

                    # Convert to RGB for displaying
                    if pil_img.mode != "RGB":
                        pil_img = pil_img.convert("RGB")

                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    enc = base64.b64encode(buf.getvalue()).decode("utf-8")
                    img_html = f'<img src="data:image/png;base64,{enc}" style="max-width:600px;" />'
                # except Exception as e:
                #     print(f"Could not open {image_path}: {e}")

            row_html = f"""
            <div style="border:1px solid #ddd; padding:10px; margin:10px 0;">
                <p><b>Question:</b> {question}</p>
                {img_html}
                <p><b>Chain:</b> {chain_text.replace('<','&lt;').replace('>','&gt;')}</p>
            </div>
            """
            html_content += row_html

        html_content += "</body></html>"
        wandb.log({"mcts_reasoning_chains": wandb.Html(html_content)})
        wandb.finish()
