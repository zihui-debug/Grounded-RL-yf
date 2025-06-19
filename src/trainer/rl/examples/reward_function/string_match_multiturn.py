import re

from mathruler.grader import grade_answer
import numpy as np
import ast
import math
import json

# regex to pull out only the structural tags we care about
_TAG_RE = re.compile(r"<(/?)(tool_call|observation|think|answer)>", re.IGNORECASE)
_TOOL_JSON_RE = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)

def format_reward(predict_str: str) -> float:
    s = predict_str.strip()

    # 1) must finish with </answer> (optionally followed by one EOS token)
    if not re.search(r"</answer>\s*(<\|im_end\|>)?\s*$", s, re.DOTALL):
        return 0.0

    # 2) walk through the high‑level tag sequence to enforce grammar
    tags_iter = _TAG_RE.finditer(s)
    state = "think_open"            # expected next tag
    for m in tags_iter:
        tag = m.group(0).lower()

        if state == "tool_open":
            if tag != "<tool_call>":
                return 0.0
            state = "tool_close"

        elif state == "tool_close":
            if tag != "</tool_call>":
                return 0.0
            state = "obs_open"

        elif state == "obs_open":
            if tag != "<observation>":
                return 0.0
            state = "obs_close"

        elif state == "obs_close":
            if tag != "</observation>":
                return 0.0
            state = "think_open"

        elif state == "think_open":
            if tag != "<think>":
                return 0.0
            state = "think_close"

        elif state == "think_close":
            if tag != "</think>":
                return 0.0
            state = "post_think"

        elif state == "post_think":
            if tag == "<tool_call>":
                state = "tool_close"         # start another round
            elif tag == "<answer>":
                state = "answer_close"
            else:
                return 0.0

        elif state == "answer_close":
            if tag != "</answer>":
                return 0.0
            state = "end"

        elif state == "end":
            # no structural tags allowed after </answer>
            return 0.0

    if state != "end":
        return 0.0   # we never saw a complete <answer> … </answer> block

    # 3) validate each <tool_call> JSON and coordinate schema
    # Also track unique coordinates for reward calculation
    previous_coords = []
    min_distance_threshold = 10  # Minimum distance in pixels between coordinates
    
    for m in _TOOL_JSON_RE.finditer(s):
        try:
            obj = json.loads(m.group(1))
            coord = obj.get("arguments", {}).get("coordinate", None)
            if (not isinstance(coord, list) or len(coord) != 2 or
                not all(isinstance(x, int) for x in coord)):
                return 0.0
            
            # Add valid coordinate to our tracking list
            previous_coords.append(coord)
        except Exception:
            return 0.0

    # 4) validate final answer exists within answer tags
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", s, re.DOTALL)
    if not ans_match or not ans_match.group(1).strip():
        return 0.0
    
    # 5) base reward + bonus for extra turns with sufficient diversity
    reward = 1.0

    # Count unique and sufficiently distant coordinates
    unique_coords = []
    for coord in previous_coords:
        # Check if this coordinate is too close to any we've already counted
        too_close = False
        for existing_coord in unique_coords:
            # Calculate Euclidean distance
            distance = math.sqrt((coord[0] - existing_coord[0])**2 + 
                                 (coord[1] - existing_coord[1])**2)
            if distance < min_distance_threshold:
                too_close = True
                break
        
        if not too_close:
            unique_coords.append(coord)
    
    # Award bonus only for unique, sufficiently distant coordinates
    num_unique_turns = len(unique_coords)
    if num_unique_turns > 1:
        reward += 0.2 * (num_unique_turns - 1)

    return reward

def accuracy_reward(pred_ans: str, gt_ans: str) -> float:
    # Extract content from answer tags if present
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", pred_ans, re.DOTALL | re.IGNORECASE)
    pred = m.group(1).strip().lower() if m else pred_ans.strip().lower()
    gt = gt_ans.strip().lower()

    return 1.0 if pred == gt else 0.0

def string_match_multiturn_compute_score(predict_str: str, ground_truth: str) -> float:
    format = format_reward(predict_str)
    accuracy = accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.6 * accuracy + 0.4 * format,
        "format": format,
        "accuracy": accuracy,
    }
