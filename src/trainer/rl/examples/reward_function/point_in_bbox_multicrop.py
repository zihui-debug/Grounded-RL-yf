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

    # 4) validate final answer is a tuple of two ints
    ans_match = re.search(r"<answer>\s*\(([^)]*)\)\s*</answer>", s)
    if not ans_match:
        return 0.0
    try:
        ans_tuple = ast.literal_eval("(" + ans_match.group(1).strip() + ")")
        if (not isinstance(ans_tuple, tuple) or len(ans_tuple) != 2 or
            not all(isinstance(x, int) for x in ans_tuple)):
            return 0.0
    except Exception:
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
    """
    Checks if a predicted point is within a ground truth bounding box.
    
    INPUTS:
    - gt_ans: The ground truth answer as a string representing a bounding box in format "(x1, y1, x2, y2)"
            where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
    - pred_ans: The predicted answer as a string representing a point in format "(x, y)".
    
    OUTPUTS:
    - score: 1.0 if the point is inside the bounding box, 0.0 otherwise.
    """

    gt_ans = gt_ans.strip()
    content_match = re.search(r"<answer>(.*?)</answer>", pred_ans)
    pred_ans = content_match.group(1).strip() if content_match else pred_ans.strip()

    # print(f"[INFO] gt_ans: {gt_ans}, pred_ans: {pred_ans}")

    # Attempt to parse the strings into tuple objects
    try:
        gt_bbox = tuple(ast.literal_eval(gt_ans))
        pred_point = tuple(ast.literal_eval(pred_ans))
    except (ValueError, SyntaxError, TypeError):
        # print(f"Error parsing strings; gt_ans: {gt_ans}, pred_ans: {pred_ans}")
        # If parsing fails, return 0.0
        return 0.0
    
    # Check if gt_bbox is a tuple with 4 integers (x1, y1, x2, y2)
    if not (isinstance(gt_bbox, tuple) and len(gt_bbox) == 4 and 
            all(isinstance(x, int) for x in gt_bbox)):
        # print(f"Error parsing gt_bbox; gt_bbox: {gt_bbox}")
        return 0.0
    
    # Check if pred_point is a tuple with 2 integers (x, y)
    if not (isinstance(pred_point, tuple) and len(pred_point) == 2 and 
            all(isinstance(x, int) for x in pred_point)):
        # print(f"Error parsing pred_point; pred_point: {pred_point}")
        return 0.0
    
    # Extract coordinates
    x1, y1, x2, y2 = gt_bbox

    x, y = pred_point
    
    # Check if the point is inside the bounding box
    if x1 <= x <= x2 and y1 <= y <= y2:
        return 1.0
    else:
        # print(f"Point is not inside the bounding box; x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, x: {x}, y: {y}")
        return 0.0


def point_in_bbox_multicrop_compute_score(predict_str: str, ground_truth: str) -> float:
    format = format_reward(predict_str)
    accuracy = accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.6 * accuracy + 0.4 * format,
        "format": format,
        "accuracy": accuracy,
    }
