import re

from mathruler.grader import grade_answer
import numpy as np
import ast
import math

def sat_format_reward(predict_str: str) -> float:
    # Check for proper format with think and answer tags
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    format_match = re.fullmatch(pattern, predict_str, re.DOTALL)
    
    if not format_match:
        return 0.0
    
    # Extract the answer content
    content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if not content_match:
        return 0.0
    
    answer_content = content_match.group(1).strip()

    # # Check if the answer starts with '(' and ends with ')'
    # if re.match(r'^[A-Za-z][\.\)]\s*', answer_content):
    #     return 0.0
    
    # If we have content and it's not prefixed with an answer choice, return 1.0
    if answer_content:
        return 1.0
    
    return 0.0

def sat_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    Attempts to parse gt_ans and pred_ans as tuples containing integers. 
    If parsing succeeds and both tuples are valid (all ints), computes their
    Euclidean distance and returns 1.0 if the distance is less than 
    point_matching_threshold, otherwise 0.0. If parsing or validation fails, returns 0.0.
    """

    ground_truth = ground_truth.strip()
    content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
    pred_answer = content_match.group(1).strip() if content_match else predict_str.strip()

    # Attempt to parse the strings into tuple objects
    choice_match = re.match(r'^[A-Za-z][\.\)]\s*(.*)', pred_answer)
    if choice_match:
        # Extract the actual answer text after the choice prefix
        pred_answer = choice_match.group(1).strip()
    
    # Compare the answers (case insensitive)
    return 1.0 if pred_answer.lower() == ground_truth.lower() else 0.0

def coordinate_reward(predict_str: str) -> float:
    """
    Checks if the thinking block contains at least 2 different bounding boxes
    in format (x1, y1, x2, y2) or [x1, y1, x2, y2], where coordinates can be integers or decimals.

    INPUTS:
    - predict_str: The full prediction string including think and answer tags.

    OUTPUTS:
    - score: 1.0 if at least 2 different bounding boxes are found in the thinking block, 0.0 otherwise.
    """
    # Extract the thinking content
    think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    if not think_match:
        return 0.0

    thinking_content = think_match.group(1)

    # Find all bounding boxes in the format (x1, y1, x2, y2) or [x1, y1, x2, y2]
    # 支持整数或小数，支持任意空格
    bbox_pattern = re.compile(
        r"[\(\[]\s*"
        r"(\d+(?:\.\d+)?)\s*,\s*"
        r"(\d+(?:\.\d+)?)\s*,\s*"
        r"(\d+(?:\.\d+)?)\s*,\s*"
        r"(\d+(?:\.\d+)?)\s*"
        r"[\)\]]"
    )
    bboxes = bbox_pattern.findall(thinking_content)

    # Convert to set of tuples (floats) to get unique bounding boxes
    unique_bboxes = set(tuple(map(float, bbox)) for bbox in bboxes)

    # Return 1.0 if there are at least 2 different bounding boxes
    return float(len(unique_bboxes) >= 2)

def sat_compute_score(predict_str: str, ground_truth: str) -> float:
    format_reward = sat_format_reward(predict_str)
    coordinate = coordinate_reward(predict_str)
    acc_reward = sat_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.5 * acc_reward + 0.5 * (format_reward * coordinate),
        "format": format_reward * coordinate,
        "accuracy": acc_reward,
    }


def point_in_bbox_format_reward(predict_str: str) -> float:
    # Check for proper format with think and answer tags
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    format_match = re.fullmatch(pattern, predict_str, re.DOTALL)
    
    if not format_match:
        return 0.0
    
    # Extract the answer content
    content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if not content_match:
        return 0.0
    
    answer_content = content_match.group(1).strip()

    # Check if the answer starts with '(' and ends with ')'
    if not (answer_content.startswith('(') and answer_content.endswith(')')):
        return 0.0
    
    # Try to parse the answer as a tuple
    try:
        parsed_tuple = ast.literal_eval(answer_content)
        
        # Check if it's a tuple with exactly 2 integers
        if (isinstance(parsed_tuple, tuple) and 
            len(parsed_tuple) == 2 and 
            all(isinstance(x, int) for x in parsed_tuple)):
            return 1.0
    except (ValueError, SyntaxError, TypeError):
        pass
    
    return 0.0

def point_in_bbox_accuracy_reward(pred_ans: str, gt_ans: str) -> float:
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


def point_in_bbox_compute_score(predict_str: str, ground_truth: str) -> float:
    format = point_in_bbox_format_reward(predict_str)
    accuracy = point_in_bbox_accuracy_reward(predict_str, ground_truth)
    coordinate = coordinate_reward(predict_str)
    return {
        "overall": 0.5 * accuracy + 0.5 * (format * coordinate),
        "format": format * coordinate,
        "accuracy": accuracy,
    }


def web_action_format_reward(predict_str: str) -> float:
    # Check for proper format with think and answer tags
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    format_match = re.fullmatch(pattern, predict_str, re.DOTALL)
    
    if not format_match:
        return 0.0
    
    # Extract the answer content
    content_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if not content_match:
        return 0.0
        
    return 1.0


def web_action_accuracy_reward(pred_ans: str, gt_ans: str) -> float:
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

    try:
        if "go_back" in pred_ans:
            return 1.0 if "go_back" in gt_ans else 0.0
        elif "stop" in pred_ans:
            return 1.0 if "stop" in gt_ans else 0.0
        elif "scroll" in pred_ans:
            if "[down]" in pred_ans and "[down]" in gt_ans:
                return 1.0
            elif "[up]" in pred_ans and "[up]" in gt_ans:
                return 1.0
            else:
                return 0.0
        elif "click" in pred_ans:
            # get integer wrapped in square brackets for pred_ans and gt_ans
            pred_match = re.search(r'\[(\d+)\]', pred_ans)
            gt_match = re.search(r'\[(\d+)\]', gt_ans)
            
            if not pred_match or not gt_match:
                return 0.0
                
            element_id = pred_match.group(1)
            gt_element_id = gt_match.group(1)
            action_match = int("click" in gt_ans) * 0.5
            element_id_match = int(element_id.strip() == gt_element_id.strip()) * 0.5
            return action_match + element_id_match
        elif "type" in pred_ans:
            # get string wrapped in square brackets for pred_ans and gt_ans
            pred_match = re.search(r'\[(.*?)\]', pred_ans)
            gt_match = re.search(r'\[(.*?)\]', gt_ans)
            
            if not pred_match or not gt_match:
                return 0.0
                
            element_content = pred_match.group(1)
            gt_element_content = gt_match.group(1)
            action_match = int("type" in gt_ans) * 0.5
            element_content_match = int(element_content.strip() == gt_element_content.strip()) * 0.5
            return action_match + element_content_match
        else:
            return 0.0
    except Exception as e:
        return 0.0


def web_action_compute_score(predict_str: str, ground_truth: str) -> float:
    format = web_action_format_reward(predict_str)
    coordinate = coordinate_reward(predict_str)
    # format = format * coordinate
    accuracy = web_action_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.5 * accuracy + 0.5 * (format * coordinate),
        "format": format * coordinate,
        "accuracy": accuracy,
    }


# regex to pull out only the structural tags we care about
_TAG_RE = re.compile(r"<(/?)(tool_call|observation|think|answer)>", re.IGNORECASE)
_TOOL_JSON_RE = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)

def string_match_multiturn_format_reward(predict_str: str) -> float:
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
            coord = obj.get("arguments", {}).get("bbox", None)
                
            if (not isinstance(coord, list)):
                return 0.0
            else:
                if isinstance(coord[0], list):
                    for bbox in coord:
                        if (not isinstance(box, list) or len(box) != 4 or
                            not all(isinstance(x, int) for x in box)):
                            return 0.0
                else:
                    if (not isinstance(coord, list) or len(coord) != 4 or
                        not all(isinstance(x, int) for x in coord)):
                        return 0.0
            
            # Add valid coordinate to our tracking list
            if isinstance(coord[0], list):
                for bbox in coord:
                    previous_coords.append(bbox)
            else:
                previous_coords.append(coord)
        except Exception:
            return 0.0

    # 4) validate final answer exists within answer tags
    ans_match = re.search(r"<answer>\s*(.*?)\s*</answer>", s, re.DOTALL)
    if not ans_match or not ans_match.group(1).strip():
        return 0.0
    
    # 5) base reward + bonus for extra turns with sufficient diversity
    reward = 1.0

    # # Count unique and sufficiently distant coordinates
    # unique_coords = []
    # for coord in previous_coords:
    #     # Check if this coordinate is too close to any we've already counted
    #     too_close = False
    #     for existing_coord in unique_coords:
    #         # Calculate Euclidean distance
    #         distance = math.sqrt((coord[0] - existing_coord[0])**2 + 
    #                              (coord[1] - existing_coord[1])**2)
    #         if distance < min_distance_threshold:
    #             too_close = True
    #             break
        
    #     if not too_close:
    #         unique_coords.append(coord)
    
    # # Award bonus only for unique, sufficiently distant coordinates
    # num_unique_turns = len(unique_coords)
    # if num_unique_turns > 1:
    #     reward += 0.2 * (num_unique_turns - 1)

    return reward

def string_match_multiturn_accuracy_reward(pred_ans: str, gt_ans: str) -> float:
    # Extract content from answer tags if present
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", pred_ans, re.DOTALL | re.IGNORECASE)
    pred = m.group(1).strip().lower() if m else pred_ans.strip().lower()
    gt = gt_ans.strip().lower()

    return 1.0 if pred == gt else 0.0

def string_match_multiturn_compute_score(predict_str: str, ground_truth: str) -> float:
    format = string_match_multiturn_format_reward(predict_str)
    accuracy = string_match_multiturn_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.6 * accuracy + 0.4 * format,
        "format": format,
        "accuracy": accuracy,
    }


def compute_score(predict_str: str, ground_truth: str) -> dict:
    task_type = ground_truth["task_type"]
    ground_truth = ground_truth["answer"]
    
    if task_type == "vstar" or task_type == "spatial":
        return sat_compute_score(predict_str, ground_truth)
    elif task_type == "webgrounding":
        return point_in_bbox_compute_score(predict_str, ground_truth)
    elif task_type == "webaction":
        return web_action_compute_score(predict_str, ground_truth)
    elif task_type == "vstar_multiturn":
        return string_match_multiturn_compute_score(predict_str, ground_truth)
    else:
        raise ValueError(f"Unknown task type: {task_type}")
