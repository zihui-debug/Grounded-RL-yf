import re

from mathruler.grader import grade_answer
import numpy as np
import ast
import math

def format_reward(predict_str: str) -> float:
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

def coordinate_reward(predict_str: str) -> float:
    """
    Checks if the thinking block contains at least 2 different coordinate pairs in format (x, y).
    
    INPUTS:
    - predict_str: The full prediction string including think and answer tags.
    
    OUTPUTS:
    - score: 1.0 if at least 2 different coordinate pairs are found in the thinking block, 0.0 otherwise.
    """
    # Extract the thinking content
    think_match = re.search(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    if not think_match:
        return 0.0
    
    thinking_content = think_match.group(1)
    
    # Find all coordinate pairs in the format (x, y)
    coordinates = re.findall(r"\((\d+)\s*,\s*(\d+)\)", thinking_content)
    
    # Convert to set of tuples to get unique coordinates
    unique_coordinates = set(tuple(map(int, coord)) for coord in coordinates)
    
    # Return 1.0 if there are at least 1 different coordinate pairs
    return int(len(unique_coordinates) >= 1)

def point_in_bbox_compute_score(predict_str: str, ground_truth: str) -> float:
    format = format_reward(predict_str)
    accuracy = accuracy_reward(predict_str, ground_truth)
    coordinate = coordinate_reward(predict_str)
    return {
        "overall": 0.5 * accuracy + 0.5 * (format * coordinate),
        "format": format * coordinate,
        "accuracy": accuracy,
    }
