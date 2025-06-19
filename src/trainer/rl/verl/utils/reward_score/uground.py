import re

from mathruler.grader import grade_answer
import numpy as np
import ast
import math

# def uground_format_reward(predict_str: str) -> float:
#     pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
#     match = re.fullmatch(pattern, predict_str, re.DOTALL)
#     return 1.0 if match else 0.0

def uground_format_reward(predict_str: str) -> float:
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

# def uground_accuracy_reward(predict_str: str, ground_truth: str) -> float:
#     try:
#         ground_truth = ground_truth.strip()
#         content_match = re.search(r"<answer>(.*?)</answer>", predict_str)
#         pred_answer = content_match.group(1).strip() if content_match else predict_str.strip()
#         if grade_answer(pred_answer, ground_truth):
#             return 1.0
#     except Exception:
#         pass

#     return 0.0

def reward_exponential(pred: tuple, gt: tuple, alpha=0.01):
    """
    pred, gt: (x, y) tuples
    alpha: scale factor for the exponential decay
    Returns a float in (0, 1].
    """
    dx = pred[0] - gt[0]
    dy = pred[1] - gt[1]
    dist = math.sqrt(dx*dx + dy*dy)
    r = np.exp(-alpha * dist)
    return float(r)

def uground_accuracy_reward(predict_str: str, ground_truth: str) -> float:
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
    try:
        gt_tuple = ast.literal_eval(ground_truth)
        pred_tuple = ast.literal_eval(pred_answer)
    except (ValueError, SyntaxError, TypeError):
        return 0.0

    # Check if they are tuples
    if not (isinstance(gt_tuple, tuple) and isinstance(pred_tuple, tuple)):
        return 0.0

    # Check if all elements in both tuples are integers
    if not all(isinstance(x, int) for x in gt_tuple) or not all(isinstance(x, int) for x in pred_tuple):
        return 0.0

    # Check that they have the same size for distance calculation
    if len(gt_tuple) != len(pred_tuple):
        return 0.0

    reward = reward_exponential(pred_tuple, gt_tuple)

    return reward


def uground_compute_score(predict_str: str, ground_truth: str) -> float:
    reward_dict = {}
    format_reward = uground_format_reward(predict_str)
    acc_reward = uground_accuracy_reward(predict_str, ground_truth)
    reward = acc_reward + format_reward
    reward /= 2
    reward_dict["acc_reward"] = acc_reward
    reward_dict["format_reward"] = format_reward
    reward_dict["reward"] = reward
    return reward_dict
