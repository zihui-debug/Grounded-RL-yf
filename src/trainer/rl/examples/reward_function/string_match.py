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
    return 1.0 if pred_answer.lower() == ground_truth else 0.0


def sat_compute_score(predict_str: str, ground_truth: str) -> float:
    format_reward = sat_format_reward(predict_str)
    acc_reward = sat_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.5 * acc_reward + 0.5 * format_reward,
        "format": format_reward,
        "accuracy": acc_reward,
    }
