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
        
    return 1.0

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
    format = format_reward(predict_str)
    accuracy = accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.5 * accuracy + 0.5 * format,
        "format": format,
        "accuracy": accuracy,
    }
