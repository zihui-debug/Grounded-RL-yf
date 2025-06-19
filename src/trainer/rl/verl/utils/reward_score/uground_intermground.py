import re

from mathruler.grader import grade_answer
import numpy as np
import ast
import math

# def uground_format_reward(predict_str: str) -> float:
#     # First, extract all content between <think> tags
#     think_tags = re.findall(r"<think>(.*?)</think>", predict_str, re.DOTALL)
    
#     # Check if there are at least 2 think tags
#     if len(think_tags) < 2:
#         return 0.0
    
#     # Check that each <think> tag contains coordinates in the format (x,y)
#     for thought in think_tags:
#         # Look for coordinates at the end of the thought
#         coord_match = re.search(r"\([0-9]+,\s*[0-9]+\)", thought)
#         if not coord_match:
#             return 0.0
    
#     # Check for answer tag with proper format
#     answer_pattern = r"<answer>\s*\([0-9]+,\s*[0-9]+\)\s*</answer>"
#     answer_match = re.search(answer_pattern, predict_str, re.DOTALL)
#     if not answer_match:
#         return 0.0
    
#     # Extract the answer content
#     content_match = re.search(r"<answer>\s*(\([0-9]+,\s*[0-9]+\))\s*</answer>", predict_str, re.DOTALL)
#     if not content_match:
#         return 0.0
    
#     answer_content = content_match.group(1).strip()
    
#     # Try to parse the answer as a tuple of two integers
#     try:
#         parsed_tuple = ast.literal_eval(answer_content)
        
#         # Check if it's a tuple with exactly 2 integers
#         if (isinstance(parsed_tuple, tuple) and 
#             len(parsed_tuple) == 2 and 
#             all(isinstance(x, int) for x in parsed_tuple)):
#             return 1.0
#     except (ValueError, SyntaxError, TypeError):
#         pass
    
#     return 0.0

def uground_format_reward(predict_str: str) -> float:
    """
    Validates that the response follows the required format:
    - A single <think> block containing reasoning steps grounded with (x, y) coordinates.
    - Each reasoning step must be followed by coordinates in (x, y) format.
    - A single <answer> block containing the final (x, y) coordinate.
    - No irrelevant text outside of the required tags.
    
    Returns:
    - 1.0 if the format is valid, 0.0 otherwise.
    """
    
    errors = []

    # Ensure a single <think> block exists
    think_matches = re.findall(r'<think>(.*?)</think>', predict_str, re.DOTALL)
    if len(think_matches) != 1:
        errors.append("Error: There must be exactly one <think> block.")
    
    # Extract reasoning text inside <think> and check coordinate grounding
    if think_matches:
        think_content = think_matches[0].strip()
        
        # First check: ensure at least 2 coordinates total (minimum for proper grounding)
        coordinates = re.findall(r'\(\d+,\s*\d+\)', think_content)
        if len(coordinates) < 2:
            errors.append("Error: <think> block must contain at least two coordinate references.")
        
        # Break the content into reasoning steps based on sentence structure
        # Consider a reasoning step as content that ends with coordinates
        # This regex finds text segments ending with coordinates (allowing for ending punctuation)
        reasoning_steps = re.findall(r'[^.!?]*?(?:\(\d+,\s*\d+\))[.!?\s]*', think_content)
        
        # Check if any text is not covered by these reasoning steps
        remaining_text = think_content
        for step in reasoning_steps:
            remaining_text = remaining_text.replace(step, '', 1)
        
        # After removing all valid steps, check if any substantial text remains
        remaining_text = remaining_text.strip()
        if remaining_text and len(remaining_text) > 10:  # Allow for minor spacing/formatting differences
            errors.append(f"Error: Ungrounded reasoning text found: '{remaining_text[:50]}...'")
    
    # Ensure a single <answer> block with valid coordinates
    answer_matches = re.findall(r'<answer>\s*\((\d+),\s*(\d+)\)\s*</answer>', predict_str)
    if len(answer_matches) != 1:
        errors.append("Error: There must be exactly one <answer> block containing a valid (x, y) coordinate.")

    # NEW CHECK: Ensure no irrelevant text outside of the required tags
    # Remove the <think> and <answer> blocks from the text
    cleaned_text = predict_str
    if think_matches:
        think_block = re.search(r'<think>.*?</think>', predict_str, re.DOTALL)
        if think_block:
            cleaned_text = cleaned_text.replace(think_block.group(0), '', 1)
    
    answer_block = re.search(r'<answer>.*?</answer>', cleaned_text, re.DOTALL)
    if answer_block:
        cleaned_text = cleaned_text.replace(answer_block.group(0), '', 1)
    
    # Check if any non-whitespace content remains
    remaining_content = cleaned_text.strip()
    if remaining_content:
        errors.append(f"Error: Irrelevant text found outside of required tags: '{remaining_content[:50]}...'")

    if errors:
        return 0.0
    return 1.0


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


def uground_intermediate_ground_compute_score(predict_str: str, ground_truth: str) -> float:
    reward_dict = {}
    format_reward = uground_format_reward(predict_str)
    if format_reward == 0.0:
        acc_reward = 0.0
    else:
        acc_reward = uground_accuracy_reward(predict_str, ground_truth)
    reward = acc_reward + format_reward
    reward /= 2
    reward_dict["acc_reward"] = acc_reward
    reward_dict["format_reward"] = format_reward
    reward_dict["reward"] = reward
    return reward_dict

if __name__ == "__main__":
    # Example Test Cases
    response_correct = """
    <think> 
    First, I locate the general area described, which appears to be in the top-left quadrant of the image (120, 200). Next, I refine my focus based on color and shape, narrowing it to a more defined structure (135, 215). Finally, I confirm the central point of the most relevant object for precise targeting (140, 220).
    </think>
    <answer> (140, 220) </answer>
    """

    response_missing_coordinates = """
    <think> 
    First, I locate the general area described, which appears to be in the top-left quadrant of the image. Next, I refine my focus based on color and shape, narrowing it to a more defined structure (135, 215). Finally, I confirm the central point of the most relevant object for precise targeting.
    </think>
    <answer> (140, 220) </answer>
    """

    response_multiple_think_blocks = """
    <think> 
    First, I locate the general area (120, 200).
    </think>
    <think>
    Then, I refine my focus (135, 215).
    </think>
    <answer> (140, 220) </answer>
    """

    # Running validation
    print(uground_format_reward(response_correct))  # Expected: (True, "Response format is valid.")
    print(uground_format_reward(response_missing_coordinates))  # Expected: Error for missing coordinates
    print(uground_format_reward(response_multiple_think_blocks))  # Expected: Error for multiple <think> blocks
