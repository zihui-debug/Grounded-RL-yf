import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import json
from typing import List, Tuple, Optional, Dict, Union
import re

def _remove_punctuation_spaces(ans: str) -> str:

    """
    Removes punctuation from the answer and leading and trailing spaces.
    
    INPUTS:
    - ans: The answer.
    
    OUTPUTS:
    - ans_filtered: The answer without punctuation.
    
    """

    #remove any spaces 
    ans = ans.strip()

    #remove any punctuation marks
    ans_filtered = ans.replace(".", "")
    ans_filtered = ans_filtered.replace("?", "")
    ans_filtered = ans_filtered.replace("!", "")
    ans_filtered = ans_filtered.replace(",", "")
    ans_filtered = ans_filtered.replace(";", "")
    ans_filtered = ans_filtered.replace(":", "")
    ans_filtered = ans_filtered.replace("'", "")
    ans_filtered = ans_filtered.replace('"', "")
    ans_filtered = ans_filtered.replace("(", "")
    ans_filtered = ans_filtered.replace(")", "")
    ans_filtered = ans_filtered.replace("[", "")
    ans_filtered = ans_filtered.replace("]", "")
    ans_filtered = ans_filtered.replace("{", "")
    ans_filtered = ans_filtered.replace("}", "")
    ans_filtered = ans_filtered.replace("<", "")
    ans_filtered = ans_filtered.replace(">", "")
    ans_filtered = ans_filtered.replace("/", "")
    ans_filtered = ans_filtered.replace("\\", "")
    ans_filtered = ans_filtered.replace("|", "")
    ans_filtered = ans_filtered.replace("=", "")
    ans_filtered = ans_filtered.replace("+", "")
    ans_filtered = ans_filtered.replace("-", "")
    ans_filtered = ans_filtered.replace("_", "")
    ans_filtered = ans_filtered.replace("*", "")
    ans_filtered = ans_filtered.replace("&", "")
    ans_filtered = ans_filtered.replace("^", "")
    ans_filtered = ans_filtered.replace("%", "")
    ans_filtered = ans_filtered.replace("$", "")
    ans_filtered = ans_filtered.replace("#", "")
    ans_filtered = ans_filtered.replace("@", "")
    ans_filtered = ans_filtered.replace("`", "")
    ans_filtered = ans_filtered.replace("~", "")
    ans_filtered = ans_filtered.replace(" ", "")
    ans_filtered = ans_filtered.strip()
    ans_filtered = ans_filtered.lower()

    return ans_filtered

def _string_matching(gt_ans: str, pred_ans: str) -> float:

    """
    Judges the predicted answer and returns a float score. 1.0 for the correct answer, 0.0 for the wrong answer.
    
    INPUTS:
    - gt_ans: The ground truth answer.
    - pred_ans: The predicted answer.
    
    OUTPUTS:
    - score: The score of the predicted answer.
    
    """

    #Accounitng for the fact that the answers may be in different cases
    gt_ans = gt_ans.lower()
    pred_ans = pred_ans.lower()

    pred_ans = _remove_punctuation_spaces(pred_ans)
    gt_ans = _remove_punctuation_spaces(gt_ans)

    if gt_ans == pred_ans or gt_ans in pred_ans:
        # logger.debug(f"Judge response : 1.0")
        return 1.0
    else:
        # logger.debug(f"Judge response : 0.0")
        return 0.0

def calcu_acc(gt_jsonl, result_dir):
    gt = []
    with open(gt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            gt.append(json.loads(line.strip()))
    result = []
    for filename in os.listdir(result_dir):
        if filename.startswith("rollouts") and filename.endswith(".jsonl"):
            file_path = os.path.join(result_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    result.append(json.loads(line.strip()))
    correct = 0
    for item in result:
        # if item['judge_score']>0:
        #     correct+=1
        gt_ans = item["true_answer"]
        pred_ans = item["final_answer"]
        score = _string_matching(gt_ans, pred_ans)
        if score > 0:
            correct += 1
        else:
            print(item['id'])
            print(f"GT: {gt_ans} | Pred: {pred_ans}")
    # acc = correct / len(gt)
    acc = correct / len(result)
    print(f'acc:  {acc}')
    print(f'result length:  {len(result)}')
    print(f'gt length:  {len(gt)}')


def calcu_acc_and_save_wrong_samples(gt_jsonl, result_dir):
    gt = []
    with open(gt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            gt.append(json.loads(line.strip()))
    result = []
    for filename in os.listdir(result_dir):
        if filename.startswith("rollouts") and filename.endswith(".jsonl"):
            file_path = os.path.join(result_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    result.append(json.loads(line.strip()))
    correct = 0
    wrong_samples = []
    for item in result:
        # if item['judge_score']>0:
        #     correct+=1
        gt_ans = item["true_answer"]
        pred_ans = item["final_answer"]
        score = _string_matching(gt_ans, pred_ans)
        if score > 0:
            correct += 1
        else:
            print(item['id'])
            print(f"GT: {gt_ans} | Pred: {pred_ans}")
            wrong_samples.append(item)
    # acc = correct / len(gt)
    acc = correct / len(result)
    print(f'acc:  {acc}')
    print(f'result length:  {len(result)}')
    print(f'gt length:  {len(gt)}')

    # 保存错误样本
    if wrong_samples:
        wrong_output_path = os.path.join(result_dir, "wrong_samples.jsonl")
        with open(wrong_output_path, "w", encoding="utf-8") as f:
            for sample in wrong_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"❗ Wrong samples saved to: {wrong_output_path}")
    else:
        print("🎉 No wrong samples detected.")


if __name__ == "__main__":

    vstar_test_jsonl = '/home/zhaochaoyang/yangfan/dataset/gsarch/visual_search/vstar/vstarbench_test.jsonl'
    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-gqa-multiturn-sft-maxpixel12845056_vstar_test_maxturn5_20251020_232545'
    calcu_acc_and_save_wrong_samples(vstar_test_jsonl, qwen25_sft_result_dir)
    qwen25_rl_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/vigorl_qwen2_5_vl_7b_traj_vstar_multiturn_20251019_205621_vstar_test_maxturn5_20251020_130032"
    calcu_acc_and_save_wrong_samples(vstar_test_jsonl, qwen25_rl_result_dir)
