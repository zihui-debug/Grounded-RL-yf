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

def _string_matching(gt_ans: str, pred_ans: str, question: str = None) -> float:
    """
    Enhanced string matcher that compares the ground truth and predicted answer,
    handling the following cases:
      - Predicted answer may be a letter choice (e.g., "A", "B.")
      - Predicted answer may be "A. right" or "A) right"
      - Predicted answer may be just the text ("right")
    If the prediction is a letter, it will look up the corresponding option text from the question.

    Args:
        gt_ans (str): ground truth answer (text only, e.g. "right")
        pred_ans (str): model's predicted answer (may be letter or text)
        question (str): question string containing "Answer Choices" section

    Returns:
        float: 1.0 if correct, else 0.0
    """
    if not isinstance(pred_ans, str) or not isinstance(gt_ans, str):
        return 0.0

    # Clean up and normalize
    gt_ans_clean = _remove_punctuation_spaces(gt_ans)
    pred_ans_clean = pred_ans.strip()

    # --- Step 1: Try to extract (letter, text) from prediction ---
    match = re.match(r"^\s*([A-Za-z])[.)]\s*(.*)$", pred_ans_clean)
    if match:
        letter = match.group(1).upper()
        after_text = match.group(2).strip()
    else:
        letter, after_text = pred_ans_clean.upper(), pred_ans_clean

    # --- Step 2: Parse choices if question provided ---
    choice_map = {}
    if question and "Answer Choices:" in question:
        # Extract choices like "A. right", "B. left"
        choices_section = question.split("Answer Choices:")[-1]
        matches = re.findall(r"([A-Z])[\.\)]\s*([a-zA-Z0-9_\-\s]+)", choices_section)
        choice_map = {m[0].upper(): _remove_punctuation_spaces(m[1]) for m in matches}

    # --- Step 3: Resolve predicted text ---
    if letter and letter in choice_map:
        pred_text = choice_map[letter].lower()
    elif after_text:
        pred_text = _remove_punctuation_spaces(after_text)
    else:
        pred_text = pred_ans_clean.lower()

    # --- Step 4: Compare clean text ---
    if pred_text == gt_ans_clean or gt_ans_clean in pred_text:
        return 1.0
    else:
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
        score = _string_matching(gt_ans, pred_ans, item.get("question", ""))
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
        score = _string_matching(gt_ans, pred_ans, item.get("question", ""))
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


def calcu_acc_and_save_correct_samples(gt_jsonl, result_dir):
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
    correct_samples = []
    for item in result:
        # if item['judge_score']>0:
        #     correct+=1
        gt_ans = item["true_answer"]
        pred_ans = item["final_answer"]
        score = _string_matching(gt_ans, pred_ans, item.get("question", ""))
        if score > 0:
            correct += 1
            correct_samples.append(item)
        else:
            print(item['id'])
            print(f"GT: {gt_ans} | Pred: {pred_ans}")
    # acc = correct / len(gt)
    acc = correct / len(result)
    print(f'acc:  {acc}')
    print(f'result length:  {len(result)}')
    print(f'gt length:  {len(gt)}')

    # 保存错误样本
    if correct_samples:
        correct_output_path = os.path.join(result_dir, "correct_samples.jsonl")
        with open(correct_output_path, "w", encoding="utf-8") as f:
            for sample in correct_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"❗ correct samples saved to: {correct_output_path}")
    else:
        print("🎉 No correct samples detected.")


def compare_experiment_results(dir1, dir2, output_dir):
    """
    比较两个实验结果目录下的 correct_samples.jsonl 与 wrong_samples.jsonl，
    找出：
      1. 第一次实验正确但第二次错误的样本
      2. 第一次实验错误但第二次正确的样本
    并分别保存为新的 jsonl 文件。

    Args:
        dir1 (str): 第一次实验结果目录路径
        dir2 (str): 第二次实验结果目录路径
        output_dir (str): 输出文件保存目录路径
    """

    # 定义输入文件路径
    correct1_path = os.path.join(dir1, "correct_samples.jsonl")
    wrong1_path = os.path.join(dir1, "wrong_samples.jsonl")
    correct2_path = os.path.join(dir2, "correct_samples.jsonl")
    wrong2_path = os.path.join(dir2, "wrong_samples.jsonl")

    # 读取 JSONL 文件
    def read_jsonl(path):
        if not os.path.exists(path):
            print(f"⚠️ File not found: {path}")
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f if line.strip()]
        # 用 doc_id 作为唯一标识（也可换成其他唯一键，如 image_name）
        return {d.get("id"): d for d in data if "id" in d}

    correct1 = read_jsonl(correct1_path)
    wrong1 = read_jsonl(wrong1_path)
    correct2 = read_jsonl(correct2_path)
    wrong2 = read_jsonl(wrong2_path)

    # ✅ 第一次正确但第二次错误
    only_correct_in_1_1 = {
        k: v for k, v in correct1.items() if k in wrong2
    }
    only_correct_in_1_2 = {
        k: v for k, v in wrong2.items() if k in correct1
    }

    # ❌ 第一次错误但第二次正确
    only_correct_in_2_1 = {
        k: v for k, v in wrong1.items() if k in correct2
    }
    only_correct_in_2_2 = {
        k: v for k, v in correct2.items() if k in wrong1
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 输出路径
    out1_1 = os.path.join(output_dir, "correct_in_1_wrong_in_2_1.jsonl")
    out2_1 = os.path.join(output_dir, "wrong_in_1_correct_in_2_1.jsonl")
    out1_2 = os.path.join(output_dir, "correct_in_1_wrong_in_2_2.jsonl")
    out2_2 = os.path.join(output_dir, "wrong_in_1_correct_in_2_2.jsonl")

    # 写出结果
    def write_jsonl(path, data_dict):
        with open(path, "w", encoding="utf-8") as f:
            for v in data_dict.values():
                f.write(json.dumps(v, ensure_ascii=False) + "\n")

    write_jsonl(out1_1, only_correct_in_1_1)
    write_jsonl(out2_1, only_correct_in_2_1)
    write_jsonl(out1_2, only_correct_in_1_2)
    write_jsonl(out2_2, only_correct_in_2_2)

    print(f"✅ Saved {len(only_correct_in_1_1)} samples to {out1_1}")
    print(f"✅ Saved {len(only_correct_in_2_1)} samples to {out2_1}")
    print(f"✅ Saved {len(only_correct_in_1_2)} samples to {out1_2}")
    print(f"✅ Saved {len(only_correct_in_2_2)} samples to {out2_2}")


if __name__ == "__main__":

    vstar_test_jsonl = '/home/zhaochaoyang/yangfan/dataset/gsarch/visual_search/vstar/vstarbench_test.jsonl'
    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-gqa-multiturn-sft-maxpixel12845056_vstar_test_maxturn5_20251020_232545'
    # calcu_acc_and_save_wrong_samples(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_minio3_ori_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-ori-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1031_vstar_test_maxturn5_20251031_200430'
    # calcu_acc_and_save_wrong_samples(vstar_test_jsonl, qwen25_minio3_ori_sft_result_dir)
    # calcu_acc_and_save_correct_samples(vstar_test_jsonl, qwen25_minio3_ori_sft_result_dir)

    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v2-optimized-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1103_vstar_test_maxturn5_20251103_175937'
    # calcu_acc_and_save_wrong_samples(vstar_test_jsonl, qwen25_sft_result_dir)
    # calcu_acc_and_save_correct_samples(vstar_test_jsonl, qwen25_sft_result_dir)

    sft_minio3_ori_vs_v2_optimized_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/eval_processed/minio3_ori_vs_v2_optimized'
    # compare_experiment_results(qwen25_minio3_ori_sft_result_dir, qwen25_sft_result_dir, sft_minio3_ori_vs_v2_optimized_dir)

    # 比较用chatgpt润色前后推理结果（v4 vs v9)
    qwen25_v4_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v4-optimized-sft-maxpixel2000000-maxlength32768-lr1e-5-bs32-ep3_1104_vstar_test_maxturn5_20251105_104447'
    calcu_acc_and_save_wrong_samples(vstar_test_jsonl, qwen25_v4_sft_result_dir)
    calcu_acc_and_save_correct_samples(vstar_test_jsonl, qwen25_v4_sft_result_dir)

    qwen25_v9_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v9-optimized-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1105_vstar_test_maxturn5_20251106_110347'
    calcu_acc_and_save_wrong_samples(vstar_test_jsonl, qwen25_v9_sft_result_dir)
    calcu_acc_and_save_correct_samples(vstar_test_jsonl, qwen25_v9_sft_result_dir)

    sft_minio3_v4_vs_v9_optimized_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/eval_processed/minio3_v4_vs_v9'
    compare_experiment_results(qwen25_v4_sft_result_dir, qwen25_v9_sft_result_dir, sft_minio3_v4_vs_v9_optimized_dir)





    qwen25_rl_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/vigorl_qwen2_5_vl_7b_traj_vstar_multiturn_20251019_205621_vstar_test_maxturn5_20251020_130032"
    # calcu_acc_and_save_wrong_samples(vstar_test_jsonl, qwen25_rl_result_dir)
