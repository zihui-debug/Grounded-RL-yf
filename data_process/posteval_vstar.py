import os
import json
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

# def _string_matching(gt_ans: str, pred_ans: str) -> float:

#     """
#     Judges the predicted answer and returns a float score. 1.0 for the correct answer, 0.0 for the wrong answer.
    
#     INPUTS:
#     - gt_ans: The ground truth answer.
#     - pred_ans: The predicted answer.
    
#     OUTPUTS:
#     - score: The score of the predicted answer.
    
#     """

#     #Accounitng for the fact that the answers may be in different cases
#     gt_ans = gt_ans.lower()
#     pred_ans = pred_ans.lower()

#     pred_ans = _remove_punctuation_spaces(pred_ans)
#     gt_ans = _remove_punctuation_spaces(gt_ans)

#     if gt_ans == pred_ans or gt_ans in pred_ans:
#         # logger.debug(f"Judge response : 1.0")
#         return 1.0
#     else:
#         # logger.debug(f"Judge response : 0.0")
#         return 0.0

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
        # matches = re.findall(r"([A-Z])[\.\)]\s*([a-zA-Z0-9_\-\s]+)", choices_section)
        matches = re.findall(
            r"([A-Z])[\.\)]\s*([^\n]+)",
            choices_section,
            flags=re.MULTILINE
        )
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
    # === 读取 GT（ground truth） ===
    gt = []
    with open(gt_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            gt.append(json.loads(line.strip()))

    # === 读取结果文件 ===
    result = []
    for filename in os.listdir(result_dir):
        if filename.startswith("rollouts") and filename.endswith(".jsonl"):
            file_path = os.path.join(result_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    result.append(json.loads(line.strip()))

    # === 统计总体与任务分类准确率 ===
    total, correct = 0, 0
    task_stats = {
        "direct_attributes": {"correct": 0, "total": 0},
        "relative_position": {"correct": 0, "total": 0}
    }

    for item in result:
        # debug
        # if item['id'] != '98':
        #     continue
        total += 1
        gt_ans = item["true_answer"]
        pred_ans = item["final_answer"]
        score = _string_matching(gt_ans, pred_ans, item.get("question", ""))
        # is_correct = item.get("judge_score", 0) > 0
        is_correct = score > 0
        if is_correct:
            correct += 1
        else:
            print(item['id'])
            # print(f"Question: {item['question']} | GT: {gt_ans} | Pred: {pred_ans}")
            print(f"GT: {gt_ans} | Pred: {pred_ans}")

        # 判断属于哪个任务
        img_path = item.get("image", "").lower()
        if "direct_attributes" in img_path:
            task = "direct_attributes"
        elif "relative_position" in img_path:
            task = "relative_position"
        else:
            task = None

        if task:
            if is_correct:
                task_stats[task]["correct"] += 1
    for item in gt:
        img_path = item.get("image", "").lower()
        if "direct_attributes" in img_path:
            task = "direct_attributes"
        elif "relative_position" in img_path:
            task = "relative_position"
        else:
            task = None

        if task:
            task_stats[task]["total"] += 1

    # === 打印总体准确率 ===
    acc = correct / len(gt) if len(gt) > 0 else 0
    print(f"\n🔹 Overall Accuracy: {acc:.4f} ({correct}/{len(gt)})")
    print(f"GT samples: {len(gt)}")
    print(f"Pred samples: {len(result)}")

    # === 打印子任务准确率 ===
    for task, stats in task_stats.items():
        if stats["total"] > 0:
            sub_acc = stats["correct"] / stats["total"]
            print(f"  ✅ {task}: {sub_acc:.4f} ({stats['correct']}/{stats['total']})")
        else:
            print(f"  ⚠️ {task}: No samples found")



if __name__ == "__main__":
    # _string_matching("right", "Right")
    vstar_test_jsonl = '/home/zhaochaoyang/yangfan/dataset/gsarch/visual_search/vstar/vstarbench_test.jsonl'
    reason_bench = "/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/traj_eval/reasonbench_v2/merged_reasonbench_converted.jsonl"
    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-gqa-multiturn-sft-maxpixel12845056_vstar_test_maxturn5_20251020_232545'
    # calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-sft-maxpixel12845056-maxlength32768-lr2e-6-ep10_1025_vstar_test_maxturn5_20251030_000545'
    # calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-trajformat-sft-maxpixel12845056-maxlength32768-lr2e-6-ep10_1026_vstar_test_maxturn5_20251026_220435'
    # calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-trajformat-sft-maxpixel12845056-maxlength32768-lr2e-6-ep10_1026_vstar_test_maxturn5_20251029_230559'
    # calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-all-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1028_vstar_test_maxturn5_20251029_225751'
    # calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-all-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1030_vstar_test_maxturn5_20251030_144720'
    # calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-all-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1028_vstar_test_maxturn5_20251029_225751'
    # calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-ori-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1031_vstar_test_maxturn5_20251031_200430"
    # calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v2-optimized-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1103_vstar_test_maxturn5_20251103_175937"
    # calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)
    
    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v4-optimized-sft-maxpixel2000000-maxlength32768-lr1e-5-bs32-ep3_1104_vstar_test_maxturn5_20251105_104447"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v9-optimized-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1105_vstar_test_maxturn5_20251106_110347"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-ori-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1106_vstar_test_maxturn5_20251106_170625"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v4-optimized-sft-maxpixel2000000-maxlength32768-lr1e-5-bs32-ep3_1104_vstar_test_maxturn11_20251106_203741"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v11-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1106_reasonbench_test_maxturn5_20251107_121518"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v11_v2-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1106_reasonbench_test_maxturn5_20251107_121231"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/sft-qwen2_5vl-7b-full-our_minio3format_nopolish_reasonbench_test_maxturn5_20251108_023936"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/sft-qwen2_5vl-7b-full-our_minio3format_7k_reasonbench_test_maxturn5_20251108_100131"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-our_minio3_format_3k-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1108_vstar_test_maxturn5_20251108_120603"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v4-optimized-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1108_vstar_test_maxturn5_20251108_163629"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/sft-qwen2_5vl-7b-full-our_minio3format_7k_absolute_vstar_test_maxturn5_20251109_110105"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)

    qwen25_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/sft-qwen2_5vl-7b-full-our_minio3format_7k-full-our_minio3format_7k_absolute_vstar_test_maxturn5_20251109_110021"
    calcu_acc(vstar_test_jsonl, qwen25_sft_result_dir)
 
    qwen25_rl_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/vigorl_qwen2_5_vl_7b_traj_vstar_multiturn_20251019_205621_vstar_test_maxturn5_20251020_130032"
    # calcu_acc(vstar_test_jsonl, qwen25_rl_result_dir)

    qwen25_rl_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/vigorl_qwen2_5_vl_7b_traj_vstar_multiturn_uniquebboxreward_20251022_000702_step_225_vstar_test_maxturn5_20251024_135524"
    # calcu_acc(vstar_test_jsonl, qwen25_rl_result_dir)

    qwen25_rl_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/vigorl_qwen2_5_vl_7b_traj_minio3_multiturn_vigorl_multiturn_traj__mrl32768_maxite6_lim10_cs512_os50_kl0.0_lr1.0e-6_wd1.0e-2_fvttrue_rbs32_gbs32_mgn0.2_wr0.05_20251026_113732_global_step_45_vstar_test_maxturn5_20251030_000728"
    # calcu_acc(vstar_test_jsonl, qwen25_rl_result_dir)

    qwen25_rl_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/Qwen2.5-VL-7B-Instruct-minio3-v4-optimized-sft-maxpixel2000000-maxlength32768-lr1e-5-bs32-ep3_1104_reasonbench_test_maxturn11_20251106_231345"
    # calcu_acc(reason_bench, qwen25_rl_result_dir)