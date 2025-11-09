import os
import json
import re

from collections import defaultdict

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


def compute_accuracy_by_task(result_dir: str, json_path: str):
    """
    计算不同任务类型的回答正确率。

    参数：
        result_dir: 包含多个子目录，每个目录中有 traj.jsonl。
        json_path: 含任务元信息的 json 文件路径（如包含 images 和 doc_id）。
    """

    # 读取任务元信息（doc_id -> task_type）
    docid_to_task = {}
    with open(json_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
        for item in data_list:
            doc_id = item["doc_id"]
            if "images" in item and item["images"]:
                # 从路径中解析任务类型，例如 direct_attributes
                img_path = item["images"][0]
                task_type = img_path.split("/")[-2]  # e.g., direct_attributes
            else:
                task_type = "unknown"
            docid_to_task[doc_id] = task_type

    # 初始化统计信息
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    overall_correct = 0
    overall_total = 0


    # 遍历 result_dir 下的子目录
    for subdir in os.listdir(result_dir):
        traj_path = os.path.join(result_dir, subdir, "traj.jsonl")
        if not os.path.isfile(traj_path):
            continue

        ground_truth = None
        final_answer = None
        doc_id = None
        question = None

        # 读取文件
        with open(traj_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                if "initial_prompt" in data:
                    question = data["initial_prompt"]

                if "doc_id" in data:
                    doc_id = data["doc_id"]

                if "ground_truth" in data:
                    ground_truth = data["ground_truth"].strip().lower()

                if "text_output" in data and "</answer>" in data["text_output"]:
                    match = re.search(r"<answer>\s*(.*?)\s*</answer>", data["text_output"], re.DOTALL)
                    if match:
                        final_answer = match.group(1).strip().lower()
                        break
                    else:
                        final_answer = data["text_output"].split('</answer>')[0].strip().lower()

        if not doc_id or not ground_truth or not final_answer:
            stats[task_type]["total"] += 1
            overall_total += 1
            print(f"❌ 错误 - DocID: {doc_id}, 任务类型: {task_type}, 预测答案: '{final_answer}', 正确答案: '{ground_truth}'")
            continue

        # 确定任务类型
        task_type = docid_to_task.get(doc_id, "unknown")

        stats[task_type]["total"] += 1
        overall_total += 1
        # if final_answer.endswith(ground_truth) or ground_truth in final_answer:
        #     stats[task_type]["correct"] += 1
        is_correct = _string_matching(ground_truth, final_answer, question=data.get("question", ""))
        if is_correct==0.0:
            print(f"❌ 错误 - DocID: {doc_id}, 任务类型: {task_type}, 预测答案: '{final_answer}', 正确答案: '{ground_truth}'")
        else:
            # print(f"✅ 正确 - DocID: {doc_id}, 任务类型: {task_type}, 预测答案: '{final_answer}', 正确答案: '{ground_truth}'")
            pass
        stats[task_type]["correct"] += int(is_correct)
        overall_correct += int(is_correct)

    # 输出结果
    print("\n📊 各任务类型准确率统计：")
    for task, s in stats.items():
        total, correct = s["total"], s["correct"]
        acc = correct / total if total > 0 else 0.0
        print(f" - {task:20s} 正确数: {correct:4d} / {total:4d} | 准确率: {acc:.2%}")
    overall_acc = overall_correct / overall_total if overall_total > 0 else 0.0
    print("\n🌍 整体准确率统计：")
    print(f" 总正确数: {overall_correct} / {overall_total} | 整体准确率: {overall_acc:.2%}")

    return stats

if __name__ == "__main__":

    minio3_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/Mini-o3/eval_output/vstar_eval/qwen2_5vl-7b-mini-o3-coldstart-sft_test_forceanswer"
    vstar_val_json = "/home/zhaochaoyang/yangfan/dataset/gsarch/visual_search/vstar/val.json"
    compute_accuracy_by_task(minio3_sft_result_dir, vstar_val_json)

    our_minio3format_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/Mini-o3/eval_output/vstar_eval/qwen2_5vl-7b-our_minio3format-sft"
    # compute_accuracy_by_task(our_minio3format_sft_result_dir, vstar_val_json)

    our_minio3format_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/Mini-o3/eval_output/vstar_eval/qwen2_5vl-7b-our_minio3format-sft_test_forceanswer"
    compute_accuracy_by_task(our_minio3format_sft_result_dir, vstar_val_json)

    our_minio3format_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/Mini-o3/eval_output/vstar_eval/qwen2_5vl-7b-our_minio3format_nopolish-sft_test_maxturn6_forceanswer"
    compute_accuracy_by_task(our_minio3format_sft_result_dir, vstar_val_json)

    our_minio3format_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/Mini-o3/eval_output/vstar_eval/qwen2_5vl-7b-our_minio3format_nopolish-sft_test_maxturn12_forceanswer"
    compute_accuracy_by_task(our_minio3format_sft_result_dir, vstar_val_json)

    our_minio3format_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/Mini-o3/eval_output/vstar_eval/qwen2_5vl-7b-our_minio3format_7k-sft_test_maxturn6_forceanswer"
    compute_accuracy_by_task(our_minio3format_sft_result_dir, vstar_val_json)

    our_minio3format_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/Mini-o3/eval_output/vstar_eval/Qwen2.5-VL-7B-Instruct-our_minio3_format_3k-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1108_maxturn6_forceanswer_maxpixel12960000"
    compute_accuracy_by_task(our_minio3format_sft_result_dir, vstar_val_json)

    our_minio3format_sft_result_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/Mini-o3/eval_output/vstar_eval/Qwen2.5-VL-7B-Instruct-minio3-v4-optimized-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1108"
    compute_accuracy_by_task(our_minio3format_sft_result_dir, vstar_val_json)