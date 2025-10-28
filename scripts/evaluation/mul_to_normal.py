import json
import re
import os

import json
import re
import os

def convert_thoughts_to_reasoning_jsonl(input_jsonl_path, output_json_path, crop_dir="images"):
    """
    将简化版 JSONL (包含 thoughts 数组) 转换为 ViGoRL / VSTAR 风格 reasoning_steps 格式。
    自动判断最后一步为 final_answer，不重复添加。
    """
    results = []

    with open(input_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            question = data["question"]
            system_prompt = data.get("system_prompt", "")
            image_path = data["image"]
            thoughts = data["thoughts"]
            true_answer = data["true_answer"]
            final_answer = data["final_answer"]

            reasoning_steps = []

            for i, thought in enumerate(thoughts, 1):
                # 🚫 跳过纯答案（没有 <think> 且没有 <tool_call>）
                if "<think>" not in thought and "<tool_call>" not in thought:
            
                    continue

                step = {"step_id": i, "raw_content": thought}

                # 提取 <think> 内容并清洗
                think_match = re.search(r"<think>(.*?)</think>", thought, re.S)
                if think_match:
                    think_text = think_match.group(1).strip()
                    # 去掉末尾类似 "A. red" / "B. green" / "C. pink"
                    think_text = re.sub(r"[A-D]\.\s*[a-zA-Z]+$", "", think_text).strip()
                    step["think"] = think_text

                # # 提取 <tool_call> 中 bbox
                # bbox_match = re.search(r"\"bbox\":\s*\[([^\]]+)\]", thought)
                # if bbox_match:
                #     bbox_values = [float(x.strip()) for x in bbox_match.group(1).split(",")]
                #     step["tool_call"] = {
                #         "name": "search_bbox",
                #         "arguments": {"bbox": bbox_values}
                #     }
                # else:
                #     # 没有 bbox 且是最后一步 → 说明是最终答案
                #     step["answer"] = final_answer

                # 提取 <tool_call> 中 bbox（支持多个二维 bbox）
                bbox_match = re.search(r"\"bbox\":\s*(\[[^\]]+\])", thought)
                if bbox_match:
                    bbox_str = bbox_match.group(1)
                    try:
                        # 尝试解析为 Python 对象（支持单个或多个 bbox）
                        bbox_data = json.loads(bbox_str)

                        # 如果是单个 bbox（如 [10, 20, 30, 40]），包装成二维列表
                        if isinstance(bbox_data[0], (int, float)):
                            bbox_data = [bbox_data]

                        step["tool_call"] = {
                            "name": "search_bbox",
                            "arguments": {"bbox": bbox_data}
                        }

                    except json.JSONDecodeError:
                        print(f"[Warning] Failed to parse bbox JSON: {bbox_str}")
                        step["answer"] = final_answer
                else:
                    # 没有 bbox 且是最后一步 → 说明是最终答案
                    step["answer"] = final_answer

                reasoning_steps.append(step)


            output_data = {
                "question": question,
                "system_prompt": system_prompt,
                "reasoning_steps": reasoning_steps,
                "final_answer": final_answer,
                "gt": true_answer,
                "total_steps": len(reasoning_steps),
                "image_path": image_path
            }

            results.append(output_data)

    # 保存为 JSON 文件（列表形式）
    with open(output_json_path, "w", encoding="utf-8") as f_out:
        json.dump(results, f_out, indent=2, ensure_ascii=False)

    print(f"✅ 已转换 {len(results)} 条样本，保存到：{output_json_path}")




if __name__ == "__main__":
    # convert_thoughts_to_reasoning_jsonl(
    #     input_jsonl_path="/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/vigorl_qwen2_5_vl_7b_traj_vstar_multiturn_20251019_205621_vstar_test_maxturn5_20251020_130032/rollouts_vigorl_qwen2_5_vl_7b_traj_vstar_multiturn_20251019_205621_vstar_test_maxturn5_20251020_130032_worker3_final.jsonl",
    #     output_json_path="output_reasoning.json",
    #     crop_dir="images"
    # )

    rl_wrong_samples = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data/eval/traj_multiturn_vstar_test_novllm/vigorl_qwen2_5_vl_7b_traj_vstar_multiturn_20251019_205621_vstar_test_maxturn5_20251020_130032/wrong_samples.jsonl"
    rl_wrong_reasoning_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/output_reasoning_rl_wrong.json"
    convert_thoughts_to_reasoning_jsonl(
        input_jsonl_path=rl_wrong_samples,
        output_json_path=rl_wrong_reasoning_json,
        crop_dir="images"
    )