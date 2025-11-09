import os
import json
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np


# def visualize_reasoning_full(save_info, pil_img, question, step_records, save_path="./"):
#     """
#     可视化整轮推理过程（包含多轮裁剪+推理）：
#     - 子图1：原图 + 红框 + 问题
#     - 后续子图对：每轮追加一组 <reasoning> + 裁剪图（如有）
#     - 最后一张子图：Final Step（预测 vs GT）
#     """
#     os.makedirs(save_path, exist_ok=True)

#     # ✅ 计算子图总数：1张原图 + 每轮2张 + 1张Final
#     num_tool_calls = sum(1 for step in step_records if step.get("tool_call"))
#     num_subplots = 2 + 2 * num_tool_calls + 1 # 1:原图+问题, +1:Final Step

#     fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots * 4), constrained_layout=True)

#     idx = 0
#     # ===== 子图0：原图 + 红框 + 问题 =====
#     axs[idx].imshow(pil_img)
#     axs[idx].axis("off")
#     axs[idx].set_title(f"[Q] {question}", fontsize=12, pad=2)

#     if "bbox_gt" in save_info:
#         x1, y1, x2, y2 = save_info["bbox_gt"]
#         rect = patches.Rectangle(
#             (x1, y1), x2 - x1, y2 - y1, edgecolor="red", facecolor="none", linewidth=2
#         )
#         axs[idx].add_patch(rect)
#     idx += 1

#     # ===== 动态追加每轮 reasoning 和裁剪图 =====
#     for turn_idx, step in enumerate(step_records):
#         # import pdb; pdb.set_trace()

#         response_message = step["response_message"]
#         bbox = step.get("bbox")
#         cropped_img = step.get("cropped_img")

#         # ---- 推理文本 ----
#         axs[idx].axis("off")
#         reasoning_text = f"Turn {turn_idx}:\n"
#         if "<think>" in response_message:
#             think = response_message.split("<think>")[1].split("</think>")[0].strip()
#             reasoning_text += f"<think>\n{think}\n</think>\n"
#         if "<tool_call>" in response_message:
#             tool = response_message.split("<tool_call>")[1].split("</tool_call>")[0].strip()
#             reasoning_text += f"<tool_call>\n{tool}\n</tool_call>\n"
#         if "<answer>" in response_message:
#             ans = response_message.split("<answer>")[1].split("</answer>")[0].strip()
#             reasoning_text += f"<answer>\n{ans}\n</answer>\n"

#         axs[idx].text(0.01, 0.5, reasoning_text, fontsize=10, wrap=True)
#         idx += 1

#         # ---- 裁剪图 ----
#         if bbox and cropped_img:
#             fig_crop, axs_crop = plt.subplots(1, 2, figsize=(10, 4))
#             axs_crop[0].imshow(pil_img)
#             axs_crop[0].axis("off")
#             axs_crop[0].set_title(f"Predicted BBox (Turn {turn_idx})")

#             x1, y1, x2, y2 = bbox
#             rect = patches.Rectangle(
#                 (x1, y1), x2 - x1, y2 - y1, edgecolor="green", facecolor="none", linewidth=2
#             )
#             axs_crop[0].add_patch(rect)

#             axs_crop[1].imshow(cropped_img)
#             axs_crop[1].axis("off")
#             axs_crop[1].set_title("Cropped Region")

#             buf = BytesIO()
#             fig_crop.savefig(buf, format="png")
#             buf.seek(0)
#             img_crop_combined = Image.open(buf)
#             plt.close(fig_crop)

#             axs[idx].imshow(img_crop_combined)
#             axs[idx].axis("off")
#             axs[idx].set_title("Inference and corresponding cropping area", fontsize=10)
#             idx += 1

#     # ===== 最后一张子图：Final Step =====
#     final_answer = None
#     if step_records and step_records[-1].get("answer"):
#         final_answer = step_records[-1]["answer"]
#     gt = save_info.get("gt", "N/A")

#     axs[idx].axis("off")
#     axs[idx].set_title(
#         f"Final Step\nAnswer: {final_answer or 'N/A'} | GT: {gt}",
#         fontsize=12,
#         pad=5,
#     )

#     # ===== 保存结果 =====
#     fig.savefig(
#         os.path.join(save_path, f"{save_info['image'].replace('.jpg', '')}_multi_vis.png"),
#         bbox_inches="tight",
#     )
#     plt.close(fig)




# def main(json_file, save_path):
#     """主函数：读取 JSON 文件并批量生成可视化"""
#     with open(json_file, "r", encoding="utf-8") as f:
#         data_list = json.load(f)

#     os.makedirs(save_path, exist_ok=True)
#     all_outputs = []

#     for idx, data in enumerate(data_list, 1):
#         question = data["question"]
#         original_image_path = data.get("image_path") or data["images"]["original_image"]
#         pil_img = Image.open(original_image_path).convert("RGB")

#         step_records = []
#         for step in data["reasoning_steps"]:
#             record = {
#                 "response_message": step["raw_content"],
#                 "bbox": None,
#                 "cropped_img": None,
#                 "tool_call": step.get("tool_call"),
#                 "answer": step.get("answer")
#             }

#             if step.get("tool_call"):
#                 bbox = step["tool_call"]["arguments"]["bbox"]
#                 x1, y1, x2, y2 = map(int, bbox)
#                 record["bbox"] = [x1, y1, x2, y2]
#                 cropped_img = pil_img.crop((x1, y1, x2, y2))
#                 record["cropped_img"] = cropped_img

#             step_records.append(record)

#         save_info = {"image": os.path.basename(original_image_path), "gt": data.get("gt")}

#         visualize_reasoning_full(
#             save_info=save_info,
#             pil_img=pil_img,
#             question=question,
#             step_records=step_records,
#             save_path=save_path
#         )

#         out_file = os.path.join(
#             save_path, f"{os.path.basename(original_image_path).replace('.jpg', '')}_multi_vis.png"
#         )
#         all_outputs.append(out_file)
#         print(f"✅ [{idx}/{len(data_list)}] 已保存: {out_file}")

#     return all_outputs


def visualize_reasoning_full(save_info, pil_img, question, step_records, final_answer, save_path="./"):
    """
    可视化整轮推理过程（支持多bbox、多轮推理）：
    - 原图 + 红框 + 问题
    - 每轮推理展示 reasoning + 裁剪图（可多bbox）
    - 最后一张展示 Final Step（预测 vs GT）
    """
    os.makedirs(save_path, exist_ok=True)

    num_tool_calls = sum(1 for step in step_records if step.get("tool_call"))
    # 1 for original image + len(step_records) for reasoning text + num_tool_calls for cropped images + 1 for final step
    num_subplots = 2 + len(step_records) + num_tool_calls

    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots * 4), constrained_layout=True)
    idx = 0

    # ===== 子图0：原图 + 红框 + 问题 =====
    axs[idx].imshow(pil_img)
    axs[idx].axis("off")
    axs[idx].set_title(f"[Q] {question}", fontsize=12, pad=2)

    if "bbox_gt" in save_info:
        x1, y1, x2, y2 = save_info["bbox_gt"]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="red", facecolor="none", linewidth=2)
        axs[idx].add_patch(rect)
    idx += 1

    # ===== 遍历推理步骤 =====
    for turn_idx, step in enumerate(step_records):
        response_message = step["response_message"]
        bboxes = step.get("bboxes")  # ✅ 支持多bbox
        cropped_imgs = step.get("cropped_imgs")

        # ---- 推理文本 ----
        axs[idx].axis("off")
        reasoning_text = f"Turn {turn_idx}:\n"
        if "<think>" in response_message:
            think = response_message.split("<think>")[1].split("</think>")[0].strip()
            reasoning_text += f"<think>\n{think}\n</think>\n"
        if "<tool_call>" in response_message:
            tool = response_message.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            reasoning_text += f"<tool_call>\n{tool}\n</tool_call>\n"
        if "<answer>" in response_message:
            ans = response_message.split("<answer>")[1].split("</answer>")[0].strip()
            reasoning_text += f"<answer>\n{ans}\n</answer>\n"
        axs[idx].text(0.01, 0.5, reasoning_text, fontsize=10, wrap=True)
        idx += 1

        # ---- 多bbox裁剪图展示 ----
        if bboxes and cropped_imgs:
            fig_crop, axs_crop = plt.subplots(1, len(bboxes) + 1, figsize=(5 * (len(bboxes) + 1), 4))
            # ✅ 确保 axs_crop 总是一个 list
            if not isinstance(axs_crop, (list, np.ndarray)):
                axs_crop = [axs_crop]
            elif isinstance(axs_crop, np.ndarray):
                axs_crop = axs_crop.flatten().tolist()

            # 原图标框
            axs_crop[0].imshow(pil_img)
            axs_crop[0].axis("off")
            axs_crop[0].set_title(f"Predicted BBoxes (Turn {turn_idx})")
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="green", facecolor="none", linewidth=2)
                axs_crop[0].add_patch(rect)

            # 裁剪区域拼图
            for j, crop_img in enumerate(cropped_imgs, start=1):
                axs_crop[j].imshow(crop_img)
                axs_crop[j].axis("off")
                axs_crop[j].set_title(f"Cropped {j}")

            # 转成单图放入主图
            buf = BytesIO()
            fig_crop.savefig(buf, format="png")
            buf.seek(0)
            img_crop_combined = Image.open(buf)
            plt.close(fig_crop)

            axs[idx].imshow(img_crop_combined)
            axs[idx].axis("off")
            axs[idx].set_title("Inference and corresponding cropping area", fontsize=10)
            idx += 1

    # ===== Final Step =====
    # final_answer = step_records[-1].get("final_answer") if step_records else None
    gt = save_info.get("gt", "N/A")

    axs[idx].axis("off")
    axs[idx].set_title(
        f"Final Step\nAnswer: {final_answer or 'N/A'} | GT: {gt}",
        fontsize=12,
        pad=5,
    )

    # ===== 保存结果 =====
    fig.savefig(
        os.path.join(save_path, f"{save_info['image'].replace('.jpg', '')}_multi_vis.png"),
        bbox_inches="tight",
    )
    plt.close(fig)


def main(json_file, save_path):
    """主函数：读取 JSON 文件并批量生成可视化（支持多bbox）"""
    with open(json_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    os.makedirs(save_path, exist_ok=True)
    all_outputs = []

    for idx, data in enumerate(tqdm(data_list, desc="Visualizing")):
        question = data["question"]
        original_image_path = data.get("image_path") or data["images"]["original_image"]
        pil_img = Image.open(original_image_path).convert("RGB")

        step_records = []
        for step in data["reasoning_steps"]:
            record = {
                "response_message": step["raw_content"],
                "bboxes": [],
                "cropped_imgs": [],
                "tool_call": step.get("tool_call"),
                "answer": step.get("answer"),
            }

            # ✅ 支持多bbox（二维列表）
            if step.get("tool_call"):
                bbox_data = step["tool_call"]["arguments"]["bbox"]

                # 如果是一维bbox（[x1,y1,x2,y2]），转为二维形式
                if isinstance(bbox_data[0], (int, float)):
                    bbox_data = [bbox_data]

                for bbox in bbox_data:
                    # 验证bbox格式
                    if len(bbox) == 2 and isinstance(bbox[0], (list, tuple)) and isinstance(bbox[1], (list, tuple)):
                        # 格式为 [[x1, y1], [x2, y2]]，转换为 [x1, y1, x2, y2]
                        x1, y1 = map(int, bbox[0])
                        x2, y2 = map(int, bbox[1])
                    elif len(bbox) == 4:
                        # 标准格式 [x1, y1, x2, y2]
                        x1, y1, x2, y2 = map(int, bbox)
                    else:
                        print(f"⚠️ 跳过无效的bbox格式: {bbox} (长度: {len(bbox)})")
                        continue

                    record["bboxes"].append([x1, y1, x2, y2])
                    cropped_img = pil_img.crop((x1, y1, x2, y2))
                    record["cropped_imgs"].append(cropped_img)

            step_records.append(record)

        save_info = {"image": os.path.basename(original_image_path), "gt": data.get("gt")}

        visualize_reasoning_full(
            save_info=save_info,
            pil_img=pil_img,
            question=question,
            step_records=step_records,
            final_answer=data.get("final_answer"),
            save_path=save_path,
        )

        out_file = os.path.join(save_path, f"{Path(original_image_path).stem}_multi_vis.png")
        all_outputs.append(out_file)
        print(f"✅ [{idx+1}/{len(data_list)}] 已保存: {out_file}")

    return all_outputs


if __name__ == "__main__":
    json_file = "output_reasoning.json"
    save_path = "visualize_results"
    # out_path = main(json_file, save_path)
    # print("可视化结果已保存到:", out_path)

    rl_wrong_reasoning_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/output_reasoning_rl_wrong.json"
    save_path_rl = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/visualize_results_rl_wrong"
    # out_path_rl = main(rl_wrong_reasoning_json, save_path_rl)

    # sft_minio3_correct_v2_wrong_correct_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/minio3_vs_v2/minio3_correct_v2_wrong_correct.json"
    # sft_minio3_correct_v2_wrong_correct_visualize_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/visualize/minio3_vs_v2/minio3_correct_v2_wrong_correct"
    # main(sft_minio3_correct_v2_wrong_correct_json, sft_minio3_correct_v2_wrong_correct_visualize_dir)

    # sft_minio3_correct_v2_wrong_wrong_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/minio3_vs_v2/minio3_correct_v2_wrong_wrong.json"
    # sft_minio3_correct_v2_wrong_wrong_visualize_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/visualize/minio3_vs_v2/minio3_correct_v2_wrong_wrong"
    # main(sft_minio3_correct_v2_wrong_wrong_json, sft_minio3_correct_v2_wrong_wrong_visualize_dir)

    # sft_minio3_wrong_v2_correct_correct_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/minio3_vs_v2/minio3_wrong_v2_correct_correct.json"
    # sft_minio3_wrong_v2_correct_correct_visualize_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/visualize/minio3_vs_v2/minio3_wrong_v2_correct_correct"
    # main(sft_minio3_wrong_v2_correct_correct_json, sft_minio3_wrong_v2_correct_correct_visualize_dir)

    # sft_minio3_wrong_v2_correct_wrong_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/minio3_vs_v2/minio3_wrong_v2_correct_wrong.json"
    # sft_minio3_wrong_v2_correct_wrong_visualize_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/visualize/minio3_vs_v2/minio3_wrong_v2_correct_wrong"
    # main(sft_minio3_wrong_v2_correct_wrong_json, sft_minio3_wrong_v2_correct_wrong_visualize_dir)


    # ----------可视化chatgpt前后结果不一致的样本------------------
    sft_v4_correct_v9_wrong_correct_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/minio3_v4_vs_v9/minio3_correct_v2_wrong_correct.json"
    sft_v4_correct_v9_wrong_correct_visualize_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/visualize/minio3_v4_vs_v9/minio3_correct_v2_wrong_correct"
    # main(sft_v4_correct_v9_wrong_correct_json, sft_v4_correct_v9_wrong_correct_visualize_dir)

    sft_v4_correct_v9_wrong_wrong_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/minio3_v4_vs_v9/minio3_correct_v2_wrong_wrong.json"
    sft_v4_correct_v9_wrong_wrong_visualize_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/visualize/minio3_v4_vs_v9/minio3_correct_v2_wrong_wrong"
    # main(sft_v4_correct_v9_wrong_wrong_json, sft_v4_correct_v9_wrong_wrong_visualize_dir)

    sft_v4_wrong_v9_correct_correct_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/minio3_v4_vs_v9/minio3_wrong_v2_correct_correct.json"
    sft_v4_wrong_v9_correct_correct_visualize_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/visualize/minio3_v4_vs_v9/minio3_wrong_v2_correct_correct"
    main(sft_v4_wrong_v9_correct_correct_json, sft_v4_wrong_v9_correct_correct_visualize_dir)

    sft_v4_wrong_v9_correct_wrong_json = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/minio3_v4_vs_v9/minio3_wrong_v2_correct_wrong.json"
    sft_v4_wrong_v9_correct_wrong_visualize_dir = "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/scripts/evaluation/visualize/minio3_v4_vs_v9/minio3_wrong_v2_correct_wrong"
    main(sft_v4_wrong_v9_correct_wrong_json, sft_v4_wrong_v9_correct_wrong_visualize_dir)
