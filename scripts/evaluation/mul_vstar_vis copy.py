import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import numpy as np


def visualize_reasoning_full(save_info, pil_img, question, step_records, save_path="./"):
    """
    可视化整轮推理过程（包含多轮裁剪+推理）：
    - 子图1：原图 + 红框 + 问题
    - 后续子图对：每轮追加一组 <reasoning> + 裁剪图（如有）
    参数：
    - step_records: List[Dict], 每一轮字典格式：
        {
            "response_message": ...,     # full message text
            "bbox": ...,                 # [x, y, w, h] or None
            "cropped_img": PIL.Image or None
        }
    """
    os.makedirs(save_path, exist_ok=True)
    num_subplots = 2  # 原图+问题
    for step in step_records:
        if "tool_call" in step and step["tool_call"]:
            num_subplots += 2
        if "answer" in step and step["answer"]:
            num_subplots += 1   # ✅ 不用 elif，而是 if，这样最后的 answer 一定能加上

    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, num_subplots * 4), constrained_layout=True)


    idx = 0
    # 子图0：原图 + 红框 + 问题
 
    axs[idx].imshow(pil_img)

    axs[idx].axis('off')
    axs[idx].set_title(f"[Q] {question}", fontsize=12, pad=2)
    if 'bbox_gt' in save_info:  # 可选：真实 bbox 红框
        x1, y1, x2, y2 = save_info['bbox_gt']
        rect = patches.Rectangle((x1, y1), x2, y2, edgecolor='red', facecolor='none', linewidth=2)
        axs[idx].add_patch(rect)
    # fig.canvas.draw()
    # extent = axs[idx].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig(os.path.join(save_path, f"subplot_{idx}.png"), bbox_inches=extent)

    idx += 1

    # 动态追加每轮 reasoning 和裁剪图

    for turn_idx, step in enumerate(step_records):
        response_message = step['response_message']
        bbox = step.get('bbox', None)
        cropped_img = step.get('cropped_img', None)

        # 子图1-N：推理文本
 

        axs[idx].axis('off')
        reasoning_text = f"Turn {turn_idx}:\n"
        if "<think>" in response_message:
            think = response_message.split('<think>')[1].split('</think>')[0].strip()
            reasoning_text += f"<think>\n{think}\n</think>\n"
        if "<tool_call>" in response_message:
            tool = response_message.split('<tool_call>')[1].split('</tool_call>')[0].strip()
            reasoning_text += f"<tool_call>\n{tool}\n</tool_call>\n"
        if "<answer>" in response_message:
            ans = response_message.split('<answer>')[1].split('</answer>')[0].strip()
            reasoning_text += f"<answer>\n{ans}\n</answer>\n"
        axs[idx].text(0.01, 0.5, reasoning_text, fontsize=10, wrap=True)
        # fig.canvas.draw()
        # extent = axs[idx].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # fig.savefig(os.path.join(save_path, f"subplot_{idx}.png"), bbox_inches=extent)

        idx += 1

        # 子图2-N（如有 bbox）：原图 + 绿框 + 裁剪图拼图
        if bbox and cropped_img:
            fig_crop, axs_crop = plt.subplots(1, 2, figsize=(10, 4))
            axs_crop[0].imshow(pil_img)
            axs_crop[0].axis('off')
            axs_crop[0].set_title("Predicted BBox")
            x1, y1, x2, y2 = bbox
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor='green', facecolor='none', linewidth=2)
            axs_crop[0].add_patch(rect)

            axs_crop[1].imshow(cropped_img)
            axs_crop[1].axis('off')
            axs_crop[1].set_title("Cropped Region")

            # 将fig_crop渲染为图像，再插入axs[idx]中；无法直接赋值
            from io import BytesIO
            buf = BytesIO()
            fig_crop.savefig(buf, format='png')
            buf.seek(0)
            img_crop_combined = Image.open(buf)
            plt.close(fig_crop)
            axs[idx].imshow(img_crop_combined)
    

            axs[idx].axis('off')
            if "answer" in step and step["answer"]:
                gt = save_info.get("gt", "N/A")
                ans = step["answer"]
                axs[idx].set_title(f"Final Step\nAnswer: {ans} | GT: {gt}", fontsize=11)
            else:
                axs[idx].set_title("Inference and corresponding cropping area")


            # fig.canvas.draw()
            # extent = axs[idx].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # fig.savefig(os.path.join(save_path, f"subplot_{idx}.png"), bbox_inches=extent)

            idx += 1
    # ✅ 循环结束后，额外画一张 Final Answer 子图
    final_answer = None
    if step_records and step_records[-1].get("answer"):
        final_answer = step_records[-1]["answer"]
    gt = save_info.get("gt", "N/A")

    if final_answer:
        axs[idx].axis("off")
        axs[idx].set_title(f"Final Step\nAnswer: {final_answer} | GT: {gt}", fontsize=11)
        idx += 1

    fig.savefig(os.path.join(save_path, f"{save_info['image'].replace('.jpg', '')}_multi_vis.png"))
    plt.close(fig)

import json
from PIL import Image
import os


def main(json_file, save_path):
    """
    主函数：读取json中所有样本，依次生成可视化结果
    """
    # 加载 JSON（包含多个样本）
    with open(json_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    os.makedirs(save_path, exist_ok=True)
    all_outputs = []

    # ✅ 遍历每个样本
    for idx, data in enumerate(data_list, 1):
        question = data["question"]

        # 原始图像
        original_image_path = data.get("image_path") or data["images"]["original_image"]
        pil_img = Image.open(original_image_path).convert("RGB")

        # step_records 转换
        step_records = []
        for step in data["reasoning_steps"]:
            record = {
                "response_message": step["raw_content"],
                "bbox": None,
                "cropped_img": None,
                "tool_call": step.get("tool_call"),
                "answer": step.get("answer")
            }

            # 如果有 bbox，则自动从原图裁剪
            if step.get("tool_call"):
                bbox = step["tool_call"]["arguments"]["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                record["bbox"] = [x1, y1, x2, y2]
                cropped_img = pil_img.crop((x1, y1, x2, y2))
                record["cropped_img"] = cropped_img

            step_records.append(record)

        # 保存信息
        save_info = {
            "image": os.path.basename(original_image_path),
            "gt": data.get("gt")
        }

        # ✅ 生成可视化
        visualize_reasoning_full(
            save_info=save_info,
            pil_img=pil_img,
            question=question,
            step_records=step_records,
            save_path=save_path
        )

        out_file = os.path.join(
            save_path,
            f"{os.path.basename(original_image_path).replace('.jpg', '')}_multi_vis.png"
        )
        all_outputs.append(out_file)
        print(f"✅ [{idx}/{len(data_list)}] 已保存: {out_file}")

    return all_outputs



if __name__ == "__main__":
    # 直接指定输入 json 和输出目录
    json_file = "output_reasoning.json"   # 你的 JSON 文件路径
    save_path = "visualize_results"       # 保存结果目录

    out_path = main(json_file, save_path)
    print("可视化结果已保存到:", out_path)

