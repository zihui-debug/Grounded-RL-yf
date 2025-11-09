import json

def json_to_jsonl(input_json, output_jsonl):    
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def process_answer_in_jsonl(input_file, output_file, task_type, answer_key='answer'):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        data = json.loads(line)
        data['answer'] = {"task_type": task_type, "answer": data[answer_key]}
        processed_lines.append(json.dumps(data))

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')

def filter_chart_data(input_path, output_path):
    """
    读取一个 JSONL 文件，只保留 images[0] 中包含 'chart' 的条目。
    
    Args:
        input_path (str): 输入的 JSONL 文件路径
        output_path (str): 输出的 JSONL 文件路径
    """
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        kept = 0
        total = 0
        for line in infile:
            total += 1
            data = json.loads(line.strip())
            if "images" in data and len(data["images"]) > 0:
                if "chart" in data["images"][0]:
                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                    kept += 1
        
        print(f"✅ Done. Kept {kept} of {total} entries containing 'chart' in images[0].")


if __name__ == "__main__":
    json_to_jsonl(
        "/home/zhaochaoyang/yangfan/dataset/ThinkwithImage/VisualProbe_train/train.json",
        "/home/zhaochaoyang/yangfan/dataset/ThinkwithImage/VisualProbe_train/train.jsonl"
    )
    json_to_jsonl(
        "/home/zhaochaoyang/yangfan/dataset/ThinkwithImage/DeepEyes_train_4K/train.json",
        "/home/zhaochaoyang/yangfan/dataset/ThinkwithImage/DeepEyes_train_4K/train.jsonl"
    )

    ori_jsonl_list = [
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/visual_search/vigorl_SA_RL.jsonl",
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/visual_search/vstar/vstarbench_test_RL.jsonl",
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/web_action/vigorl_ical_train_RL.jsonl",
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/web_action/vigorl_ical_val_RL.jsonl",
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/web_grounding/vigorl_osatlas_train_RL.jsonl",
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/web_grounding/vigorl_osatlas_val_RL.jsonl",
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/spatial_reasoning/vigorl_sat2_train_RL.jsonl",
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/spatial_reasoning/vigorl_sat2_val_RL.jsonl",
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/visual_search/vigorl_SA_RL.jsonl",
        # "/home/zhuyousong/yangfan/datasets/gsarch/vigorl_datasets/visual_search/vstar/vstarbench_test_RL.jsonl",
        # "/home/zhaochaoyang/yangfan/dataset/gsarch/visual_search/vigorl_SA_RL.jsonl",
        # "/home/zhaochaoyang/yangfan/dataset/gsarch/visual_search/vstar/vstarbench_test_RL.jsonl",
        "/home/zhaochaoyang/yangfan/dataset/ThinkwithImage/VisualProbe_train/train.jsonl",
        "/home/zhaochaoyang/yangfan/dataset/ThinkwithImage/DeepEyes_train_4K/train.jsonl"
    ]
    save_jsonl_list = [
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vigorl_SA_RL.jsonl",
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vstarbench_test_RL.jsonl",
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vigorl_ical_train_RL.jsonl",
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vigorl_ical_val_RL.jsonl",
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vigorl_osatlas_train_RL.jsonl",
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vigorl_osatlas_val_RL.jsonl",
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vigorl_sat2_train_RL.jsonl",
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vigorl_sat2_val_RL.jsonl",
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vigorl_SA_multiturn_RL.jsonl",
        # "/home/zhuyousong/yangfan/grounded-rl/data_process/train_format/vigorl_rl_data/vstarbench_multiturn_test_RL.jsonl",
        # "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/processed_data/vigorl_rl_data/vigorl_SA_multiturn_trajformat_RL.jsonl",
        # "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/processed_data/vigorl_rl_data/vstarbench_multiturn_trajformat_test_RL.jsonl",
        "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/processed_data/minio3/VisualProbe_train_train_multiturn.jsonl",
        "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/processed_data/minio3/DeepEyes_train_4K_train_multiturn.jsonl"
    ]
    task_types = [
        # "vstar",
        # "vstar",
        # "webaction",
        # "webaction",
        # "webgrounding",
        # "webgrounding",
        # "spatial",
        # "spatial",
        # "vstar_multiturn",
        # "vstar_multiturn"
        # "vstar_multiturn_trajformat",
        # "vstar_multiturn_trajformat",
        "vstar_multiturn",
        "vstar_multiturn"

    ]
    # for ori_file, save_file, task_type in zip(ori_jsonl_list, save_jsonl_list, task_types):
    #     process_answer_in_jsonl(ori_file, save_file, task_type, answer_key='solution')

    minio3_deepeyes_train_jsonl = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/processed_data/minio3/DeepEyes_train_4K_train_multiturn.jsonl'
    minio3_deepeyes_train_chart_jsonl = '/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/processed_data/minio3/DeepEyes_train_4K_train_multiturn_chart.jsonl'
    filter_chart_data(minio3_deepeyes_train_jsonl, minio3_deepeyes_train_chart_jsonl)