import json

def process_answer_in_jsonl(input_file, output_file, task_type):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    processed_lines = []
    for line in lines:
        data = json.loads(line)
        data['answer'] = {"task_type": task_type, "answer": data['answer']}
        processed_lines.append(json.dumps(data))

    with open(output_file, 'w', encoding='utf-8') as f:
        for line in processed_lines:
            f.write(line + '\n')


if __name__ == "__main__":
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
        "/home/zhaochaoyang/yangfan/dataset/gsarch/visual_search/vigorl_SA_RL.jsonl",
        "/home/zhaochaoyang/yangfan/dataset/gsarch/visual_search/vstar/vstarbench_test_RL.jsonl",
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
        "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/processed_data/vigorl_rl_data/vigorl_SA_multiturn_trajformat_RL.jsonl",
        "/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/data_process/processed_data/vigorl_rl_data/vstarbench_multiturn_trajformat_test_RL.jsonl",
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
        "vstar_multiturn_trajformat",
        "vstar_multiturn_trajformat",

    ]
    for ori_file, save_file, task_type in zip(ori_jsonl_list, save_jsonl_list, task_types):
        process_answer_in_jsonl(ori_file, save_file, task_type)