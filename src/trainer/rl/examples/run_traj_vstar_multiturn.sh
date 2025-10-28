#!/bin/bash
export WANDB_API_KEY=4a2e3972e8397f259ae8c1ccdacb314c6701f132
export RAY_ADDRESS="ray://192.168.100.35:10001"

RUN_VAL_ONLY=false

SAVE_MEM=false
SAVE_MEM=true

domain="traj" # web_grounding, vstar, spatial, web_action

condition="vigorl_multiturn" # vanilla_thinking, vigorl, vigorl_multiturn

SAVE_PATH_BASE="/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/checkpoints/rl"
IMAGE_ROOT="/home/zhaochaoyang/yangfan/dataset/gsarch/vigorl_datasets"

NUM_GPUS=8 # number of gpus on node
NNODES=2

trap 'ray stop --force; exit' SIGINT SIGTERM

MULTITURN=false
CROP_SIZE=512
OFFSET=50
ADD_DOT=false
MIN_PIXELS=3136
MAX_PIXELS=2097152

GROUP_BY_TASK=false
# export WANDB_MODE=offline
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1

if [ "$domain" == "web_grounding" ]; then

    if [ "$condition" == "vanilla_thinking" ]; then

        MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
        SYSTEM_PROMPT="./examples/format_prompt/osatlas_naive_grpo.jinja" # replace with the prompt for your dataset
        MODEL_TAG="vanilla_thinking_qwen2_5_vl_3b_osatlas"
        REWARD_FUNCTION=./examples/reward_function/point_in_bbox.py:point_in_bbox_compute_score

    elif [ "$condition" == "vigorl" ]; then

        MODEL_PATH="gsarch/ViGoRL-MCTS-SFT-3b-Web-Grounding" # gsarch/ViGoRL-MCTS-SFT-7b-Web-Grounding
        SYSTEM_PROMPT="./examples/format_prompt/osatlas_grounded_thinking.jinja" # replace with the prompt for your dataset
        MODEL_TAG="vigorl_qwen2_5_vl_3b_osatlas"
        REWARD_FUNCTION=./examples/reward_function/point_in_bbox_grounded_thinking.py:point_in_bbox_compute_score

    elif [ "$condition" == "vigorl_multiturn" ]; then

        SYSTEM_PROMPT="./examples/format_prompt/osatlas_multiturn_crop.jinja" # replace with the prompt for your dataset
        MODEL_TAG="vigorl_multiturn_qwen2_5_vl_3b_osatlas"
        MULTITURN=true
        MODEL_PATH="gsarch/ViGoRL-Multiturn-MCTS-SFT-3b-Web-Grounding" # gsarch/ViGoRL-Multiturn-7b-Web-Grounding
        CROP_SIZE=384
        OFFSET=50 # crop offset
        ADD_DOT=true
        REWARD_FUNCTION=./examples/reward_function/point_in_bbox_multicrop.py:point_in_bbox_multicrop_compute_score

    fi

    train_file="$DATA_ROOT/web_grounding/vigorl_osatlas_train_RL.jsonl"
    val_file="$DATA_ROOT/web_grounding/vigorl_osatlas_val_RL.jsonl"

elif [ "$domain" == "vstar" ]; then

    if [ "$condition" == "vanilla_thinking" ]; then

        echo "domain: $domain, condition: $condition not implemented"
        exit 1

    elif [ "$condition" == "vigorl" ]; then
    
        # NOTE: V* single turn not tested
        MODEL_PATH="$DATA_ROOT/checkpoints/PATH_TO_SFT_MODEL" 
        SYSTEM_PROMPT="./examples/format_prompt/general_qa.jinja" # replace with the prompt for your dataset
        MODEL_TAG="vigorl_qwen2_5_vl_3b_vstar"
        REWARD_FUNCTION=./examples/reward_function/string_match_grounded_thinking.py:sat_compute_score

    elif [ "$condition" == "vigorl_multiturn" ]; then

        SYSTEM_PROMPT="./examples/format_prompt/osatlas_multiturn_crop.jinja" # replace with the prompt for your dataset
        MODEL_TAG="vigorl_multiturn_qwen2_5_vl_3b_osatlas"
        MULTITURN=true
        MODEL_PATH="gsarch/ViGoRL-Multiturn-MCTS-SFT-3b-Visual-Search" # gsarch/ViGoRL-Multiturn-MCTS-SFT-7b-Visual-Search
        CROP_SIZE=672
        OFFSET=182 # crop offset
        ADD_DOT=false
        MIN_PIXELS=3136
        MAX_PIXELS=2850000
        REWARD_FUNCTION=./examples/reward_function/string_match_multiturn.py:string_match_multiturn_compute_score
    fi

    train_file="$DATA_ROOT/visual_search/vigorl_SA_RL.jsonl"
    val_file="$DATA_ROOT/visual_search/vstar/vstarbench_test_RL.jsonl"

elif [ "$domain" == "spatial" ]; then

    if [ "$condition" == "vanilla_thinking" ]; then

        MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
        SYSTEM_PROMPT="./examples/format_prompt/sat2_naive_grpo.jinja" # replace with the prompt for your dataset
        MODEL_TAG="vanilla_thinking_qwen2_5_vl_3b_sat2"
        REWARD_FUNCTION=./examples/reward_function/string_match.py:sat_compute_score

    elif [ "$condition" == "vigorl" ]; then

        MODEL_PATH="gsarch/ViGoRL-MCTS-SFT-3b-Spatial" # gsarch/ViGoRL-MCTS-SFT-7b-Spatial
        SYSTEM_PROMPT="./examples/format_prompt/sat2_grounded_thinking.jinja" # replace with the prompt for your dataset
        MODEL_TAG="vigorl_qwen2_5_vl_7b_sat2"
        REWARD_FUNCTION=./examples/reward_function/string_match_grounded_thinking.py:sat_compute_score

    elif [ "$condition" == "vigorl_multiturn" ]; then

        echo "domain: $domain, condition: $condition not implemented"
        exit 1

    fi

    train_file="$DATA_ROOT/spatial_reasoning/vigorl_sat2_train_RL.jsonl"
    val_file="$DATA_ROOT/spatial_reasoning/vigorl_sat2_val_RL.jsonl"

elif [ "$domain" == "web_action" ]; then

    if [ "$condition" == "vanilla_thinking" ]; then

        MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
        SYSTEM_PROMPT="./examples/format_prompt/web_action_naive_grpo.jinja" # replace with the prompt for your dataset
        MODEL_TAG="vanilla_thinking_qwen2_5_vl_7b_web_action"
        REWARD_FUNCTION=./examples/reward_function/web_action.py:web_action_compute_score

    elif [ "$condition" == "vigorl" ]; then

        MODEL_PATH="$DATA_ROOT/checkpoints/PATH_TO_SFT_MODEL" 
        SYSTEM_PROMPT="./examples/format_prompt/web_action_grounded_thinking.jinja" # replace with the prompt for your dataset
        MODEL_TAG="vigorl_qwen2_5_vl_7b_web_action"
        REWARD_FUNCTION=./examples/reward_function/web_action_grounded_thinking.py:web_action_compute_score

    elif [ "$condition" == "vigorl_multiturn" ]; then

        echo "domain: $domain, condition: $condition not implemented"
        exit 1

    fi

    train_file="$DATA_ROOT/visual_search/vigorl_SA_RL.jsonl"
    val_file="$DATA_ROOT/visual_search/vstar/vstarbench_test_RL.jsonl"

elif [ "$domain" == "traj" ]; then
    GROUP_BY_TASK=true

    if [ "$condition" == "vanilla_thinking" ]; then

        echo "domain: $domain, condition: $condition not implemented"
        exit 1

    elif [ "$condition" == "vigorl" ]; then

        MODEL_PATH="/home/zhuyousong/yangfan/grounded-rl/checkpoints/sft/Qwen2.5-VL-7B-Instruct-gqa-vaw-spatial-negative-singleturn-refinebbox-sft-maxpixel12845056-lr2e-6_1004" 
        SYSTEM_PROMPT="" # replace with the prompt for your dataset
        MODEL_TAG="vigorl_qwen2_5_vl_7b_traj"
        REWARD_FUNCTION=./examples/reward_function/all_reward.py:compute_score

    elif [ "$condition" == "vigorl_multiturn" ]; then

        MODEL_PATH="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-sft-maxpixel12845056-maxlength32768-lr2e-6-ep10_1025" 
        SYSTEM_PROMPT="" # replace with the prompt for your dataset
        MODEL_TAG="vigorl_qwen2_5_vl_7b_traj_minio3_multiturn"
        REWARD_FUNCTION=./examples/reward_function/all_reward.py:compute_score
        MULTITURN=true

    fi

    train_file="/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/src/trainer/rl/examples/input_data_vstar_multiturn.txt"
    val_file="/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/src/trainer/rl/examples/input_data_val_vstar_multiturn.txt"

fi

DATASET_TAG="${condition}_${domain}"

########################################################

# generate datetime string
DATETIME=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_TAG=""

SAVE_TAG="${MODEL_TAG}_${DATASET_TAG}"

# ROLLOUT
########################################################
ENABLE_CHUNKED_PREFILL=false
GPU_MEMORY_UTILIZATION=0.7
VAL_OVERRIDE_TEMPERATURE=0.5
MAX_PROMPT_LENGTH=16384
MAX_RESPONSE_LENGTH=32768
TENSOR_PARALLEL_SIZE=1
LIMIT_IMAGES=5 # maximum number of full + cropped images for multiturn
MAX_ITERATIONS=3 # maximum multiturn iterations
NUM_ROLLOUTS=5
MAX_GENERATION_LENGTH_PER_TURN=1024
STOP_STRINGS="</tool_call>,</answer>,<|im_end|>"
########################################################

# Trainer
########################################################
TOTAL_EPISODES=1000
SAVE_LIMIT=50
SAVE_FREQ=25
MAX_STEPS=5000 # NOTE: this will override the total_episodes
VAL_BEFORE_TRAIN=false
KL_COEF=1.0e-2
VAL_FREQ=-1
VAL_BATCH_SIZE=1024
REF_UPDATE_STEPS=99999
VAL_ONLY=false
########################################################

# OFFLOADING
########################################################
# DEFAULT
REF_OFFLOAD_PARAMS=false
ACTOR_FSDP_OFFLOAD_TO_CPU=false
ACTOR_OFFLOAD_PARAMS=true
ACTOR_OFFLOAD_OPTIMIZER=true
# # CUSTOM
REF_OFFLOAD_PARAMS=false
ACTOR_FSDP_OFFLOAD_TO_CPU=false
ACTOR_OFFLOAD_PARAMS=false
ACTOR_OFFLOAD_OPTIMIZER=false
########################################################

# actor params
########################################################
TORCH_DTYPE=bf16
OPTIM_STRATEGY=adamw_bf16
LR=1.0e-6
WEIGHT_DECAY=1.0e-2
FREEZE_VISION_TOWER=true
ROLLOUT_BATCH_SIZE=128
GLOBAL_BATCH_SIZE=64
MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE=4
MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE=16
PADDING_FREE=true
MAX_GRAD_NORM=1.0
WARMUP_RATIO=0.0
MASK_NEGATIVE_ADVANTAGE=false
ADAPTIVE_LR=false
########################################################

if [ "$SAVE_MEM" == "true" ]; then
  VAL_BATCH_SIZE=1024
  MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE=2
  MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE=8
  TORCH_DTYPE=bf16
  OPTIM_STRATEGY=adamw_bf16
fi

if [ "$MULTITURN" == "true" ]; then
  MAX_GRAD_NORM=0.2
  LR=1.0e-6
  NUM_ROLLOUTS=8
  ROLLOUT_BATCH_SIZE=32
  GLOBAL_BATCH_SIZE=32
  PADDING_FREE=false
  VAL_FREQ=-1
  VAL_BEFORE_TRAIN=false
  KL_COEF=0.0
  WARMUP_RATIO=0.05
  MASK_NEGATIVE_ADVANTAGE=false
  ADAPTIVE_LR=false
  SAVE_FREQ=15

  LIMIT_IMAGES=10 # maximum number of full + cropped images for multiturn
  MAX_ITERATIONS=6 # maximum multiturn iterations

  MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE=1
  MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE=2
  MAX_PIXELS=1000000
  CROP_MAX_PIXELS=1000000
fi

if [ "$RUN_VAL_ONLY" == "true" ]; then
  VAL_BATCH_SIZE=-1
  VAL_FREQ=1
  VAL_ONLY=true
  VAL_BEFORE_TRAIN=true
  # change this to the path to the checkpoint you want to load
  load_checkpoint_path="${SAVE_PATH_BASE}/qwen2_5_vl_3b_full_sft_osatlas_multiturn_sft_train_OSATLAS_1260_HARD_SAMPLES_500_KEEP_RL__mrl4096_lim5_cs512_os50_kl1.0e-2_lr5.0e-7_wd1.0e-2_fvttrue_rbs64_gbs32_mgn0.2_wr0.05_20250424_095626/global_step_40"
  echo "LOADING CHECKPOINT FROM: ${load_checkpoint_path}"
  VAL_OVERRIDE_TEMPERATURE=0.2
  export WANDB_MODE=disabled
fi

# ######
# # DEBUG
# ROLLOUT_BATCH_SIZE=32
# GLOBAL_BATCH_SIZE=8
# MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE=1
# MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE=4
# NUM_ROLLOUTS=3
# SAVE_FREQ=1
# VAL_FREQ=-1
# LIMIT_IMAGES=4 # maximum number of full + cropped images for multiturn
# MAX_ITERATIONS=4 # maximum multiturn iterations
# VAL_BEFORE_TRAIN=false
# export WANDB_MODE=disabled
# #####

HYPERPARAM_TAG="_mrl${MAX_RESPONSE_LENGTH}_maxite${MAX_ITERATIONS}_lim${LIMIT_IMAGES}_cs${CROP_SIZE}_os${OFFSET}_kl${KL_COEF}_lr${LR}_wd${WEIGHT_DECAY}_fvt${FREEZE_VISION_TOWER}_rbs${ROLLOUT_BATCH_SIZE}_gbs${GLOBAL_BATCH_SIZE}_mgn${MAX_GRAD_NORM}_wr${WARMUP_RATIO}"

SAVE_TAG="${SAVE_TAG}_${HYPERPARAM_TAG}"

# EXPERIMENT_TAG="_SFTINIT"
SAVE_PATH=${SAVE_PATH_BASE}/${SAVE_TAG}_${DATETIME}${EXPERIMENT_TAG}
mkdir -p ${SAVE_PATH}

LOG_PATH_BASE='/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/nohup_log/train/rl'
LOG_PATH="${LOG_PATH_BASE}/${SAVE_TAG}_${DATETIME}${EXPERIMENT_TAG}.log"

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${train_file} \
    data.val_files=${val_file} \
    data.format_prompt=${SYSTEM_PROMPT} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.min_pixels=${MIN_PIXELS} \
    data.max_pixels=${MAX_PIXELS} \
    data.max_side_length=${MAX_SIDE_LENGTH} \
    data.val_batch_size=${VAL_BATCH_SIZE} \
    data.image_root=${IMAGE_ROOT} \
    data.group_by_task=${GROUP_BY_TASK} \
    algorithm.kl_coef=${KL_COEF} \
    algorithm.ref_update_steps=${REF_UPDATE_STEPS} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.fsdp.torch_dtype=${TORCH_DTYPE} \
    worker.actor.optim.strategy=${OPTIM_STRATEGY} \
    worker.actor.optim.lr_warmup_ratio=${WARMUP_RATIO} \
    worker.actor.micro_batch_size_per_device_for_experience=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_EXPERIENCE} \
    worker.actor.micro_batch_size_per_device_for_update=${MICRO_BATCH_SIZE_PER_DEVICE_FOR_UPDATE} \
    worker.actor.padding_free=${PADDING_FREE} \
    worker.actor.max_grad_norm=${MAX_GRAD_NORM} \
    worker.actor.mask_negative_advantage=${MASK_NEGATIVE_ADVANTAGE} \
    worker.actor.adaptive_lr=${ADAPTIVE_LR} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.rollout.enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL} \
    worker.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} \
    worker.rollout.enforce_eager=false \
    worker.rollout.multiturn=${MULTITURN} \
    worker.rollout.limit_images=${LIMIT_IMAGES} \
    worker.rollout.max_iterations=${MAX_ITERATIONS} \
    worker.rollout.val_override_config.temperature=${VAL_OVERRIDE_TEMPERATURE} \
    worker.rollout.n=${NUM_ROLLOUTS} \
    worker.rollout.crop_size=${CROP_SIZE} \
    worker.rollout.crop_max_pixels=${CROP_MAX_PIXELS} \
    worker.rollout.draw_dot=${ADD_DOT} \
    worker.rollout.offset=${OFFSET} \
    worker.rollout.max_generation_length_per_turn=${MAX_GENERATION_LENGTH_PER_TURN} \
    worker.rollout.stop_strings=${STOP_STRINGS} \
    worker.ref.offload.offload_params=${REF_OFFLOAD_PARAMS} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    trainer.experiment_name=${SAVE_TAG}_${DATETIME}${EXPERIMENT_TAG} \
    trainer.n_gpus_per_node=${NUM_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.val_generations_to_log=10 \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.total_episodes=${TOTAL_EPISODES} \
    trainer.max_steps=${MAX_STEPS} \
    trainer.save_limit=${SAVE_LIMIT} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.val_freq=${VAL_FREQ} \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.val_only=${VAL_ONLY} \
    trainer.load_checkpoint_path=${load_checkpoint_path} \
    worker.actor.model.freeze_vision_tower=${FREEZE_VISION_TOWER} 2>&1 | tee ${LOG_PATH}
    # trainer.load_checkpoint_path=${load_checkpoint_path}