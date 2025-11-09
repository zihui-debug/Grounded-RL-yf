#!/bin/bash

# VLLM number of GPUs
NUM_GPUS=1 # number of GPUs to use for vllm
NUM_PROCESSES=4 # number of processes to run in parallel
SEED=42 # random seed
N_ROLLOUTS=1 # number of rollouts to run per prompt
TEMPERATURE=0.5 # temperature for the rollouts
TOP_P=0.99 # top-p for the rollouts
CHECKPOINT_INTERVAL=10 # checkpoint interval for the rollouts - larger means more samples per generated file
PORT=9001 # port for vllm server
# VLLM number of GPUs
eval_type="traj_vstar" # "vstar" or "web_grounding"

# export port so src/vlmsearch/models/qwen_vllm.py can find the vllm server
export PORT

if [ "$eval_type" == "vstar" ]; then

    SYSTEM_PROMPT_VSTAR="""You are a helpful assistant tasked with answering a question about an image. You should systematically reason through the problem step by step by checking and verifying relevant image regions, while grounding reasoning steps to specific (x, y) points in the image:\n- At each turn, first clearly reason about ONE area or element in the image enclosed in <think> </think> tags.\n- After reasoning, either:\n  a) Zoom-in on a specific region to see it better by outputting a search action formatted precisely as:\n     <tool_call>\n     {\"name\": \"search_coordinate\", \"arguments\": {\"coordinate\": [x, y]}}\n     </tool_call>\n  b) If confident you've found the correct location, output your final answer enclosed in <answer> {final answer} </answer> tags.\n- Only answer if you are confident about the answer. If you are not confident, output a search action. You should not always end after one turn.\n- You should not repeat the same coordinates in a tool call more than once. Coordinates must be unique across tool calls, including values that are the same or nearly identical (e.g., differing by only a few pixels).\n- If unclear, infer based on likely context or purpose.\n- Verify each step by examining multiple possible solutions before selecting a final answer."""
    MODEL="gsarch/ViGoRL-Multiturn-7b-Visual-Search" # gsarch/ViGoRL-Multiturn-3b-Visual-Search
    MODEL_TAG="vigorl_qwen2_5_vl_7b_vstar_multicrop"

    YES_THINKING=true
    SYSTEM_PROMPT="${SYSTEM_PROMPT_VSTAR}"

    CROP_SIZE=672
    CROP_OFFSET=182 # crop offset
    DRAW_DOT=false

    DATA_FILE="${DATA_ROOT}/visual_search/vstar/vstarbench_test.jsonl"
    IMAGE_ROOT=$DATA_ROOT
    SAVE_TAG="vstar_test"

    JUDGE="string_match"

elif [ "$eval_type" == "web_grounding" ]; then

    SYSTEM_PROMPT_WEB_GROUNDING="""You are an assistant tasked with identifying precise (x,y) coordinates of a described element in an image.\nYour task involves multiple turns of reasoning, each with EXACTLY one <think> step and one action:\n- At each turn, first clearly reason about ONE area or element in the image enclosed in <think> </think> tags.\n- After reasoning, either:\n  a) Output a search action formatted precisely as:\n     <tool_call>\n     {\"name\": \"search_coordinate\", \"arguments\": {\"coordinate\": [x, y]}}\n     </tool_call>\n  b) If confident you've found the correct location, output your final answer enclosed in <answer> (x, y) </answer> tags.\n- Only answer if you are confident about the answer. If you are not confident, output a search action. You should not always end after one turn.\n- You should not repeat the same coordinates in a tool call more than once. Coordinates must be unique across tool calls, including values that are the same or nearly identical (e.g., differing by only a few pixels)."""
    MODEL="gsarch/ViGoRL-Multiturn-3b-Web-Grounding" # gsarch/ViGoRL-Multiturn-7b-Web-Grounding
    MODEL_TAG="vigorl_multiturn_qwen2_5_vl_3b_osatlas"

    CROP_SIZE=512
    CROP_OFFSET=100
    DRAW_DOT=true

    YES_THINKING=true
    SYSTEM_PROMPT="${SYSTEM_PROMPT_WEB_GROUNDING}"

    JUDGE="point_in_bbox"

elif [ "$eval_type" == "traj_vstar" ]; then

    SYSTEM_PROMPT_WEB_GROUNDING="""You are a helpful assistant tasked with answering a question about an image. You should systematically examine different regions and phrases in the image by requesting to see specific bounding box regions:\n- At each turn, first reason about what you want to examine enclosed in <think> </think> tags.\n- Then request to see a specific region by outputting a search action formatted as:\n     <tool_call>(x1, y1, x2, y2)</tool_call>\n- After examining all relevant regions, provide your final answer enclosed in <answer> {final answer} </answer> tags.\n- Use the information from each region to build comprehensive understanding before answering."""
    # MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-gqa-multiturn-sft-maxpixel12845056-maxlength32768-lr2e-6_1020"
    # MODEL_TAG="Qwen2.5-VL-7B-Instruct-gqa-multiturn-sft-maxpixel12845056-maxlength32768-lr2e-6_1020"
    # MODEL="/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/checkpoints/rl/vigorl_qwen2_5_vl_7b_traj_vstar_multiturn_uniquebboxreward_vigorl_multiturn_traj__mrl32768_maxite6_lim10_cs512_os50_kl0.0_lr1.0e-6_wd1.0e-2_fvttrue_rbs32_gbs32_mgn0.2_wr0.05_20251022_000702/global_step_225/actor/huggingface" # gsarch/ViGoRL-Multiturn-7b-Web-Grounding
    # MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-sft-maxpixel12845056-maxlength32768-lr2e-6-ep10_1025"
    # MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-trajformat-sft-maxpixel12845056-maxlength32768-lr2e-6-ep10_1026"
    # MODEL="/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/checkpoints/rl/vigorl_qwen2_5_vl_7b_traj_minio3_multiturn_vigorl_multiturn_traj__mrl32768_maxite6_lim10_cs512_os50_kl0.0_lr1.0e-6_wd1.0e-2_fvttrue_rbs32_gbs32_mgn0.2_wr0.05_20251026_113732/global_step_45/actor/huggingface"
    # MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-all-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1028"
    # MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-multiturn-all-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1030"
    # MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-ori-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1031"
    # MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-v2-optimized-sft-maxpixel1000000-maxlength32768-lr2e-6-ep5_1103"
    MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-v4-optimized-sft-maxpixel2000000-maxlength32768-lr1e-5-bs32-ep3_1104"
    # MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-v9-optimized-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1105"
    # MODEL="/home/zhaochaoyang/yangfan/project/Qwen2.5-VL-traj/checkpoints/trajvlm/Qwen2.5-VL-7B-Instruct-minio3-ori-sft-maxpixel1000000-maxlength32768-lr1e-5-bs32-ep3_1106"
    MODEL_TAG="Qwen2.5-VL-7B-Instruct-minio3-v4-optimized-sft-maxpixel2000000-maxlength32768-lr1e-5-bs32-ep3_1104"

    CROP_SIZE=512 # 不起作用，py里直接用bbox大小
    CROP_OFFSET=0
    DRAW_DOT=false

    YES_THINKING=true
    SYSTEM_PROMPT="${SYSTEM_PROMPT_WEB_GROUNDING}"
    DATA_FILE="/home/zhaochaoyang/yangfan/dataset/gsarch/visual_search/vstar/vstarbench_test.jsonl"
    IMAGE_ROOT="/home/zhaochaoyang/yangfan/dataset/gsarch"

    JUDGE="string_match"
    SAVE_TAG="vstar_test"

fi
#######################
MAX_TURNS=11

SAVE_TAG="${MODEL_TAG}_${SAVE_TAG}_maxturn${MAX_TURNS}"

ACTOR_MODEL="qwen2_5_vl_traj"

MAX_IMAGE_SIDE=3600
MAX_PIXELS=$((MAX_IMAGE_SIDE*MAX_IMAGE_SIDE))
USE_MAX_IMAGE_SIDE=false
USE_MAX_PIXELS=true


echo "CROP_OFFSET: $CROP_OFFSET"
echo "CROP_SIZE: $CROP_SIZE"

# RUN ARGUMENTS
MAX_NEW_TOKENS=2048 # maximum number of tokens to generate
THOUGHT_TOKEN_BEGIN="<think>" # token to begin the thought section
THOUGHT_TOKEN_END="</think>" # token to end the thought section
FINAL_TOKEN_BEGIN="<answer>" # token to begin the answer section
FINAL_TOKEN_END="</answer>" # token to end the answer section

# Additional arguments
# rollout_no_thinking: model does not think, just generates the answer
# generate_upfront: model generates the answer upfront, without iteratively querying the model
# first_rollout_no_sample: first rollout gets temperature = 0 (other rollouts get temperature = TEMPERATURE)
if [ "$YES_THINKING" = true ]; then
    args_to_add=""
else
    args_to_add="--rollout_no_thinking"
fi

if [ "$USE_MAX_IMAGE_SIDE" = true ]; then
    args_to_add="$args_to_add --max_image_side $MAX_IMAGE_SIDE"
elif [ "$USE_MAX_PIXELS" = true ]; then
    args_to_add="$args_to_add --max_pixels $MAX_PIXELS"
fi

if [ "$DRAW_DOT" = true ]; then
    args_to_add="$args_to_add --draw_dot"
fi

# # Add a cleanup function to kill the vLLM server on script exit
# cleanup() {
#   echo "[INFO] Cleaning up resources..."
#   # Find and kill the vLLM server process
#   if pgrep -f "vllm serve" >/dev/null 2>&1; then
#     echo "[INFO] Killing vLLM server process..."
#     pkill -f "vllm serve"
#   fi
#   exit 0
# }

# # Set trap to call cleanup function on script exit, including Ctrl+C (SIGINT)
# trap cleanup EXIT INT TERM

# # ---------------------------------------------------------------------
# # 1) If ACTOR_MODEL is "qwen_vllm", check if the vllm server is running.
# #    If not running, start serve_qwen.sh in the background.
# #    Then wait for it to become reachable.
# # ---------------------------------------------------------------------
# if [ "$ACTOR_MODEL" == "qwen_vllm" ] || [ "$ACTOR_MODEL" == "qwen_vllm_traj" ]; then
#   # Simple check: see if something is already listening on port 8000
#   # or if "vllm serve" is in the process list
#   if lsof -Pi :9001 -sTCP:LISTEN -t >/dev/null 2>&1 || pgrep -f "vllm serve" >/dev/null 2>&1; then
#     echo "[INFO] vllm server is already running on port 9001"
#   else
#     echo "[INFO] Starting vllm server via serve_qwen.sh ..."
#     echo "[INFO] VLLM logs saving to vllm_logs/ ..."
#     # You can pass $MODEL and GPU count from environment or SLURM variables
#     bash scripts/vllm/serve_qwen.sh "$MODEL" "${NUM_GPUS}" "${PORT}" &
#     sleep 5  # brief pause before checking
#   fi

#   # ----------------------------------------------------------------
#   # 2) Wait for the vllm server to be reachable on port 8000
#   #    We'll try up to 30 times, sleeping 5 seconds each time.
#   # ----------------------------------------------------------------
#   echo "[INFO] Please wait for vllm server to be responsive..."
#   for i in {1..360}; do
#     if curl --max-time 2 -s -o /dev/null http://localhost:9001/v1/models; then
#       echo "[INFO] vllm server is up!"
#       break
#     else
#       echo "[WARN] checked vllm server but it is not up yet... retry ($i/360) in 10s"
#       sleep 10
#     fi

#     # if last attempt also fails, exit
#     if [ "$i" -eq 360 ]; then
#       echo "[ERROR] vllm server did not respond after many attempts, exiting."
#       exit 1
#     fi
#   done
# fi

python -m src.vlmsearch \
    --num_processes=${NUM_PROCESSES} \
    --use_python_mp \
    --model ${ACTOR_MODEL} \
    --seed ${SEED} \
    --judge ${JUDGE} \
    --search_method single_path_rollouts \
    --n_rollouts ${N_ROLLOUTS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --checkpoint_interval ${CHECKPOINT_INTERVAL} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --do_data_checkpoint \
    --system_prompt "${SYSTEM_PROMPT}" \
    --pretrained "${MODEL}" \
    --image_root "${IMAGE_ROOT}" \
    --data_files "${DATA_FILE}" \
    --save_tag "${SAVE_TAG}" \
    --crop_offset ${CROP_OFFSET} \
    --crop_size ${CROP_SIZE} \
    --max_depth ${MAX_TURNS} \
    --check_for_crop \
    --multicrop \
    --save_rollouts_dir "data/eval/traj_multiturn_vstar_test_novllm" \
    ${args_to_add}
