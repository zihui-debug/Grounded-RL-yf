#!/bin/bash

# VLLM number of GPUs
NUM_GPUS=1 # number of GPUs to use for vllm
NUM_PROCESSES=10 # number of processes to run in parallel
SEED=42 # random seed
N_ROLLOUTS=1 # number of rollouts to run per prompt
TEMPERATURE=0.5 # temperature for the rollouts
TOP_P=1.0 # top-p for the rollouts
CHECKPOINT_INTERVAL=10 # checkpoint interval for the rollouts - larger means more samples per generated file
PORT=9001 # port for vllm server
eval_type="sat2_test" # "sat2" or "blink"

# export port so src/vlmsearch/models/qwen_vllm.py can find the vllm server
export PORT

SYSTEM_PROMPT_BASE_MODEL="""You are a helpful assistant. With the image as context, pick the correct answer choice to answer the provided question. Only return the text of the correct answer choice."""
SYSTEM_PROMPT_VANILLA_THINKING="""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.\n\nIMPORTANT:\n\n- The Assistant must always include a <think> section first, where it reasons step by step.\n- After the <think> section, the Assistant provides the final answer inside an <answer> section.\n- Both sections are required in every response. Do not skip the <think> section.\n- The <answer> must contain only a single string in the format with the text of selected answer choice.\n- Your task is to help the user identify the correct answer choice from the options provided.\n- If the question is unclear or ambiguous, infer the most relevant choice from the options."""
SYSTEM_PROMPT_GROUNDED_THINKING="""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant systematically reasons through the problem step by step by checking and verifying possible solutions and image regions, while grounding reasoning steps to specific objects and their relationships in the image using (x,y) coordinates. There may be one image or two images concatenated together, in which case the Assistant must compare the spatial relationships between the two images.\n\nAll reasoning processes must be enclosed within a single set of '<think>' tags, and reasoning steps must include specific reference coordinates:\n\nFor example, <think>\n{Reasoning text}. {Further reasoning text} {more reasoning} \n</think>\n\nThe final answer should be enclosed in '<answer>' tags in the format:\n<answer> {text of selected answer choice} </answer>\n\nThe Assistant must help the user identify the correct answer choice from the options provided.\n-Your answer should be the **exact text** of the selected answer option, without additional explanations or reasoning or the option text. For example, if the answer is A. right , your response should just be <answer>right</answer> (not <answer>A. right</answer>).\n-If the correct answer is unclear, select the most relevant option based on the spatial relationships and dynamics within the image.\n- The Assistant should verify each step and check multiple possible solutions before selecting the final answer."""

####################
# MODEL TO LOAD

REPETITION_PENALTY=1.05

MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
SYSTEM_PROMPT=${SYSTEM_PROMPT_BASE_MODEL}
YES_THINKING=false # set to true if model outputs <think> and <answer> tags
MODEL_TAG="vigorl_qwen2_5_vl_3b_base_model"

MODEL="gsarch/ViGoRL-7b-Spatial" # gsarch/ViGoRL-3b-Spatial
SYSTEM_PROMPT=${SYSTEM_PROMPT_GROUNDED_THINKING}
YES_THINKING=true # set to true if model outputs <think> and <answer> tags
MODEL_TAG="vigorl_qwen2_5_vl_3b_sat2"
###########################

if [ "$eval_type" == "sat2_test" ]; then

    DATA_FILE="${DATA_ROOT}/spatial_reasoning/sat2_test/sat2_test.jsonl"
    IMAGE_ROOT=$DATA_ROOT
    SAVE_TAG="sat2_test"

elif [ "$eval_type" == "blink" ]; then

    DATA_FILE="${DATA_ROOT}/spatial_reasoning/blink/blink_validation.jsonl"
    IMAGE_ROOT=$DATA_ROOT
    SAVE_TAG="blink"

fi

echo "DATA_ROOT: $DATA_ROOT"
echo "MODEL: $MODEL"
echo "DATA_FILE: $DATA_FILE"

SAVE_TAG="${MODEL_TAG}_${SAVE_TAG}"

ACTOR_MODEL="qwen_vllm"
JUDGE="string_match"

MAX_IMAGE_SIDE=1260
MAX_PIXELS=$((MAX_IMAGE_SIDE*MAX_IMAGE_SIDE))
USE_MAX_IMAGE_SIDE=true
USE_MAX_PIXELS=false

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
    args_to_add="--generate_upfront \
    --first_rollout_no_sample"
else
    args_to_add="--rollout_no_thinking \
    --generate_upfront \
    --first_rollout_no_sample"
fi

if [ "$USE_MAX_IMAGE_SIDE" = true ]; then
    args_to_add="$args_to_add --max_image_side $MAX_IMAGE_SIDE"
elif [ "$USE_MAX_PIXELS" = true ]; then
    args_to_add="$args_to_add --max_pixels $MAX_PIXELS"
fi

# Add a cleanup function to kill the vLLM server on script exit
cleanup() {
  echo "[INFO] Cleaning up resources..."
  # Find and kill the vLLM server process
  if pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "[INFO] Killing vLLM server process..."
    pkill -f "vllm serve"
  fi
  exit 0
}

# Set trap to call cleanup function on script exit, including Ctrl+C (SIGINT)
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------
# 1) If ACTOR_MODEL is "qwen_vllm", check if the vllm server is running.
#    If not running, start serve_qwen.sh in the background.
#    Then wait for it to become reachable.
# ---------------------------------------------------------------------
if [ "$ACTOR_MODEL" == "qwen_vllm" ]; then
  # Simple check: see if something is already listening on port 8000
  # or if "vllm serve" is in the process list
  if lsof -Pi :9001 -sTCP:LISTEN -t >/dev/null 2>&1 || pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "[INFO] vllm server is already running on port 9001"
  else
    echo "[INFO] Starting vllm server via serve_qwen.sh ..."
    echo "[INFO] VLLM logs saving to vllm_logs/ ..."
    # You can pass $MODEL and GPU count from environment or SLURM variables
    bash scripts/vllm/serve_qwen.sh "$MODEL" "${NUM_GPUS}" "${PORT}" &
    sleep 5  # brief pause before checking
  fi

  # ----------------------------------------------------------------
  # 2) Wait for the vllm server to be reachable on port 8000
  #    We'll try up to 30 times, sleeping 5 seconds each time.
  # ----------------------------------------------------------------
  echo "[INFO] Please wait for vllm server to be responsive..."
  for i in {1..360}; do
    if curl --max-time 2 -s -o /dev/null http://localhost:9001/v1/models; then
      echo "[INFO] vllm server is up!"
      break
    else
      echo "[WARN] checked vllm server but it is not up yet... retry ($i/360) in 10s"
      sleep 10
    fi

    # if last attempt also fails, exit
    if [ "$i" -eq 360 ]; then
      echo "[ERROR] vllm server did not respond after many attempts, exiting."
      exit 1
    fi
  done
fi

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
    --repetition_penalty ${REPETITION_PENALTY} \
    ${args_to_add}
