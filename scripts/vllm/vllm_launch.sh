ACTOR_MODEL="qwen_vllm"
# MODEL="/home/zhuyousong/zhengsr/hf_cache/Qwen/Qwen2.5-VL-72B-Instruct"
MODEL="/home/zhaochaoyang/yangfan/project/Grounded-RL-yf/checkpoints/rl/vigorl_qwen2_5_vl_7b_traj_vstar_multiturn_uniquebboxreward_vigorl_multiturn_traj__mrl32768_maxite6_lim10_cs512_os50_kl0.0_lr1.0e-6_wd1.0e-2_fvttrue_rbs32_gbs32_mgn0.2_wr0.05_20251022_000702/global_step_225/actor/huggingface"
NUM_GPUS=4
PORT=9001
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