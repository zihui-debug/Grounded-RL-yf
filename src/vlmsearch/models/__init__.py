import importlib
import os
import sys

from loguru import logger

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

AVAILABLE_MODELS = {
    "qwen2_vl": "Qwen2_VL",
    "qwen2_5_vl": "Qwen2_5_VL",
    "gpt4o": "GPT4o",
    "qwen_vllm": "Qwen_VLLM",
    "qwen_vllm_traj": "Qwen_VLLM_Traj",
    "qwen2_5_vl_traj": "Qwen2_5_VL_Traj",
}


def get_model(model_name):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not found in available models.")

    model_class = AVAILABLE_MODELS[model_name]
    if "." not in model_class:
        model_class = f"vlmsearch.models.{model_name}.{model_class}"

    try:
        model_module, model_class = model_class.rsplit(".", 1)
        module = __import__(model_module, fromlist=[model_class])
        return getattr(module, model_class)
    except Exception as e:
        logger.error(f"Failed to import {model_class} from {model_name}: {e}")
        raise
