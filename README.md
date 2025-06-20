# ViGoRL: Visually Grounded Reinforcement Learning

This repository contains the official code implementation for the paper:

> **Grounded Reinforcement Learning for Visual Reasoning**
> Gabriel Sarch, Snigdha Saha, Naitik Khandelwal, Ayush Jain, Michael J. Tarr, Aviral Kumar, Katerina Fragkiadaki

[**Paper**](https://arxiv.org/abs/2505.23678) | [**Datasets**](https://huggingface.co/datasets/gsarch/vigorl_datasets) | [**Models**](https://huggingface.co/collections/gsarch/vigorl-6855655677fd6ff5864f65f2) | [**Project Page**](https://visually-grounded-rl.github.io/)

---

## Overview

Visually-Grounded Reinforcement Learning (**ViGoRL**) integrates vision-language models (VLMs) with reinforcement learning (RL) to produce spatially grounded reasoning steps explicitly anchored to visual regions. By using a novel multi-turn RL formulation, ViGoRL dynamically zooms into relevant image areas to improve visual attention, grounding, and reasoning capabilities across various visual reasoning tasks.

---

## Features

* **Supported Models**

  * Qwen2/Qwen2.5-VL
  * Easily supports additional models compatible with vLLM

* **Supported Algorithms**

  * Async, distributed Monte Carlo Tree Search (MCTS) over thought steps with VLMs (& visual grounding), and tree search linearization
  * Supervised Fine-Tuning (SFT)
  * Group Relative Policy Optimization (GRPO) reinforcement learning
  * Multi-turn GRPO with visual feedback loops
  * Async, distributed VLM evaluation 

* **Supported Tasks and Benchmarks**

  * **Spatial Reasoning:** SAT-2, BLINK, RoboSpatial
  * **Visual Search:** V\*Bench
  * **Web Grounding and Interaction:** OS-ATLAS, ScreenSpot (V2 and Pro), VisualWebArena
  * **Any vision-text dataset** in a specific format

* **Key Features**

  * Spatial grounding of textual reasoning steps to visual coordinates
  * Dynamic, zoom-in visual feedback for multi-turn reasoning
  * Efficient, async inference via vLLM and multiprocessing

---

## Requirements

* Python >= 3.9
* transformers >= 4.51.0
* flash-attn >= 2.4.3
* vllm >= 0.8.3

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Gabesarch/grounded-rl.git
cd grounded-rl
```

### Environment Setup

```bash
conda create -n grounded-rl python=3.10
conda activate grounded-rl
pip install uv
uv pip install -e .
uv pip install flash-attn --no-build-isolation
uv pip install deepspeed

# RL dependencies
cd src/trainer/rl
uv pip install -e .
cd ../../..

# SFT dependencies
cd src/trainer/offline
uv pip install -e ".[torch,metrics]"
```

---

## Data Setup

Set the dataset path environment variable before running scripts:

```bash
export DATA_ROOT="/path/to/your/data/root"
```

### Downloading Datasets

Run the provided script to download and extract data:

```bash
python download_data.py
```

Datasets include:

* Spatial reasoning tasks (SAT-2, BLINK)
* Visual search tasks (V\*Bench)
* Web grounding and action tasks (OS-ATLAS, ScreenSpot, VisualWebArena)

---

## Model Checkpoints

Pretrained ViGoRL checkpoints are available via Hugging Face:

| Task                  | 3B Model                                                                                                                                                                                                                  | 7B Model                                                                                                                                                           |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Visual Search**     | [ViGoRL Multiturn](https://huggingface.co/gsarch/ViGoRL-Multiturn-3b-Visual-Search), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-Multiturn-MCTS-SFT-3b-Visual-Search)                                                        | [ViGoRL Multiturn](https://huggingface.co/gsarch/ViGoRL-Multiturn-7b-Visual-Search), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-Multiturn-MCTS-SFT-7b-Visual-Search) |
| **Web Grounding**     | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-3b-Web-Grounding), [ViGoRL Multiturn](https://huggingface.co/gsarch/ViGoRL-Multiturn-3b-Web-Grounding), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-MCTS-SFT-3b-Web-Grounding) | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-7b-Web-Grounding)                                                                                                    |
| **Web Action**        | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-3b-Web-Action)                                                                                                                                                              | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-7b-Web-Action)                                                                                                       |
| **Spatial Reasoning** | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-3b-Spatial), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-MCTS-SFT-3b-Spatial)                                                                                           | [ViGoRL](https://huggingface.co/gsarch/ViGoRL-7b-Spatial), [MCTS-SFT](https://huggingface.co/gsarch/ViGoRL-MCTS-SFT-7b-Spatial)                                    |

---

## Repository Structure

```
grounded-rl
├── data/                     # Data and rollouts
├── scripts/
│   ├── evaluation/           # Evaluation scripts
│   └── mcts/                 # MCTS-related scripts
├── src/
│   ├── vlmsearch/
│   │   ├── arguments.py
│   │   └── tree_search/
│   │       ├── mcts_search.py
│   │       └── single_path_rollouts.py
│   └── trainer/
│       ├── offline/          # SFT (Llama-Factory based)
│       └── rl/               # GRPO (EasyR1 based)
└── download_data.py
```

---

## Training Pipeline

### Step 1: Run MCTS to Generate Grounded Traces

```bash
bash scripts/mcts/run_mcts_qwen72b.sh
```

* Specify dataset and vLLM port at the top of the script.

### Step 2: Linearize MCTS Trees

Single-turn:

```bash
python scripts/mcts/build_reasoning_chains_from_mcts.py
```

Multi-turn:

```bash
python scripts/mcts/build_reasoning_chains_from_mcts_multiturn.py
```

### Step 3: Supervised Fine-Tuning (SFT)

Add reasoning chains to `src/trainer/offline/data/dataset_info.json` and run:

```bash
bash src/trainer/offline/examples/train_qwen2_5_vl_sft.sh
```

### Step 4: Reinforcement Learning (GRPO)

Run GRPO on top of the SFT model:

```bash
bash src/trainer/rl/examples/run_vigorl.sh
```

Checkpoints are sharded; convert to HF format using:

```bash
python src/trainer/rl/scripts/model_merger.py
```

---

## Evaluation

Configure `MODEL` and `SYSTEM_PROMPT` variables at the top of each script:

* Multi-turn visual search & ScreenSpot:

```bash
bash scripts/evaluation/eval_multiturn.sh
```

* Spatial reasoning (BLINK, SAT-2):

```bash
bash scripts/evaluation/eval_spatial.sh
```

* Web grounding (ScreenSpot-Pro):

```bash
bash scripts/evaluation/eval_web_grounding.sh
```

---

## Citation

```bibtex
@article{sarch2025vigorl,
    title={Grounded Reinforcement Learning for Visual Reasoning},
    author={Sarch, Gabriel and Saha, Snigdha and Khandelwal, Naitik and Jain, Ayush and Tarr, Michael J and Kumar, Aviral and Fragkiadaki, Katerina},
    year={2025}
}
```

---

## Acknowledgements
This project builds on the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [EasyR1](https://github.com/hiyouga/EasyR1) projects to support visually-grounded RL, we thank all the authors for providing such a high-performance training framework.

## License

The code is released under [MIT License](LICENSE).
