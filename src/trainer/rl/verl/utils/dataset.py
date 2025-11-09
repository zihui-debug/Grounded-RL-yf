# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..models.transformers.qwen2_vl import get_rope_index
from . import torch_functional as VF
from qwen_vl_utils import fetch_image


GQA_PROMPT = '''
You are a helpful visual assistant. Given an image and a question, carefully observe the image, identify important visual elements, and reason step by step to answer the question. You must systematically examine and verify relevant regions of the image, grounding the reasoning steps to specific (x1, y1, x2, y2) regions.

** Instructions **:

- Observe the image and identify important visual elements relevant to the question.
- For each region, describe it concisely and provide its bounding box coordinates in the format [Region (x1, y1, x2, y2)], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
- Coordinates must be absolute image regions formatted as: (x1, y1, x2, y2).
- Reason about a region's relevance to the question and—if visible—its relation to prior steps.
- At the end, summarize the conclusion and provide the answer enclosed in <answer> tags.

** Output format **:

<think>
[Step-by-step reasoning, referencing regions as [Region (x_min, y_min, x_max, y_max)] and connecting observations to answer the question.]
</think>
<answer>
[Final answer to the question based on the reasoning above.]
</answer>
'''

SPATIAL_PROMPT = '''
You are a helpful visual assistant. Given an image and a question about spatial relationship, carefully observe the image, identify important visual elements, and reason step by step to answer the question. You must systematically examine and verify relevant regions of the image, grounding the reasoning steps to specific (x1, y1, x2, y2) regions.

** Instructions **:

- Observe the image and identify important visual elements relevant to the question.
- Focus on the spatial relationships between the target objects mentioned in the question.
- Examine each region to determine the spatial relationship asked in the question, describe it concisely and provide its bounding box coordinates in the format [Region (x1, y1, x2, y2)], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
- Coordinates must be absolute image regions formatted as: (x1, y1, x2, y2).
- Reason about a region's relevance to the question and—if visible—its relation to prior steps.
- Consider the target object locations when reasoning about their spatial relationships
- At the end, summarize the conclusion and provide the answer enclosed in <answer> tags.

** Output format **:

<think>
[Step-by-step reasoning, referencing regions as [Region (x_min, y_min, x_max, y_max)] and connecting observations to answer the question.]
</think>
<answer>
[Final answer to the question based on the reasoning above.]
</answer>
'''

ATTRIBUTE_PROMPT = '''
You are a helpful visual assistant. Given an image and a question about visual attribute, carefully observe the image, identify important visual elements, and reason step by step to answer the question. You must systematically examine and verify relevant regions of the image, grounding the reasoning steps to specific (x1, y1, x2, y2) regions.

** Instructions **:

- Observe the image and identify important visual elements relevant to the question.
- Focus on the visual attributes of the target object mentioned in the question
- For each region, describe it concisely and provide its bounding box coordinates in the format [Region (x1, y1, x2, y2)], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
- Coordinates must be absolute image regions formatted as: (x1, y1, x2, y2).
- Reason about a region's relevance to the question and—if visible—its relation to prior steps.
- At the end, summarize the conclusion and provide the answer enclosed in <answer> tags.

** Output format **:

<think>
[Step-by-step reasoning, referencing regions as [Region (x_min, y_min, x_max, y_max)] and connecting observations to answer the question.]
</think>
<answer>
[Final answer to the question based on the reasoning above.]
</answer>
'''

# negative问题prompt里应该不能指出是negative，所以跟qa保持一致？
NEGATIVE_PROMPT = '''
You are a helpful visual assistant. Given an image and a question, carefully observe the image, identify important visual elements, and reason step by step to answer the question. You must systematically examine and verify relevant regions of the image, grounding the reasoning steps to specific (x1, y1, x2, y2) regions.

** Instructions **:

- Observe the image and identify important visual elements relevant to the question.
- For each region, describe it concisely and provide its bounding box coordinates in the format [Region (x1, y1, x2, y2)], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
- Coordinates must be absolute image regions formatted as: (x1, y1, x2, y2).
- Reason about a region's relevance to the question and—if visible—its relation to prior steps.
- At the end, summarize the conclusion and provide the answer enclosed in <answer> tags.

** Output format **:

<think>
[Step-by-step reasoning, referencing regions as [Region (x_min, y_min, x_max, y_max)] and connecting observations to answer the question.]
</think>
<answer>
[Final answer to the question based on the reasoning above.]
</answer>
'''

WEBGROUNDING_PROMPT = '''
A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant systematically reasons through the problem step by step, verifying each step and grounding every step to a specific point in the image.

All reasoning processes must be enclosed within a single set of '<think>' tags, with each reasoning step explicitly referencing a coordinate:

<think>
Step-by-step reasoning, referencing regions as [Region (x_min, y_min, x_max, y_max)] and connecting observations to answer the question.
</think>

The final answer should be enclosed in '<answer>' tags in the format:\n<answer> (xf, yf) </answer>

Your task is to help the user identify the precise coordinates (x, y) of a specific area/element/object on the screen based on a description.
- Aim to point to the center or a representative point within the described area/element/object as accurately as possible.
- If the description is unclear or ambiguous, infer the most relevant area or element based on its likely context or purpose.
- The final output should be the single most precise coordinate for the requested element.
- The Assistant should verify each step and check multiple possible solutions before selecting the final answer.
'''

WEBACTION_PROMPT = '''
You are a helpful Assistant tasked with navigating a web browser. These tasks will be accomplished through the use of specific actions you can issue. Your task is to choose the action that makes the most progress towards an objective. You should systematically reason through the problem step by step by checking and verifying possible actions and webpage regions, while grounding reasoning steps to specific (x1, y1, x2, y2) regions in the image:
Each reasoning step must be enclosed within '<think>' tags and reference exactly specific regions (x1, x2, y1, y2):
<think>
Step-by-step reasoning, referencing regions as [Region (x_min, y_min, x_max, y_max)] and connecting observations to answer the question.
</think>
When ready to provide the final answer, enclose it within '<answer>' tags:
<answer> {action} </answer>
- Each reasoning step must explicitly describe and evaluate the region’s relevance to the objective and proposing an action.
- Never repeat coordinates from previous steps.
- Look at diverse webpage regions to figure out which action should be taken.
- Verify your selection by examining multiple possible solutions.

**Inputs**
Here's the information you'll have:\n1. OBJECTIVE: This is the task you are trying to complete.\n2. The web page screenshot: This is a screenshot of the current webpage you are on, with each interactable element assigned a unique numerical id. Each bounding box and its respective id shares the same color.\n3. PREVIOUS ACTIONS: This is the actions that you have performed prior to getting to the current page, but instead of the button id, the button text of the actions taken on the previously navigated pages are provided.\n\n**Action Space**\nYou can take the following actions:\n1. ```click [id]```: This action clicks on an element with a specific id on the webpage.\n2. ```type [id] [content]```: Use this to type the content into the field with id. By default, typing the content simulates pressing the "Enter" key afterward to submit the text.\n3. ```scroll [down]```: Scroll the page down.\n4. ```go_back```: Navigate to the previously viewed page.\n5. ```stop [answer]```: Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If no answer is required, output empty brackets.\n\n**Guidelines**\nTo be successful, it is very important to follow the following rules:\n2. Generate the final action in the correct format. For example, '<answer> click [1234] </answer>'.\n3. Issue the stop action (i.e. stop [answer]) when you think you have achieved the objective. Don't generate anything after stop.\n4. In your final answer, you should only output a single action and should never output a prediction involving taking multiple actions.
'''

# VSTAR_MULTITURN_PROMPT = '''
# You are a helpful assistant tasked with answering a question about an image. You should systematically examine different regions and phrases in the image by requesting to see specific bounding box regions:\n- At each turn, first reason about what you want to examine enclosed in <think> </think> tags.\n- Then request to see a specific region by outputting a search action formatted as:\n     <tool_call>\n     {\"name\": \"search_bbox_region\", \"arguments\": {\"bbox\": [x1, y1, x2, y2], \"phrase\": \"phrase_description\"}}\n     </tool_call>\n- After examining all relevant regions, provide your final answer enclosed in <answer> {final answer} </answer> tags.\n- Use the information from each region to build comprehensive understanding before answering.
# '''

VSTAR_MULTITURN_PROMPT = '''
You are a helpful assistant tasked with answering a question about an image. You should systematically examine different regions and phrases in the image by requesting to see specific bounding box regions:\n- At each turn, first reason about what you want to examine enclosed in <think> </think> tags.\n- Then request to see a specific region by outputting a search action formatted as:\n     <tool_call>(x1, y1, x2, y2)</tool_call>\n- After examining all relevant regions, provide your final answer enclosed in <answer> {final answer} </answer> tags.\n- Use the information from each region to build comprehensive understanding before answering.
'''

SYSTEM_PROMPT = {
    "vstar": GQA_PROMPT,
    "vaw": ATTRIBUTE_PROMPT,
    "spatial": SPATIAL_PROMPT,
    "negative": NEGATIVE_PROMPT,
    "webgrounding": WEBGROUNDING_PROMPT,
    "webaction": WEBACTION_PROMPT,

    "vstar_multiturn": VSTAR_MULTITURN_PROMPT,
    "vstar_multiturn_trajformat": VSTAR_MULTITURN_PROMPT,
}


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


class ImageProcessMixin:
    max_pixels: int
    min_pixels: int
    max_side_length: int
    min_side_length: int
    image_root: Optional[str]

    def process_image(self, image: Union[Dict[str, Any], ImageObject]) -> ImageObject:
        if isinstance(image, dict):
            image = Image.open(BytesIO(image["bytes"]))
        elif isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        elif isinstance(image, str):
            if self.image_root is not None:
                image = os.path.join(self.image_root, image)
            image = Image.open(image)

        if self.max_side_length is not None and max(image.width, image.height) > self.max_side_length:
            sf = self.max_side_length / max(image.width, image.height)
            image =fetch_image({"image": image, "resized_width": int(image.width * sf), "resized_height": int(image.height * sf)})

        if self.min_side_length is not None and min(image.width, image.height) < self.min_side_length:
            sf = self.min_side_length / min(image.width, image.height)
            image =fetch_image({"image": image, "resized_width": int(image.width * sf), "resized_height": int(image.height * sf)})

        if (image.width * image.height) > self.max_pixels:
            image = fetch_image({"image": image, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels})

        if (image.width * image.height) < self.min_pixels:
            image = fetch_image({"image": image, "min_pixels": self.min_pixels, "max_pixels": self.max_pixels})

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image


class RLHFDataset(Dataset, ImageProcessMixin):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str, # 这里可以是单个 jsonl/parquet/dataset，也可以是 txt
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        max_side_length: Optional[int] = None,
        min_side_length: Optional[int] = None,
        image_root: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.filter_overlong_prompts = filter_overlong_prompts
        self.max_side_length = max_side_length
        self.min_side_length = min_side_length
        self.image_root = image_root
        # --------------------------
        # 1. 判断 data_path 是否是 txt
        # --------------------------
        self.data_path = data_path
        all_datasets = []
        if data_path.endswith(".txt"):
            with open(data_path, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                if line.count("&") == 1:
                    task_type, path = [x.strip() for x in line.split("&")]
                    dataset = self._load_single_dataset(path)
                    # 给 dataset 增加一个 "task_type" 字段
                    dataset = dataset.add_column("task_type", [task_type] * len(dataset))
                else:
                    raise ValueError(f"Each line in txt should contain one '&' to separate task_type and data_path, but got: {line}")
                    
                all_datasets.append(dataset)

            # 合并多个子数据集
            from datasets import concatenate_datasets
            self.dataset = concatenate_datasets(all_datasets)

        else:
            # 单数据集情况，兼容原始逻辑
            dataset = self._load_single_dataset(data_path)
            dataset = dataset.add_column("task_type", ["default"] * len(dataset))
            self.dataset = dataset

        # format_prompt
        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()
            print(f"Format prompt: {self.format_prompt}")

        if self.filter_overlong_prompts:
            self.dataset = self.dataset.filter(self._filter_overlong_prompts, desc="Filtering overlong prompts")

        print(f"Image processor: {self.processor.image_processor.__class__.__name__}")

    def _load_single_dataset(self, path: str):
        """根据路径加载单个数据集"""
        if "@" in path:
            path, data_split = path.split("@")
        else:
            data_split = "train"

        if path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=path, split=data_split)
        elif os.path.isdir(path):
            dataset = load_dataset("parquet", data_dir=path, split="train")
        elif os.path.isfile(path):
            dataset = load_dataset("parquet", data_files=path, split="train")
        else:
            dataset = load_dataset(path, split=data_split)
        return dataset

    def _build_messages(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        messages = []
        if "task_type" in example and example["task_type"] in SYSTEM_PROMPT:
            system_prompt = SYSTEM_PROMPT[example["task_type"]]
            messages.append({"role": "system", "content": system_prompt.strip()})
        elif self.format_prompt:
            messages.append({"role": "system", "content": self.format_prompt.strip()})

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return messages + [{"role": "user", "content": content_list}]
        else:
            return messages + [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_messages(example)
        processing_class = self.processor if self.processor is not None else self.tokenizer
        return (
            len(processing_class.apply_chat_template(messages, add_generation_prompt=True)) <= self.max_prompt_length
        )

    def __len__(self):
        return len(self.dataset)
    
    @property
    def task_types(self):
        task_list = []
        for sample in self.dataset:
            task_type = sample.get("task_type", "general")
            task_list.append(task_type)
        return task_list

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            image_paths = example.pop(self.image_key)
            images = [self.process_image(image) for image in image_paths]
            model_inputs = self.processor(images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"image": images}
            if 'multiturn' in example['task_type']:
                example["multi_modal_data"]['image_path'] = os.path.join(self.image_root, image_paths[0]) 
                example["multi_modal_data"]["input_width"], example["multi_modal_data"]["input_height"] = images[0].size
            example["multi_modal_inputs"] = dict(model_inputs)
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and self.processor.image_processor.__class__.__name__ in ["Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast"]:
            # qwen2vl mrope
            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw"),
                attention_mask=attention_mask,
            )  # (3, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        example["problem"] = example.pop(self.prompt_key)
        return example
