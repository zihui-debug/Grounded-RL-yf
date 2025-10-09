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

from typing import Optional, List, Dict
import random

import torch
from torch.utils.data import RandomSampler, SequentialSampler, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig

class TaskGroupedSampler(Sampler):
    """
    Sampler that ensures each batch only contains samples from the same task type.
    Supports both random.Random and torch.Generator for reproducibility.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        task_types: List[str],   # 每个样本对应的任务类型 (len == dataset size)
        generator=None,
    ):
        if task_types is None:
            raise ValueError("Task types must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.task_types = task_types
        self.generator = generator

        # 任务 -> indices 映射
        self.task_to_indices: Dict[str, List[int]] = {}
        for idx, task in enumerate(task_types):
            self.task_to_indices.setdefault(task, []).append(idx)

    def __len__(self):
        return len(self.task_types)

    def _shuffle(self, indices: List[int]) -> List[int]:
        """根据 generator 类型决定 shuffle 方式"""
        if isinstance(self.generator, torch.Generator):
            perm = torch.randperm(len(indices), generator=self.generator).tolist()
            return [indices[i] for i in perm]
        else:
            indices = indices[:]
            random.Random(self.generator).shuffle(indices) if isinstance(self.generator, int) else random.shuffle(indices)
            return indices

    def __iter__(self):
        all_batches = []

        # 每个任务池单独 shuffle，并划分 batch
        for task, indices in self.task_to_indices.items():
            shuffled = self._shuffle(indices)

            # 按 (batch_size * world_size) 切块
            for i in range(0, len(shuffled), self.batch_size * self.world_size):
                chunk = shuffled[i:i + self.batch_size * self.world_size]

                # 再切分成 world_size 个 batch（用于分布式）
                for j in range(0, len(chunk), self.batch_size):
                    batch = chunk[j:j + self.batch_size]
                    if len(batch) == self.batch_size:
                        all_batches.append(batch)

        # 打乱 batch 顺序（否则所有同任务 batch 会连在一起）
        all_batches = self._shuffle(all_batches)

        # 展开成 index 序列
        return iter([idx for batch in all_batches for idx in batch])


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        max_side_length=config.max_side_length,
        min_side_length=config.min_side_length,
        image_root=config.image_root,
    )
    # use sampler for better ckpt resume
    # TODO: 加上TaskGroupSampler
    if config.group_by_task:
        # assert processor is not None, "Processor must be provided when grouping by task."
        task_types = train_dataset.task_types
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = TaskGroupedSampler(
            batch_size=config.rollout_batch_size,
            world_size=1, #config.rollout_batch_size是global batch size
            task_types=task_types,
            generator=train_dataloader_generator,
        )
    elif config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=config.rollout_batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        max_side_length=config.max_side_length,
        min_side_length=config.min_side_length,
        image_root=config.image_root,
    )
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=len(val_dataset) if config.val_batch_size == -1 else config.val_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    print(f"data path: {config.train_files}")
    print(f"data path: {config.val_files}")

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    return train_dataloader, val_dataloader