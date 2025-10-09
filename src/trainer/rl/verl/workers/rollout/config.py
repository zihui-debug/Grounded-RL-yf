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
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 65536
    disable_log_stats: bool = True
    val_override_config: Dict[str, Any] = field(default_factory=dict)
    """auto keys"""
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    max_generation_length_per_turn: int = field(default=-1, init=False)
    multiturn: bool = False
    max_iterations: int = 10
    trust_remote_code: bool = field(default=False, init=False)
    max_pixels: int = 4194304
    min_pixels: int = 262144
    crop_size: int = 512
    draw_dot: bool = True
    offset: int = 50
    stop_strings: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)
