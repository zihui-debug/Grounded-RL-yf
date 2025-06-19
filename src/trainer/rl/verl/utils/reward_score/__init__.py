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


from .math import math_compute_score
from .r1v import r1v_compute_score
from .uground import uground_compute_score
from .uground_intermground import uground_intermediate_ground_compute_score
from .sat import sat_compute_score
from .point_in_bbox import point_in_bbox_compute_score
from .point_in_bbox_multicrop import point_in_bbox_multicrop_compute_score
from .web_action import web_action_compute_score
__all__ = [
    "math_compute_score", 
    "r1v_compute_score", 
    "uground_compute_score", 
    "uground_intermediate_ground_compute_score",
    "sat_compute_score",
    "point_in_bbox_compute_score", 
    "point_in_bbox_multicrop_compute_score",
    "web_action_compute_score",
    ]
