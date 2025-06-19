import os
import re
from contextlib import contextmanager
from typing import Any, List, Union, Optional, Tuple

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from PIL import Image, ImageDraw
import json
import copy
from tqdm import tqdm
import ast

from ....protocol import DataProto
from ..config import RolloutConfig
from ....models.transformers.qwen2_vl import get_rope_index
from ....utils import torch_functional as VF

from qwen_vl_utils import fetch_image

import logging
logger = logging.getLogger(__name__)

# regex to pull out only the structural tags we care about
_TAG_RE = re.compile(r"<(/?)(tool_call|observation|think|answer)>", re.IGNORECASE)
_TOOL_JSON_RE = re.compile(r"<tool_call>\s*({.*?})\s*</tool_call>", re.DOTALL)

def format_reward(predict_str: str) -> float:
    s = predict_str.strip()

    # 1) must finish with </answer> (optionally followed by one EOS token)
    if not re.search(r"</answer>\s*(<\|im_end\|>)?\s*$", s, re.DOTALL):
        return 0.0

    # 2) walk through the high‑level tag sequence to enforce grammar
    tags_iter = _TAG_RE.finditer(s)
    state = "think_open"            # expected next tag
    for m in tags_iter:
        tag = m.group(0).lower()

        if state == "tool_open":
            if tag != "<tool_call>":
                return 0.0
            state = "tool_close"

        elif state == "tool_close":
            if tag != "</tool_call>":
                return 0.0
            state = "obs_open"

        elif state == "obs_open":
            if tag != "<observation>":
                return 0.0
            state = "obs_close"

        elif state == "obs_close":
            if tag != "</observation>":
                return 0.0
            state = "think_open"

        elif state == "think_open":
            # if tag == "<answer>":
            #     # allowed right after observation
            #     # REMOVE THIS if statement if do not want to allow answer right after observation
            #     state = "answer_close"
            #     continue
            if tag != "<think>":
                return 0.0
            state = "think_close"

        elif state == "think_close":
            if tag != "</think>":
                return 0.0
            state = "post_think"

        elif state == "post_think":
            if tag == "<tool_call>":
                state = "tool_close"         # start another round
            elif tag == "<answer>":
                state = "answer_close"
            else:
                return 0.0

        elif state == "answer_close":
            if tag != "</answer>":
                return 0.0
            state = "end"

        elif state == "end":
            # no structural tags allowed after </answer>
            return 0.0

    if state != "end":
        return 0.0   # we never saw a complete <answer> … </answer> block

    # 3) validate each <tool_call> JSON and coordinate schema
    for m in _TOOL_JSON_RE.finditer(s):
        try:
            obj = json.loads(m.group(1))
            coord = obj.get("arguments", {}).get("coordinate", None)
            if (not isinstance(coord, list) or len(coord) != 2 or
                not all(isinstance(x, int) for x in coord)):
                return 0.0
        except Exception:
            return 0.0

    # 4) validate final answer is a tuple of two ints
    ans_match = re.search(r"<answer>\s*\(([^)]*)\)\s*</answer>", s)
    if not ans_match:
        return 0.0
    try:
        ans_tuple = ast.literal_eval("(" + ans_match.group(1).strip() + ")")
        if (not isinstance(ans_tuple, tuple) or len(ans_tuple) != 2 or
            not all(isinstance(x, int) for x in ans_tuple)):
            return 0.0
    except Exception:
        return 0.0

    return 1.0

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

def get_eos_mask_last(response_ids: torch.Tensor, eos_token_id: Union[int, List[int]] = 2, dtype: torch.dtype = torch.long):
    """Get the mask for the response ids, the mask will be 0 after the last eos token.

    eos_token_id can be int or list: 1 or [1, 2].
    ```
    e.g. eos_token = 1
    response_ids: [0, 0, 2, 4, 3, 5, 1, 0, 0]
    eos_mask:     [1, 1, 1, 1, 1, 1, 1, 0, 0]
    ```
    """
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]

    # Create a mask that marks all EOS token positions
    eos_mask = torch.zeros_like(response_ids, dtype=torch.bool)
    for token_id in eos_token_id:
        eos_mask |= response_ids.eq(token_id)
    
    # For each sequence in the batch, find the position of the last EOS token
    batch_size = response_ids.shape[0]
    result_mask = torch.ones_like(response_ids, dtype=dtype)
    
    for i in range(batch_size):
        # Find positions of all EOS tokens in this sequence
        eos_positions = torch.nonzero(eos_mask[i], as_tuple=True)[0]
        if len(eos_positions) > 0:
            # Get the position of the last EOS token
            last_eos_pos = eos_positions[-1]
            # Set mask to 0 for all positions after the last EOS token
            result_mask[i, last_eos_pos+1:] = 0
    
    return result_mask

def _parse_coordinate(text: str):
    """
    Extract a coordinate from text of the form '(x, y)'.
    Returns (x, y) as a tuple of ints if found, otherwise None.
    """
    try:
        text = text.strip()
        json_text = json.loads(text)
        return json_text["arguments"]["coordinate"]
    except Exception as e:
        return None

def _get_point_crop(image: Image.Image, point: List[int], offset: int = 50, crop_size: int = 512, draw_dot: bool = True) -> Image.Image:
    """
    Get a crop of the image centered on the point with a side length of 2*offset.
    Also draws a dot at the point location within the crop.
    """
    x, y = point
    width, height = image.size

    # Ensure crop boundaries are within image dimensions
    left = max(0, x - offset)
    top = max(0, y - offset)
    right = min(width, x + offset)
    bottom = min(height, y + offset)
    
    # Ensure that right > left and bottom > top
    if right <= left:
        right = left + 1
    if bottom <= top:
        bottom = top + 1
    
    # Create the crop
    crop = image.crop((left, top, right, bottom))

    if draw_dot:
        # Draw a dot at the point location (relative to the crop)
        draw = ImageDraw.Draw(crop)
        dot_x = x - left
        dot_y = y - top
        radius = 7
        draw.ellipse(
            [(dot_x - radius, dot_y - radius), (dot_x + radius, dot_y + radius)],
            fill="red",
            outline="white",
            width=2
        )

    # Super-resolution to crop_size x crop_size
    # crop = crop.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    crop = fetch_image({"image": crop, "resized_width": crop_size, "resized_height": crop_size})

    return crop

class RolloutMultiturn:
    """
    A light‑weight coordinator that repeatedly calls an underlying single‑turn
    vLLMRollout until every conversation in the batch emits an <answer>…</answer>.
    Nothing inside vLLMRollout is modified, so TP / PP sharding still works.
    """
    TOOL_RE   = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    ANSWER_RE = re.compile(r"<answer>(.*?)</answer>",      re.DOTALL)

    def __init__(
        self,
        actor_rollout_wg: "vLLMRollout",          # local object or Ray‑actor‑handle
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
        n: int = 1,
        offset: int = 50
    ):
        self.rollout   = actor_rollout_wg        # keep a reference
        self.cfg       = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.pad_id    = tokenizer.pad_token_id
        self.n         = n
        # self.offset    = offset

        print(f"Crop size: {self.cfg.crop_size}")
        print(f"Offset: {self.cfg.offset}")
        print(f"Stop strings: {self.cfg.stop_strings.split(',')}")

        # tokens that should receive loss‑mask = 0
        toks_to_mask = ["<|im_start|>", "<|im_start|>user",
                        "<|im_start|>assistant", "<|image_pad|>", 
                        "<|vision_end|>", "<|vision_start|>",
                        "<|im_start|>assistant\n<think> I've reached the maximum number of image crops allowed. Based on all the information I've gathered, I'll now provide my final coordinate answer. I will note that it can be different from the previous coordinates I provided and should represent my best guess of the correct coordinate. </think>\n"
                        ]
        self.mask_seq_ids = [
            tokenizer.encode(s, add_special_tokens=False) for s in toks_to_mask
        ]

        self.eos_token_id = self.tokenizer.eos_token_id
        self.eos_txt = self.tokenizer.decode(self.eos_token_id)
        self.vocab_size = len(self.tokenizer)

        # vLLM sharding requires equal chunks, so need to feed in dummy ids
        # TODO: this is a hack, we should find a better way to do this
        self.dummy_ids = tokenizer.encode("<|im_start|>user\nOutput nothing.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    # ------------------------------------------------------------------ helpers
    def _append_observation(
        self,
        prompt_ids: List[int],
        response_ids: List[int],
        img_list   : List[Image.Image],
        coord      : Tuple[int, int],
        offset     : int = 50,
        crop_size  : int = 512,
        draw_dot   : bool = True
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Adds an <observation> section with a newly cropped image and returns
        (extended_prompt_ids, loss_mask_for_these_new_tokens)
        """
        try:
            crop = _get_point_crop(img_list[0], coord, offset=offset, crop_size=crop_size, draw_dot=draw_dot)
        except Exception as e:
            return prompt_ids, response_ids, img_list

        if len(img_list) == self.cfg.limit_images - 1:
            # last image – immediately open an <answer> tag so the model can stop
            # obs_txt = "\n<|im_start|>user\n<observation>\nHere is the crop of the image centered on the coordinate, with a red dot at the coordinate location:\n<|vision_start|><|image_pad|><|vision_end|></observation><|im_end|>\n<|im_start|>assistant\n<think> I've reached the maximum number of image crops allowed. Based on all the information I've gathered, I'll now provide my final coordinate answer. I will note that it can be different from the previous coordinates I provided and should represent my best guess of the correct coordinate. </think>\n<answer>"
            if draw_dot:
                obs_txt = "\n<|im_start|>user\n<observation>\nHere is the crop of the image centered on the coordinate, with a red dot at the coordinate location:\n<|vision_start|><|image_pad|><|vision_end|></observation><|im_end|>\n<|im_start|>assistant\n<think> Based on all the information I've gathered, I'll now provide my final answer. </think>\n<answer>"
            else:
                obs_txt = "\n<|im_start|>user\n<observation>\nHere is the crop of the image centered on the coordinate:\n<|vision_start|><|image_pad|><|vision_end|></observation><|im_end|>\n<|im_start|>assistant\n<think> Based on all the information I've gathered, I'll now provide my final answer. </think>\n<answer>"
        else:
            if draw_dot:
                obs_txt = "\n<|im_start|>user\n<observation>\nHere is the crop of the image centered on the coordinate, with a red dot at the coordinate location:\n<|vision_start|><|image_pad|><|vision_end|></observation><|im_end|>\n<|im_start|>assistant\n"
            else:
                obs_txt = "\n<|im_start|>user\n<observation>\nHere is the crop of the image centered on the coordinate:\n<|vision_start|><|image_pad|><|vision_end|></observation><|im_end|>\n<|im_start|>assistant\n"

        new_ids = self.tokenizer.encode(obs_txt, add_special_tokens=False)

        # Filter out any token IDs that are outside the tokenizer's vocabulary
        new_ids = [token_id for token_id in new_ids if token_id < self.vocab_size]

        image_tokens = int(crop.size[0] * crop.size[1] / (28*28)) # qwen uses 28x28 patches, TODO: make this an argument

        if len(response_ids) + len(new_ids) + image_tokens >= self.cfg.response_length:
            # truncate the response
            return prompt_ids, response_ids, img_list

        prompt_ids.extend(new_ids)
        response_ids.extend(new_ids)
        img_list.append(crop) 
        return prompt_ids, response_ids, img_list

    def _response_loss_mask(
        self,
        resp_ids: List[int],
        # eos_id: int,
        device: torch.device,
        num_mask_left: int = 0
    ) -> torch.Tensor:

        ids = torch.tensor(resp_ids, device=device)
        mask = torch.ones_like(ids)

        mask[:num_mask_left] = 0

        # Tokenize common whitespace tokens explicitly
        whitespace_ids = [
            self.tokenizer.encode(" ", add_special_tokens=False),
            self.tokenizer.encode("\n", add_special_tokens=False),
            self.tokenizer.encode("\n\n", add_special_tokens=False),
        ]

        # Helper to find spans matching multiple candidate patterns
        def find_spans(patterns: List[List[int]], allow_trailing_ws: bool=False) -> List[Tuple[int, int]]:
            spans = []
            for pat in patterns:
                pat_t = torch.tensor(pat, device=device)
                k = len(pat_t)
                if k > len(ids):
                    continue
                windows = ids.unfold(0, k, 1)
                hits = (windows == pat_t).all(dim=1)
                starts = hits.nonzero(as_tuple=False).squeeze(-1).tolist()
                for s in starts:
                    e = s + k
                    if allow_trailing_ws:
                        # Greedily expand span to include subsequent whitespace tokens
                        expanded = True
                        while expanded and e < len(ids):
                            expanded = False
                            for ws in whitespace_ids:
                                ws_t = torch.tensor(ws, device=device)
                                ws_len = len(ws_t)
                                if e + ws_len <= len(ids) and torch.equal(ids[e:e+ws_len], ws_t):
                                    e += ws_len
                                    expanded = True
                                    break
                    spans.append((s, e))
            spans.sort()
            return spans

        # Mask EOS tokens
        # mask[ids == eos_id] = 0

        # Mask explicit meta tokens, allowing trailing whitespace
        for pat in self.mask_seq_ids:
            spans = find_spans([pat], allow_trailing_ws=True)
            for s, e in spans:
                mask[s:e] = 0

        # # Mask all <observation>...</observation> spans (both newline and no-newline variants)
        # open_tags = [
        #     self.tokenizer.encode("<observation>\n", add_special_tokens=False),
        #     self.tokenizer.encode("<observation>", add_special_tokens=False),
        # ]
        # close_tags = [
        #     self.tokenizer.encode("</observation>\n", add_special_tokens=False),
        #     self.tokenizer.encode("</observation>", add_special_tokens=False),
        # ]

        # Mask all <observation>...</observation> spans (both newline and no-newline variants)
        open_tags = [
            self.tokenizer.encode("<observation>\n", add_special_tokens=False),
            # self.tokenizer.encode("<observation>", add_special_tokens=False),
        ]
        close_tags = [
            self.tokenizer.encode("</observation><|im_end|>\n", add_special_tokens=False),
            # self.tokenizer.encode("</observation>", add_special_tokens=False),
        ]

        open_spans = find_spans(open_tags, allow_trailing_ws=True)
        close_spans = find_spans(close_tags, allow_trailing_ws=True)

        # Pair opens and closes sequentially
        close_iter = iter(close_spans)
        current_close = next(close_iter, None)

        for o_start, o_end in open_spans:
            while current_close and current_close[0] < o_start:
                current_close = next(close_iter, None)
            if current_close is None:
                break
            mask[o_start:current_close[1]] = 0
            current_close = next(close_iter, None)

        return mask

    def _strip_pad_tokens(self, tokens: List[int]) -> List[int]:

        # Filter out any token IDs that are outside the tokenizer's vocabulary
        tokens = [token_id for token_id in tokens if token_id < self.vocab_size]

        return [t for t in tokens if t != self.pad_id]

    @contextmanager
    def update_n(self, **kwargs):
        # update sampling params
        if 'n' in kwargs:
            old_n = self.n
            self.n = kwargs['n']
        else:
            old_n = self.n

        yield

        # roll back to previous sampling params
        self.n = old_n

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Multi‑turn roll‑out.  The tensors provided are *only* the initial prompt
        (turn‑0).  We keep extending them in Python lists, and only when a whole
        batch‑turn is ready we re‑pack into a DataProto and delegate to
        `self.rollout.generate_sequences`.
        """

        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = self.eos_token_id
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        device     = input_ids.device
        batch_size = input_ids.size(0)

        # ------------------------------------------------ conversation state
        # Every list below has length == batch_size
        # input_ids   = [ids.tolist() for ids in input_ids]             # running prompt
        prompt_ids   = [ids for ids in non_tensor_batch["raw_prompt_ids"]]             # running prompt
        img_buffers  = [md["image"] if "image" in md else []           # keep PIL images
                        for md in non_tensor_batch["multi_modal_data"]]
        
        with self.update_n(**prompts.meta_info):
            print(f"n is set to: {self.n}")
            if self.n > 1:
                batch_size = batch_size * self.n
                input_ids = _repeat_interleave(input_ids, self.n)
                attention_mask = _repeat_interleave(attention_mask, self.n)
                position_ids = _repeat_interleave(position_ids, self.n)
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.n
                    )
                prompt_ids_ = []
                for i in range(len(prompt_ids)):
                    for _ in range(self.n):
                        prompt_ids_.append(prompt_ids[i].copy())
                prompt_ids = prompt_ids_
                img_buffers_ = []
                for i in range(len(img_buffers)):
                    for _ in range(self.n):
                        img_buffers_.append([im.copy() for im in img_buffers[i]])
                img_buffers = img_buffers_

        finished     = [False] * batch_size
        resp_tokens  = [[]      for _ in range(batch_size)]            # accumulated response ids

        try:
            self.rollout.begin_conversation()

            turn = 0
            while not all(finished) and turn < self.cfg.max_iterations:
                turn += 1

                # ------------------------------------------------ prepare active batch
                active_idx = [i for i, done in enumerate(finished) if not done]
                inactive_idx = [i for i, done in enumerate(finished) if done]
                if not active_idx:
                    break

                # directly create a single DataProto with just the active indices
                active_count = len(active_idx)

                # Get active tensors
                input_ids_active = input_ids[active_idx]
                attention_mask_active = attention_mask[active_idx]
                position_ids_active = position_ids[active_idx]
                
                # Create TensorDict with active tensors + dummy tensors for inactive indices
                if len(position_ids_active.shape) == 3:
                    active_td = TensorDict({
                        "input_ids": torch.cat([input_ids_active, torch.zeros(len(inactive_idx), input_ids_active.size(1), device=device)]),
                        "attention_mask": torch.cat([attention_mask_active, torch.zeros(len(inactive_idx), attention_mask_active.size(1), device=device)]),
                        "position_ids": torch.cat([position_ids_active, torch.zeros([len(inactive_idx), 3, position_ids_active.size(2)], device=device)])
                    }, batch_size=(batch_size,))
                else:
                    active_td = TensorDict({
                        "input_ids": torch.cat([input_ids_active, torch.zeros(len(inactive_idx), input_ids_active.size(1), device=device)]),
                        "attention_mask": torch.cat([attention_mask_active, torch.zeros(len(inactive_idx), attention_mask_active.size(1), device=device)]),
                        "position_ids": torch.cat([position_ids_active, torch.zeros(len(inactive_idx), position_ids_active.size(1), device=device)])
                    }, batch_size=(batch_size,))
                
                # Gather non-tensor batch data for active indices
                active_nt = {
                    "multi_modal_inputs": np.array([non_tensor_batch["multi_modal_inputs"][i] for i in active_idx] + [{'pixel_values': [], 'image_grid_thw': []}] * len(inactive_idx)),
                    "raw_prompt_ids": np.array([prompt_ids[i] for i in active_idx] + [self.dummy_ids] * len(inactive_idx), dtype=object),
                    "multi_modal_data": np.array([{'image': img_buffers[i]} for i in active_idx] + [{'image': []} for _ in inactive_idx])
                }
                
                # Create a single DataProto for active indices
                active_dp = DataProto(
                    batch=active_td, 
                    non_tensor_batch=active_nt,
                    meta_info={"eos_token_id": eos_token_id}
                )

                # ------------------------------------------------ single‑turn generation
                # out_dp = self.rollout.generate_sequences(active_dp)
                out_dp = self.rollout.generate_step(active_dp)

                # vLLMRollout returns exactly one continuation per item (n==1)
                out_ids  = out_dp.batch["responses"].cpu().tolist()
                # ------------------------------------------------ post‑process each item
                for local_idx, i in enumerate(active_idx):
                    toks = out_ids[local_idx]
                    toks = self._strip_pad_tokens(toks)

                    patch_size = 28
                    image_tokens = int(((self.cfg.crop_size * self.cfg.crop_size) / (patch_size*patch_size)) * (len(img_buffers[i])-1))

                    if len(resp_tokens[i]) + len(toks) + image_tokens >= self.cfg.response_length:
                        # truncate the response
                        finished[i] = True
                        continue
                    
                    resp_tokens[i].extend(toks)
                    prompt_ids[i].extend(toks)

                    txt = self.tokenizer.decode(toks)

                    if txt.endswith(tuple(self.cfg.stop_strings)) and not txt.endswith(self.eos_txt):
                        resp_tokens[i].extend([self.tokenizer.eos_token_id])
                        prompt_ids[i].extend([self.tokenizer.eos_token_id])
                        txt += self.eos_txt
                        # print(f"txt for {i}: {txt}")

                    # check for <answer>
                    if self.ANSWER_RE.search(txt) or "</answer>" in txt:
                        finished[i] = True
                        continue

                    if len(img_buffers[i]) == self.cfg.limit_images:
                        finished[i] = True
                        continue

                    # check for first tool_call (only one per turn)
                    m = self.TOOL_RE.search(txt)
                    if m and not finished[i]:
                        coord = _parse_coordinate(m.group(1))
                        if coord is not None:
                            prompt_ids[i], resp_tokens[i], img_buffers[i] = self._append_observation(
                                prompt_ids[i], resp_tokens[i], img_buffers[i], coord, 
                                offset=self.cfg.offset, crop_size=self.cfg.crop_size, 
                                draw_dot=self.cfg.draw_dot
                            )
                    else:
                        finished[i] = True

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e
        finally:
            self.rollout.end_conversation()
        
        ############################################################
        #### 2. Process multi-turn prompts
        ############################################################
        sequence_ids = []
        # sequence_loss_mask = []
        response_loss_mask = []
        position_ids = []
        attention_mask = []
        multi_modal_inputs = []
        # format_rewards = []
        for i in tqdm(
            range(len(prompt_ids)), 
            desc="Processing multi-turn responses", 
            leave=False
            ):

            images = [{"image": img, "max_pixels": 16777216, "min_pixels": 1024} for img in img_buffers[i]]
            images = [fetch_image(img) for img in images]

            if i == 0:
                print(f"Size of images: {[img.size for img in images]}")

            input_text = self.tokenizer.decode(prompt_ids[i])
            obs_inputs = self.processor(
                images=images,
                text=[input_text],
                add_special_tokens=False,
                return_tensors="pt",
                truncation=True,
                max_length=self.cfg.prompt_length + self.cfg.response_length,
            )
            num_pad_left = len(torch.where(input_ids[i]==self.pad_id)[0])
            num_input_tokens = len(input_ids[i]) - num_pad_left

            # format_rewards.append(format_reward(self.tokenizer.decode(resp_tokens[i])))
            
            # # Ensure we don't exceed the maximum sequence length
            # max_obs_length = self.cfg.response_length + self.cfg.prompt_length - num_pad_left
            # if obs_inputs["input_ids"][0].size(0) > max_obs_length:
            #     # Truncate the observation inputs if they exceed the maximum length
            #     obs_inputs["input_ids"] = obs_inputs["input_ids"][:, :max_obs_length]
            #     obs_inputs["attention_mask"] = obs_inputs["attention_mask"][:, :max_obs_length]
            #     # Truncate other tensors in obs_inputs if needed
            #     for k, v in obs_inputs.items():
            #         if isinstance(v, torch.Tensor) and v.size(-1) > max_obs_length and k not in ["input_ids", "attention_mask"]:
            #             obs_inputs[k] = v[..., :max_obs_length]
            
            padded_left_obs_inputs = torch.cat([
                torch.full((num_pad_left,), self.pad_id, device=obs_inputs["input_ids"][0].device, dtype=obs_inputs["input_ids"].dtype),
                obs_inputs["input_ids"][0]
            ])
            sequence_ids.append(padded_left_obs_inputs)
            loss_mask = self._response_loss_mask(
                obs_inputs["input_ids"][0], 
                # eos_token_id, 
                device,
                num_mask_left=num_input_tokens
            )
            response_mask = loss_mask[num_input_tokens:] # take only response tokens
            
            ############################################################
            # MASK PROMPT NOT ENDING WITH EOS TOKEN
            ############################################################
            if prompt_ids[i][-1]!=eos_token_id:
                # prompt doesnt end in eos token, mask the whole response
                response_mask = torch.zeros_like(response_mask)
                loss_mask = torch.zeros_like(loss_mask)
            ############################################################

            max_obs_length = self.cfg.response_length + self.cfg.prompt_length # - num_pad_left

            if self.processor is not None and self.processor.image_processor.__class__.__name__ in ["Qwen2VLImageProcessor", "Qwen2VLImageProcessorFast"]:
                # qwen2vl mrope
                position_ids_ = get_rope_index(
                    self.processor,
                    input_ids=obs_inputs["input_ids"][0],
                    image_grid_thw=obs_inputs["image_grid_thw"],
                    attention_mask=obs_inputs["attention_mask"][0],
                )

                # # Ensure we don't exceed the maximum sequence length
                # max_obs_length = self.cfg.response_length + self.cfg.prompt_length - num_pad_left
                # if obs_inputs["input_ids"][0].size(0) > max_obs_length:
                #     # Truncate the observation inputs if they exceed the maximum length
                #     obs_inputs["input_ids"] = obs_inputs["input_ids"][:, :max_obs_length]
                #     obs_inputs["attention_mask"] = obs_inputs["attention_mask"][:, :max_obs_length]
                #     position_ids_ = position_ids_[:, :max_obs_length]
                #     # Truncate other tensors in obs_inputs if needed
                #     for k, v in obs_inputs.items():
                #         if isinstance(v, torch.Tensor) and v.size(-1) > max_obs_length and k not in ["input_ids", "attention_mask"]:
                #             obs_inputs[k] = v[..., :max_obs_length]

                position_ids_ = torch.cat([
                    torch.zeros((3, num_pad_left), device=position_ids_.device, dtype=position_ids_.dtype),
                    position_ids_
                ], dim=1)
                # Calculate remaining length and ensure it's positive
                remaining_length = max(0, self.cfg.response_length + self.cfg.prompt_length - position_ids_.shape[1])
                position_ids_ = torch.cat([
                        position_ids_, 
                        torch.zeros(
                        3, 
                        remaining_length, 
                        device=position_ids_.device, 
                        dtype=position_ids_.dtype
                    )
                ], dim=1)
            else:
                position_ids_ = torch.clip(obs_inputs["attention_mask"][0].cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
                # Ensure we don't exceed the maximum sequence length
                
                # if obs_inputs["input_ids"][0].size(0) > max_obs_length:
                #     # Truncate the observation inputs if they exceed the maximum length
                #     obs_inputs["input_ids"] = obs_inputs["input_ids"][:, :max_obs_length]
                #     obs_inputs["attention_mask"] = obs_inputs["attention_mask"][:, :max_obs_length]
                #     position_ids_ = position_ids_[:max_obs_length]
                #     # Truncate other tensors in obs_inputs if needed
                #     for k, v in obs_inputs.items():
                #         if isinstance(v, torch.Tensor) and v.size(-1) > max_obs_length and k not in ["input_ids", "attention_mask"]:
                #             obs_inputs[k] = v[..., :max_obs_length]

                # pad position_ids_ with zeros
                position_ids_ = torch.cat([
                    torch.zeros(num_pad_left, device=position_ids_.device, dtype=position_ids_.dtype),
                    position_ids_
                ], dim=0)
                remaining_length = max(0, self.cfg.response_length + self.cfg.prompt_length - position_ids_.shape[0])
                position_ids_ = torch.cat([
                    position_ids_, 
                    torch.zeros(
                        remaining_length, 
                        device=position_ids_.device, dtype=position_ids_.dtype
                    )
                ], dim=0)

            # Ensure we don't exceed the maximum sequence length
            if obs_inputs["input_ids"][0].size(0) > max_obs_length or position_ids_.shape[-1] > max_obs_length:
                # Truncate the observation inputs if they exceed the maximum length
                obs_inputs["input_ids"] = obs_inputs["input_ids"][:, :max_obs_length]
                obs_inputs["attention_mask"] = obs_inputs["attention_mask"][:, :max_obs_length]
                if len(position_ids_.shape) == 2:
                    position_ids_ = position_ids_[:, :max_obs_length]
                else:
                    position_ids_ = position_ids_[:max_obs_length]
                # Truncate other tensors in obs_inputs if needed
                for k, v in obs_inputs.items():
                    if isinstance(v, torch.Tensor) and v.size(-1) > max_obs_length and k not in ["input_ids", "attention_mask"]:
                        obs_inputs[k] = v[..., :max_obs_length]

            # sequence_loss_mask.append(loss_mask)
            response_loss_mask.append(response_mask)
            position_ids.append(position_ids_)
            attention_mask_ = torch.cat([
                torch.zeros((num_pad_left), device=obs_inputs["attention_mask"][0].device, dtype=obs_inputs["attention_mask"].dtype),
                obs_inputs["attention_mask"][0]
            ], dim=0)
            attention_mask.append(attention_mask_)
            multi_modal_inputs.append({k: v for k, v in obs_inputs.items() if k not in ["input_ids", "attention_mask"]})

        # resp_padded = VF.pad_2d_list_to_length(resp_tokens, self.pad_id,
        #                                     max_length=self.cfg.response_length).to(device)

        seq_ids = VF.pad_2d_list_to_length(sequence_ids, self.pad_id,
                                        max_length=self.cfg.response_length+self.cfg.prompt_length).to(device)
        resp_padded = seq_ids[:, -self.cfg.response_length:]
        attn_mask = VF.pad_2d_list_to_length(attention_mask, 0,
                                        max_length=self.cfg.response_length+self.cfg.prompt_length).to(device)

        # sequence_loss_mask = VF.pad_2d_list_to_length(sequence_loss_mask, 0,
        #                                 max_length=self.cfg.response_length+self.cfg.prompt_length).to(device)
        response_loss_mask = VF.pad_2d_list_to_length(response_loss_mask, 0,
                                        max_length=self.cfg.response_length).to(device)
        # resp_mask = get_eos_mask_last(resp_padded, eos_token_id, dtype=input_ids.dtype)
        resp_mask = response_loss_mask
        position_ids = torch.stack(position_ids, dim=0)

        batch = TensorDict({
            "prompts"          : input_ids,
            "responses"        : resp_padded,
            "input_ids"        : seq_ids,
            "attention_mask"   : attn_mask,
            "response_mask"    : resp_mask,
            "response_loss_mask": response_loss_mask,
            "position_ids"     : position_ids,
        }, batch_size=batch_size)

        # the pixel_values tensors inside multi_modal_inputs were already updated
        return DataProto(batch=batch,
                         non_tensor_batch={"multi_modal_inputs":
                                           np.array(multi_modal_inputs)})
