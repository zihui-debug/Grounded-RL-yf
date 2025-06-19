import os
import re
from contextlib import contextmanager
from typing import Any, List, Union

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
# import logging
import logging
logger = logging.getLogger(__name__)

from ....protocol import DataProto
from ....utils import torch_functional as VF
from ....utils.torch_dtypes import PrecisionType
from ..base import BaseRollout
from ..config import RolloutConfig

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

# def _repeat_interleave(value: Union[torch.Tensor, np.ndarray, List], repeats: int, create_deep_copy: bool = False) -> Union[torch.Tensor, np.ndarray, List[Any]]:
#     print(f"_repeat_interleave value")
#     if isinstance(value, torch.Tensor):
#         return value.repeat_interleave(repeats, dim=0)
#     elif create_deep_copy:
#         print(f"_repeat_interleave create_deep_copy with {len(value)} items")
#         # print(value)
#         # For lists, create a new list with each element repeated 'repeats' times
#         result = []
#         for i, item in enumerate(value):
#             # Use deep copy to ensure nested structures are properly copied
#             try:
#                 for _ in range(repeats):
#                     result.append(copy.deepcopy(item))
#                 print(f"Successfully copied item {i}")
#             except Exception as e:
#                 print(f"Error deep copying item {i}: {e}")
#                 # Fallback to shallow copy if deep copy fails
#                 for _ in range(repeats):
#                     result.append(copy.copy(item))
#         return result
#     elif isinstance(value, np.ndarray):
#         return np.repeat(value, repeats, axis=0)
#     else:
#         # For other types, create a list with the value repeated
#         return [copy.deepcopy(value) for _ in range(repeats)]

def _parse_coordinate(text: str):
    """
    Extract a coordinate from text of the form '(x, y)'.
    Returns (x, y) as a tuple of ints if found, otherwise None.
    """
    try:
        # text = text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        # print(f"_parse_coordinate text: {text}")
        text = text.strip()
        json_text = json.loads(text)
        return json_text["arguments"]["coordinate"]
    except Exception as e:
        # print(f"_parse_coordinate error: {e}")
        return None

def _get_point_crop(image: Image.Image, point: List[int], offset: int = 50) -> Image.Image:
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

    # Super-resolution to 200x200
    crop = crop.resize((offset*2*2, offset*2*2), Image.Resampling.LANCZOS)

    return crop

class vLLMRolloutMultiturn(BaseRollout):
    def __init__(
        self, 
        model_path: str, 
        config: RolloutConfig, 
        tokenizer: PreTrainedTokenizer,
        processor: ProcessorMixin
        ):
        """A vLLM rollout that supports multi-turn conversation in batch.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.processor = processor
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        if not config.enforce_eager and config.free_cache_engine:
            raise ValueError("CUDA graph should be disabled when `free_cache_engine` is True.")

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        vllm_init_kwargs = {}
        if config.limit_images > 0:
            vllm_init_kwargs = {"limit_mm_per_prompt": {"image": config.limit_images}}
            self.limit_images = config.limit_images
        else:
            raise ValueError("limit_images should be greater than >>1.")

        token_seqs_to_mask = ["<|im_start|>", "<|im_start|>user", "<|im_start|>assistant", "<|im_end|>"]
        self.token_seqs_to_mask_ids = [self.tokenizer.encode(token_seq, add_special_tokens=False) for token_seq in token_seqs_to_mask]

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=config.enforce_eager,
            max_model_len=config.prompt_length + config.response_length,
            max_num_batched_tokens=config.max_num_batched_tokens,
            enable_sleep_mode=True,
            distributed_executor_backend="external_launcher",
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            disable_log_stats=config.disable_log_stats,
            enable_chunked_prefill=config.enable_chunked_prefill,
            **vllm_init_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {"max_tokens": config.response_length, "detokenize": False}
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)
        self.n = self.sampling_params.n
        self.sampling_params.n = 1

        self.max_iterations = config.max_iterations

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        old_n = self.n
        
        if kwargs:
            # Handle 'n' separately since it's stored in self.n
            if 'n' in kwargs:
                old_sampling_params_args['n'] = old_n
                self.n = kwargs['n']
            
            # Handle other sampling parameters
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

            self.sampling_params.n = 1
            # logger.info(f"Sampling params: {self.sampling_params}.")
            # logger.info(f"n: {self.n}.")

        yield
        
        # roll back to previous sampling params
        if 'n' in old_sampling_params_args:
            self.n = old_sampling_params_args.pop('n')
            
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

        self.sampling_params.n = 1
    
    def _get_response_loss_mask(
        self, 
        response_ids: List | torch.Tensor, 
        eos_token_id: int | List[int],
        additonal_token_ids_to_mask: List[List[int]] = [],
        dtype: torch.dtype = torch.int64,
        device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Return a mask (same shape as `response_ids`) of dtype `dtype` which is `0`
        wherever `eos_token_id` occurs, or where any consecutive pattern in 
        `additonal_token_ids_to_mask` is found, else `1`.

        Args:
            response_ids: [batch_size, seq_len] token IDs (or possibly just [seq_len])
            eos_token_id: integer EOS token to mask, or list of integers
            additonal_token_ids_to_mask: list of lists, each list being a consecutive pattern
                                        that should be masked wherever found
            dtype: output tensor dtype

        Returns:
            mask of shape == response_ids.shape, dtype = `dtype`.
        """
        if isinstance(response_ids, list):
            response_ids = torch.tensor(response_ids)

        # Ensure 2D shape for uniform handling
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)  # [1, seq_len]

        bsz, seq_len = response_ids.shape
        mask = torch.ones_like(response_ids, dtype=dtype, device=device)

        # Mark wherever eos_token_id is found
        if isinstance(eos_token_id, list):
            for token_id in eos_token_id:
                mask[response_ids.eq(token_id)] = 0
        else:
            mask[response_ids.eq(eos_token_id)] = 0

        # For each row in the batch, mark occurrences of each consecutive pattern
        for b in range(bsz):
            row = response_ids[b]
            for pattern in additonal_token_ids_to_mask:
                pat_len = len(pattern)
                if pat_len == 0:
                    continue

                # Slide over this row and check for consecutive matches
                for start_idx in range(seq_len - pat_len + 1):
                    segment = row[start_idx : start_idx + pat_len]
                    # Compare this segment to the pattern
                    if torch.all(segment.eq(torch.tensor(pattern, device=row.device))):
                        mask[b, start_idx : start_idx + pat_len] = 0

        # If we artificially added a batch dimension, squeeze it out
        if response_ids.size(0) == 1 and bsz == 1:
            return mask.squeeze(0)
        return mask

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """
        Multi-turn, batched conversation:
          - Each item in `prompts` is one conversation session.
          - We keep generating in "turns" for only those sessions that haven't output <answer>...</answer>.
          - If <tool>...</tool> is found, we parse its coordinate and append <observation> <image> </observation>.
          - End when each conversation has an <answer>.
        """

        print("HERE1")

        # For safety, have a max iteration to avoid infinite loops
        iteration_count = 0

        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # # Save inputs for isolated testing
        # import os
        # import pickle
        # import time
        
        # save_dir = "/data/group_data/katefgroup/datasets/tmp"
        # os.makedirs(save_dir, exist_ok=True)
        # timestamp = int(time.time())
        
        # # Save tensor inputs
        # torch.save(input_ids, f"{save_dir}/input_ids_{timestamp}.pt")
        # torch.save(attention_mask, f"{save_dir}/attention_mask_{timestamp}.pt")
        # torch.save(position_ids, f"{save_dir}/position_ids_{timestamp}.pt")
        
        # # Save meta info and non-tensor batch
        # with open(f"{save_dir}/meta_info_{timestamp}.pkl", "wb") as f:
        #     pickle.dump(prompts.meta_info, f)
        
        # with open(f"{save_dir}/non_tensor_batch_{timestamp}.pkl", "wb") as f:
        #     # Create a copy to avoid modifying the original
        #     non_tensor_batch_copy = {k: v for k, v in prompts.non_tensor_batch.items()}
        #     # Handle non-serializable objects if needed
        #     pickle.dump(non_tensor_batch_copy, f)
            
        # # Save the entire prompts object if possible
        # try:
        #     with open(f"{save_dir}/prompts_{timestamp}.pkl", "wb") as f:
        #         pickle.dump(prompts, f)
        # except Exception as e:
        #     print(f"Could not save entire prompts object: {e}")

        # By default, we use the sampling_params we set up in the constructor.
        with self.update_sampling_params(**prompts.meta_info):

            print(f"Sampling params: {self.sampling_params}.")
            print(f"n: {self.n}.")

            print("HERE2")

            if self.n > 1:
                batch_size = batch_size * self.n
                input_ids = _repeat_interleave(input_ids, self.n)
                attention_mask = _repeat_interleave(attention_mask, self.n)
                position_ids = _repeat_interleave(position_ids, self.n)
                raw_prompt_ids = _repeat_interleave(
                        non_tensor_batch["raw_prompt_ids"], self.n
                        )
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    print("HERE2.1")
                    non_tensor_batch["multi_modal_inputs"] = _repeat_interleave(
                        non_tensor_batch["multi_modal_inputs"], self.n
                        )
                    input_images = []
                    for i, image_data in enumerate(non_tensor_batch["multi_modal_data"]):
                        for _ in range(self.n):
                            input_images.append([im.copy() for im in image_data["image"]])
                    print("HERE2.4")
            else:
                raw_prompt_ids = non_tensor_batch["raw_prompt_ids"]
                if "multi_modal_inputs" in non_tensor_batch.keys():
                    input_images = []
                    for i, image_data in enumerate(non_tensor_batch["multi_modal_data"]):
                        input_images.append([im.copy() for im in image_data["image"]])
                    print("HERE2.4")
                   

            print("HERE3")

            is_finished = [False]*batch_size

            if batch_size != len(raw_prompt_ids):
                raise RuntimeError("vllm sharding manager is not work properly.")

            # if "multi_modal_data" in non_tensor_batch:
            #     vllm_inputs = []
            #     for raw_prompt_ids, multi_modal_data in zip(
            #         non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            #     ):
            #         vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids), "multi_modal_data": multi_modal_data})
            # else:
            #     vllm_inputs = [
            #         {"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            #     ]

            if "multi_modal_data" in non_tensor_batch:
                vllm_inputs = []
                for raw_prompt_ids_, multi_modal_data in zip(
                    raw_prompt_ids, input_images
                ):
                    vllm_inputs.append({"prompt_token_ids": list(raw_prompt_ids_), "multi_modal_data": {"image": multi_modal_data}})
            else:
                vllm_inputs = [
                    {"prompt_token_ids": list(raw_prompt_ids_)} for raw_prompt_ids_ in raw_prompt_ids
                ]

            response_ids = [[] for _ in range(batch_size)]

            response_loss_mask = [torch.tensor([], dtype=attention_mask.dtype, device=attention_mask.device) for _ in range(batch_size)]

            print("HERE4")

        
            while not all(is_finished) and (iteration_count < self.max_iterations):
                iteration_count += 1

                # Build prompts for only the *active* items
                active_indices = [i for i, done in enumerate(is_finished) if not done]
                if not active_indices:
                    break

                # For each active item, we pass the entire conversation so far
                batch_prompts = []
                for i in active_indices:
                    batch_prompts.append(vllm_inputs[i])

                print("HERE5")

                # Generate in *one batch* for all active conversations
                generation_results: List[RequestOutput] = self.inference_engine.generate(
                    prompts=batch_prompts, 
                    sampling_params=self.sampling_params, 
                    use_tqdm=(self.rank == 0)
                )

                if self.rank == 0:
                    print("HERE6")

                # Now parse results: each item in generation_results corresponds to one prompt in `batch_prompts`.
                for idx, output in tqdm(enumerate(generation_results), 
                                       desc="Processing generation results", 
                                       leave=False,
                                       disable=self.rank != 0):
                    i = active_indices[idx]  # index in the original batch
                    if is_finished[i]:
                        continue  # skip if finished

                    if not output.outputs:
                        # No model output?
                        is_finished[i] = True
                        continue

                    tokens = list(output.outputs[0].token_ids)

                    # Append to the conversation
                    vllm_inputs[i]["prompt_token_ids"] += tokens

                    response_ids[i] += tokens

                    response_loss_mask_ = torch.ones(len(tokens), dtype=attention_mask.dtype, device=attention_mask.device)
                    response_loss_mask_ = response_loss_mask_ * self._get_response_loss_mask(tokens, eos_token_id, self.token_seqs_to_mask_ids, dtype=attention_mask.dtype, device=attention_mask.device)

                    response_loss_mask[i] = torch.cat([response_loss_mask[i], response_loss_mask_], dim=-1)

                    cur_text = self.tokenizer.decode(tokens)

                    # print("HERE7")

                    # Check for <answer>...</answer>
                    answer_match = re.search(r"<answer>(.*?)</answer>", cur_text, re.DOTALL)
                    if answer_match:
                        # Mark done
                        is_finished[i] = True

                    # Check for <tool_call>...</tool_call> 
                    tool_matches = re.findall(r"<tool_call>(.*?)</tool_call>", cur_text, re.DOTALL)
                    if tool_matches:
                        # For each found <tool>, parse coordinate, crop, and append <observation> <image> </observation>
                        for tool_text in tool_matches:
                            coords = _parse_coordinate(tool_text)
                            if coords and vllm_inputs[i]["multi_modal_data"] is not None:
                                crop = _get_point_crop(vllm_inputs[i]["multi_modal_data"]["image"][0], coords, offset=50)

                                if len(vllm_inputs[i]["multi_modal_data"]["image"]) == self.limit_images:
                                    is_finished[i] = True
                                    continue
                                elif len(vllm_inputs[i]["multi_modal_data"]["image"]) == self.limit_images - 1:
                                    obs_prompt = "\n<|im_start|>user\n<observation>\nHere is the crop of the image centered on the coordinate, with a red dot at the coordinate location:\n<|vision_start|><|image_pad|><|vision_end|>\n</observation><|im_end|>\n<|im_start|>assistant\n<answer>"
                                else:
                                    obs_prompt = "\n<|im_start|>user\n<observation>\nHere is the crop of the image centered on the coordinate, with a red dot at the coordinate location:\n<|vision_start|><|image_pad|><|vision_end|>\n</observation><|im_end|>\n<|im_start|>assistant\n"

                                vllm_inputs[i]["multi_modal_data"]["image"].append(crop)
                                
                                obs_inputs = self.processor(
                                    images=[crop],
                                    text=[obs_prompt],
                                    add_special_tokens=False,
                                    return_tensors="pt"
                                )

                                obs_input_ids = obs_inputs["input_ids"][0].cpu().tolist()
                                obs_raw_prompt_ids = self.tokenizer.encode(obs_prompt, add_special_tokens=False)

                                cur_text = self.tokenizer.decode(obs_input_ids)

                                vllm_inputs[i]["prompt_token_ids"].extend(obs_raw_prompt_ids)
                                # vllm_inputs[i]["prompt_token_ids"].extend(obs_input_ids)
                                # response_ids[i].extend(obs_input_ids)
                                response_ids[i].extend(obs_raw_prompt_ids)

                                response_loss_mask_ = torch.zeros(len(obs_input_ids), dtype=attention_mask.dtype, device=attention_mask.device)

                                if len(vllm_inputs[i]["multi_modal_data"]["image"]) >= self.limit_images:
                                    num_to_unmask = len(self.tokenizer.encode("<answer>", add_special_tokens=False))
                                    response_loss_mask_[-num_to_unmask:] = torch.ones(num_to_unmask, dtype=attention_mask.dtype, device=attention_mask.device)

                                response_loss_mask[i] = torch.cat([response_loss_mask[i], response_loss_mask_], dim=-1)

                                # print("HERE8")
                                # logger.info("HEREHERE")
                                # print("HERE8")

        if self.rank == 0:
            print("HERE7")
        logger.info("HERE7")

        response_ids = VF.pad_2d_list_to_length(
            response_ids, self.pad_token_id, max_length=self.config.response_length
        ).to(input_ids.device)

        if self.rank == 0:
            print("HERE8")
        logger.info("HERE8")

        response_loss_mask = VF.pad_2d_list_to_length(
            response_loss_mask, 0, max_length=self.config.response_length
        ).to(input_ids.device)

        if self.rank == 0:
            print("HERE9")
        logger.info("HERE9")

        # We need to re-process the new image crops with the processor.
        for i in tqdm(
            range(len(vllm_inputs)), 
            desc="Processing image crops", 
            leave=False,
            disable=self.rank != 0
        ):
            placeholder_text = " ".join(["<|vision_start|><|image_pad|><|vision_end|>"] * len(vllm_inputs[i]["multi_modal_data"]["image"]))
            obs_inputs = self.processor(
                images=vllm_inputs[i]["multi_modal_data"]["image"],
                text=[placeholder_text],
                add_special_tokens=False,
                return_tensors="pt"
            )
            non_tensor_batch["multi_modal_inputs"][i]["pixel_values"] = obs_inputs["pixel_values"]
            non_tensor_batch["multi_modal_inputs"][i]["image_grid_thw"] = obs_inputs["image_grid_thw"]
            print(f"vllm_inputs[i]['multi_modal_data']['image']: {vllm_inputs[i]['multi_modal_data']['image']}")
            print(f"Number of images: {len(vllm_inputs[i]['multi_modal_data']['image'])}")

        if self.rank == 0:
            print("HERE9a")

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        

        if self.rank == 0:
            print("HERE10")

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        attention_mask = (sequence_ids != self.pad_token_id).to(attention_mask.dtype)
        response_mask = get_eos_mask_last(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        # attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        if self.rank == 0:  
            print("HERE11")

        # tok_img = self.tokenizer.image_pad_token_id
        # n_tok  = (sequence_ids == tok_img).sum().item()
        # n_feat = sum(len(d["pixel_values"]) for d in non_tensor_batch["multi_modal_inputs"])
        # assert n_tok == n_feat, f"{n_tok=}  {n_feat=}"

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "response_loss_mask": response_loss_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        return DataProto(batch=batch, non_tensor_batch={"multi_modal_inputs": non_tensor_batch["multi_modal_inputs"]})
