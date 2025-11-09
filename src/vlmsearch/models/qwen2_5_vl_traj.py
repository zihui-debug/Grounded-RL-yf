import base64
from io import BytesIO
from typing import List, Optional, Tuple, Union

# import decord
import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)
# from lmms_eval import utils
# from lmms_eval.api.instance import Instance
# from lmms_eval.api.model import lmms
# from lmms_eval.api.registry import register_model
# from lmms_eval.models.model_utils.load_video import load_video_decord
import time

import sys
import os
from .prompt import observation_template, final_observation_template
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../scripts')))

# from attention_visual import *
try:
    from qwen_vl_utils import process_vision_info
    
except ImportError:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")

class StopOnTokens(StoppingCriteria):
    """
    A custom stopping criterion to stop generation when any one of the provided tokens appears.
    """
    def __init__(self, stop_ids: list):
        self.stop_ids = stop_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # For multi-beam, input_ids is (batch_size * num_beams, current_length).
        # Check the last token in each beam; if any is in stop_ids, stop.
        for sequence in input_ids:
            if sequence[-1] in self.stop_ids:
                return True
        return False

# @register_model("qwen2_vl")
# class Qwen2_5_VL_Traj(lmms):
class Qwen2_5_VL_Traj():
    """
    Wraps a Qwen2_5_VL model for single-step generation 
    of 'thoughts' or final answers. Loaded one instance per Accelerate process.
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map: Optional[str] = "auto",
        device: Optional[str] = "cuda",
        max_new_tokens: int = 64,
        temperature: float = 0.,
        top_p: float = None,
        top_k: int = None,
        num_beams: int = 1,
        use_cache: bool = True,
        use_flash_attention_2: Optional[bool] = True,
        max_pixels: int = 12845056,
        min_pixels: int = 3136,
        max_num_frames: int = 32,
        thought_token_begin: str = "<think>",
        thought_token_end: str = "</think>",
        final_token_begin: str = "<answer>",
        final_token_end: str = "</answer>",
        batch_size: int = 1,
        generate_attention_map: bool = False,
        multicrop: bool = False,
        **kwargs,
    ):
        """
        :param pretrained: Name (or path) of the Hugging Face model to load.
        :param device_map: Passed to from_pretrained for device placement. Usually "auto" with Accelerate.
        :param device: The device ("cuda" or "cpu").
        :param max_new_tokens: Default maximum tokens for a single generation step.
        :param temperature, top_p: Sampling parameters.
        :param use_cache: Whether to use the model's cache for faster inference.
        """
        super().__init__()

        self.max_num_frames = max_num_frames
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_cache = use_cache
        self.thought_token_begin = thought_token_begin
        self.thought_token_end = thought_token_end
        self.final_token_begin = final_token_begin
        self.final_token_end = final_token_end
        self.generate_attention_map = generate_attention_map
        self.use_custom_video_loader = False
        self.fps = None
        self.multicrop = multicrop
        # if self.fps and not self.use_custom_video_loader:
        #     raise ValueError("FPS is only applicable if use_custom_video_loader is True")
        self.max_image_size = None
        if self.max_image_size and not self.use_custom_video_loader:
            raise ValueError("max_image_size is only applicable if use_custom_video_loader is True")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        if use_flash_attention_2 and (self.generate_attention_map == False):
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained,
                torch_dtype=torch.bfloat16,
                device_map=self.device_map,
                attn_implementation="flash_attention_2",
            ).eval()
        else:
            if self.generate_attention_map:
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map, attn_implementation="eager").eval()
            else:
                self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(pretrained, torch_dtype="auto", device_map=self.device_map,).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)

        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _encode_image(self, image: Image.Image) -> str:
        """
        Converts a PIL Image to base64-encoded JPEG for Qwen2-VL's 'image' content.
        """
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG")
        base64_bytes = base64.b64encode(buffer.getvalue())
        return base64_bytes.decode("utf-8")

    # def generate_single_thought(
    #     self,
    #     system_prompt: str,
    #     previous_thoughts: List[Tuple[str, Optional[Image.Image]]],
    #     force_final: bool = False,
    #     no_sample: bool = False
    # ) -> str:
    #     """
    #     Generate a *single* piece of text from the model. 
    #     Incorporates:
    #       - A system prompt
    #       - A chain of previous thoughts (with optional images)
    #       - Possibly forces a <final> token if `force_final` is True

    #     Returns:
    #       The newly generated 'thought' or final answer.
    #     """

    #     # Build messages for Qwen2-VL
    #     messages = [
    #         {
    #             "role": "system",
    #             "content": "You are a helpful assistant."
    #         }
    #     ]

    #     # first thought is the question
    #     query = previous_thoughts[0]
    #     query_text = query[0]
    #     query_text = query_text.replace("<image>\n", "")
    #     query_text = query_text.replace("<image>", "")
    #     query_text = f"{system_prompt}\n\n{query_text}"
    #     query_image = query[1]
    #     previous_thoughts = previous_thoughts[1:]
    #     base64_string = self._encode_image(query_image)

    #     messages.append({"role": "user", "content": [{"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"}, {"type": "text", "text": query_text}]})

    #     # We'll add a single 'user' turn containing all previous_thoughts
    #     thought_text = ""
    #     if previous_thoughts:
    #         thought_text += "Previous thoughts:\n"
    #         # Accumulate the user content
    #         # user_content = [{"type": "text", "text": system_prompt}]
    #         for text_str, maybe_image in previous_thoughts:
    #             if maybe_image is not None:
    #                 raise NotImplementedError("Images are not supported yet")
    #             else:
    #                 text_str = text_str.strip()
    #                 text_str = '\n' + text_str
    #                 thought_text += text_str
    #     # If user wants to force a <final> token, we append it
    #     if force_final:
    #         # user_content.append({"type": "text", "text": "<final>"})
    #         # thought_text += f"\n{self.final_token_begin}\n"
    #         thought_text += f"\nPlease provide the final answer now in the format {self.final_token_begin} ... {self.final_token_end}. Do not provide any {self.thought_token_begin} or {self.thought_token_end} tags, only the final answer."


    #     if thought_text:
    #         messages.append({"role": "user", "content": [{"type": "text", "text": thought_text}]})

    #     # Qwen2-VL expects a list of entire conversation(s). 
    #     # We'll wrap our single conversation in a list:
    #     conversations = [messages]

    #     # Convert conversation to text that Qwen2-VL uses, while also extracting image/video
    #     texts = [self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True) 
    #              for conv in conversations]
    #     texts = [text for text in texts]

    #     # Let Qwen handle the images via process_vision_info
    #     image_inputs, video_inputs = process_vision_info(conversations)

    #     # (Optionally) reduce frames if it's a video
    #     if video_inputs is not None and len(video_inputs) > 0 and video_inputs[0] is not None:
    #         total_frames = video_inputs[0].shape[0]
    #         # Take up to self.max_num_frames frames
    #         indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
    #         # Ensure last frame is included
    #         if total_frames - 1 not in indices:
    #             indices = np.append(indices, total_frames - 1)
    #         video_inputs[0] = video_inputs[0][indices]

    #     # Now build the model inputs
    #     inputs = self.processor(
    #         text=texts,
    #         images=image_inputs,
    #         videos=video_inputs,
    #         padding=True,
    #         return_tensors="pt",
    #     )

    #     # Move inputs onto the correct device
    #     inputs = inputs.to(self.device)

    #     # Prepare generation settings
    #     pad_token_id = self.tokenizer.pad_token_id

    #     if no_sample:
    #         temperature = 0.0
    #     else:
    #         temperature = self.temperature

    #     eval_logger.debug(f"Generating with temperature {temperature}")
    #     eval_logger.debug(f"Generating with top_p {self.top_p}")
    #     eval_logger.debug(f"Generating with use_cache {self.use_cache}")
    #     eval_logger.debug(f"Generating with num_beams {self.num_beams}")
    #     eval_logger.debug(f"Generating with max_new_tokens {self.max_new_tokens}")
    #     eval_logger.debug(f"Generating with top_k {self.top_k}")
    #     # Actually run generate
    #     outputs = self.model.generate(
    #         **inputs,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         pad_token_id=pad_token_id,
    #         max_new_tokens=self.max_new_tokens,
    #         temperature=temperature,
    #         top_p=self.top_p,
    #         use_cache=self.use_cache,
    #         do_sample=True if temperature > 0 else False,
    #         # stopping_criteria=stopping_criteria,
    #         top_k=self.top_k,
    #     )

    #     # Separate out the newly generated tokens from the prompt
    #     generated_ids_trimmed = []
    #     for in_ids, out_ids in zip(inputs.input_ids, outputs):
    #         generated_ids_trimmed.append(out_ids[len(in_ids) :])

    #     # Decode them
    #     answers = self.processor.batch_decode(
    #         generated_ids_trimmed, 
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=False
    #     )

    #     eval_logger.debug(f"Generated answer: {answers[0]}")
            
    #     # Return the (first) newly generated text
    #     return answers[0] if answers else ""

    def generate_single_thought(
        self,
        system_prompt: str,
        previous_thoughts: List[Tuple[str, Optional[Image.Image]]],
        force_final: bool = False,
        no_sample: bool = False
    ) -> str:
        """
        Generate a *single* piece of text from the model. 
        Incorporates:
          - A system prompt
          - A chain of previous thoughts (with optional images)
          - Possibly forces a <final> token if `force_final` is True

        Returns:
          The newly generated 'thought' or final answer.
        """

        # Build messages for Qwen2-VL
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            }
        ]

        # We'll add a single 'user' turn containing all previous_thoughts
        if previous_thoughts:
            # Accumulate the user content
            for thought_idx, (text_str, maybe_image) in enumerate(previous_thoughts):
                if maybe_image is not None:
                    # Strip the <image> placeholder from text if needed
                    text_str = text_str.replace("<image>", "")
                    text_str = text_str.strip()
                    # Convert the image to base64
                    if isinstance(maybe_image, list):
                        base64_string = [self._encode_image(img) for img in maybe_image]
                    else:
                        base64_string = [self._encode_image(maybe_image)]
                    if thought_idx == 0:
                        messages.append(
                            {
                                "role": "user",
                                "content": [
                                    # {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}", "max_pixels": self.max_pixels, "min_pixels": self.min_pixels},
                                    {"type": "image", "image": maybe_image, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels},
                                    # {"type": "image", "image": f"data:image/jpeg;base64,{base64_string}"},
                                    {"type": "text", "text": text_str}
                                ]
                            }
                        )
                    else:
                        if self.multicrop:
                            # assume that >1 thoughts are the cropped images
                            messages.append({"role": "assistant", "content": [{"type": "text", "text": text_str}]})
                            # if len(base64_string) > 1:
                            #     text_str = f"<observation>\nHere is the crop of the image showing {len(base64_string)} regions:"
                            # else:
                            #     text_str = "<observation>\nHere is the crop of the image showing the region:"
                            text_str_before_image = f"""After the above Action {thought_idx-1}, here is the the zoom-in image (Observation {thought_idx}):\n"""
                            text_str_after_image = f""".\nContinue your reasoning process inside <think> and </think>. If needed, you can continue to zoom in on the original image or any of the observations, by outputting <tool_call> and </tool_call> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>."""
                            if not isinstance(maybe_image, list):
                                maybe_image = [maybe_image]
                            image_content = [
                                # {"type": "image", "image": f"data:image/jpeg;base64,{b64}"
                                # } for b64 in base64_string
                                {"type": "image", "image": img
                                } for img in maybe_image
                            ]
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        # {"type": "text", "text": text_str},
                                        {"type": "text", "text": text_str_before_image},
                                        *image_content,
                                        # {"type": "text", "text": "\n</observation>"},
                                        {"type": "text", "text": text_str_after_image},
                                    ]
                                }
                            )
                        else:
                            raise ValueError("Multicrop is not enabled")
                else:
                    # text_str = text_str.replace(f"{self.thought_token_begin}", "")
                    # text_str = text_str.replace(f"{self.thought_token_end}", "")
                    # text_str = text_str.strip()
                    # messages[-1]['content'][-1]['text'] += f"\n\nPrevious thought {thought_idx}: {text_str}"
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": text_str}]})
        
        # If user wants to force a <final> token, we append it
        add_answer_begin = False
        if force_final:
            if self.multicrop:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": f"<think> Based on all the regions I've examined, I can now provide my final answer. </think>\n<answer>"}]})
                continue_final_message = True
                add_generation_prompt = False
                add_answer_begin = True
                no_sample = True
            else:
                messages.append({"role": "user", "content": [{"type": "text", "text": f"\nYour turn! Max thoughts reached. You should now provide the final answer in the format {self.final_token_begin} ... {self.final_token_end}"}]})
                continue_final_message = False
                add_generation_prompt = True
        else:
            continue_final_message = False
            add_generation_prompt = True
        
        if no_sample:
            temperature = 0.
        else:
            temperature = self.temperature

        conversations = [messages]
        print(conversations)

        # Convert conversation to text that Qwen2-VL uses, while also extracting image/video
        texts = [self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=add_generation_prompt, continue_final_message=continue_final_message) 
                 for conv in conversations]
        texts = [text for text in texts]

        # Let Qwen handle the images via process_vision_info
        image_inputs, video_inputs = process_vision_info(conversations)

        # (Optionally) reduce frames if it's a video
        if video_inputs is not None and len(video_inputs) > 0 and video_inputs[0] is not None:
            total_frames = video_inputs[0].shape[0]
            # Take up to self.max_num_frames frames
            indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int)
            # Ensure last frame is included
            if total_frames - 1 not in indices:
                indices = np.append(indices, total_frames - 1)
            video_inputs[0] = video_inputs[0][indices]

        # Now build the model inputs
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs onto the correct device
        inputs = inputs.to(self.device)

        # Prepare generation settings
        pad_token_id = self.tokenizer.pad_token_id

        if no_sample:
            temperature = 0.0
        else:
            temperature = self.temperature

        eval_logger.debug(f"Generating with temperature {temperature}")
        eval_logger.debug(f"Generating with top_p {self.top_p}")
        eval_logger.debug(f"Generating with use_cache {self.use_cache}")
        eval_logger.debug(f"Generating with num_beams {self.num_beams}")
        eval_logger.debug(f"Generating with max_new_tokens {self.max_new_tokens}")
        eval_logger.debug(f"Generating with top_k {self.top_k}")
        # Actually run generate
        outputs = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=pad_token_id,
            max_new_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=self.top_p,
            use_cache=self.use_cache,
            do_sample=True if temperature > 0 else False,
            # stopping_criteria=stopping_criteria,
            top_k=self.top_k,
        )

        # Separate out the newly generated tokens from the prompt
        generated_ids_trimmed = []
        for in_ids, out_ids in zip(inputs.input_ids, outputs):
            generated_ids_trimmed.append(out_ids[len(in_ids) :])

        # Decode them
        answers = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        eval_logger.debug(f"Generated answer: {answers[0]}")
        if add_answer_begin and self.final_token_begin not in answers[0]:
            answers[0] = f"{self.final_token_begin}{answers[0]}"
            
        # Return the (first) newly generated text
        return answers[0] if answers else ""

    # def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
    #     raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    # def generate_until(self, requests: List[Instance]) -> List[str]:
    #     raise NotImplementedError("TODO: Implement generate_until")

    # def generate_until_multi_round(self, requests) -> List[str]:
    #     raise NotImplementedError("TODO: Implement multi-round generation")

