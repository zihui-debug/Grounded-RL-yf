from openai import OpenAI
import os
import base64
from PIL import Image
from io import BytesIO
from typing import List, Tuple, Optional
import json
import logging

# Configure logging to suppress debug messages for this module
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger(__name__).setLevel(logging.WARNING)
TIMEOUT = 60

client = OpenAI(
    api_key="qwen",
    base_url=f"http://localhost:{os.getenv('PORT', '9001')}/v1",
    timeout=TIMEOUT,
)

class Qwen_VLLM_Traj():
    """
    Wraps a Qwen2_VL model for single-step generation 
    of 'thoughts' or final answers. Loaded one instance per Accelerate process.
    """

    def __init__(
        self,
        max_new_tokens: int = 64,
        temperature: float = 0.,
        top_p: float = None,
        examples: str = None,
        thought_token_begin: str = "<thought>",
        thought_token_end: str = "</thought>",
        final_token_begin: str = "<final>",
        final_token_end: str = "</final>",
        multicrop: bool = False,
        repetition_penalty: float = 0.0,
        max_pixels: int = 12845056,
        min_pixels: int = 3136,
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

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.thought_token_begin = thought_token_begin
        self.thought_token_end = thought_token_end
        self.final_token_begin = final_token_begin
        self.final_token_end = final_token_end
        self.multicrop = multicrop
        self.repetition_penalty = repetition_penalty
        if examples is not None:
            with open(examples, "r") as f:
                self.examples = json.load(f)
            for example in self.examples:
                example["image"] = Image.open(example["image"])
        else:
            self.examples = None

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

    def _encode_image(self, img: Image.Image) -> str:
        with BytesIO() as image_buffer:
            img.save(image_buffer, format="PNG")
            byte_data = image_buffer.getvalue()
            img_b64 = base64.b64encode(byte_data).decode("utf-8")
            img_b64 = "data:image/png;base64," + img_b64
        return img_b64

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

        if self.examples is not None:
            for example in self.examples:
                base64_string = self._encode_image(example["image"])
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {
                                "url": f"{base64_string}"}
                            },
                            {"role": "user", "content": [{"type": "text", "text": example["question"]}]}
                        ]
                    }
                    )
                if example["true_answer"] is not None:
                    thoughts = example["thoughts"]
                else:
                    thoughts = example["thoughts"][:-1]
                for thought in thoughts:
                    messages.append({"role": "user", "content": [{"type": "text", "text": f"{self.thought_token_begin} {thought} {self.thought_token_end}"}]})
                if example["true_answer"] is not None:
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": f"{self.final_token_begin} {example['true_answer']} {self.final_token_end}"}]})
                else:
                    messages.append({"role": "assistant", "content": [{"type": "text", "text": f"{self.thought_token_begin} {example['thoughts'][-1]} {self.thought_token_end}"}]})
        
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
                                    {"type": "image_url", "image_url": {
                                        "url": f"{base64_string[0]}"}
                                    },
                                    {"type": "text", "text": text_str}
                                ]
                            }
                        )
                    else:
                        if self.multicrop:
                            # assume that >1 thoughts are the cropped images
                            messages.append({"role": "assistant", "content": [{"type": "text", "text": text_str}]})
                            if len(base64_string) > 1:
                                text_str = f"<observation>\nHere is the crop of the image showing {len(base64_string)} regions:"
                            else:
                                text_str = "<observation>\nHere is the crop of the image showing the region:"
                            image_content = [
                                {"type": "image_url", "image_url": {
                                    "url": f"{b64}"}
                                } for b64 in base64_string
                            ]
                            messages.append(
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": text_str},
                                        *image_content,
                                        {"type": "text", "text": "\n</observation>"},
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

        completion = client.chat.completions.create(
            model="qwen_vllm",
            messages=messages,
            max_tokens=self.max_new_tokens,
            temperature=temperature,
            top_p=self.top_p,
            extra_body={
                "continue_final_message": continue_final_message, 
                "add_generation_prompt": add_generation_prompt,
                "repetition_penalty": self.repetition_penalty
                }
        )
        answer = completion.choices[0].message.content

        if add_answer_begin and self.final_token_begin not in answer:
            answer = f"{self.final_token_begin}{answer}"

        return answer

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood is not implemented for Qwen2_VL")

    def generate_until(self, requests):
        raise NotImplementedError("TODO: Implement generate_until")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("TODO: Implement multi-round generation")

        
