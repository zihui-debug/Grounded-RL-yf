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
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Dict, Optional

import torch
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.modeling_flash_attention_utils import index_first_axis, pad_input, unpad_input

from ...protocol import DataProto
from ...trainer import core_algos
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: Dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                )

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(
                input_ids.unsqueeze(-1), attention_mask
            )  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_sequence_parallel_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)
        
        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        micro_batches = data.select(select_keys, non_tensor_select_keys).split(
            self.config.micro_batch_size_per_device_for_experience
        )
        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=2)

        for micro_batch in micro_batches:
            # breakpoint()
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)
        return log_probs
    
    def visualize_loss_mask(self, seq_ids, sequence_loss_mask, idx, tokenizer=None, output_html=False, advantages=None):
        """
        Visualize which parts of the text are masked out by sequence_loss_mask.
        
        Args:
            seq_ids: Tensor of token IDs [batch_size, seq_len]
            sequence_loss_mask: Tensor of same shape as seq_ids with 1s for tokens included in loss, 0s for masked tokens
            idx: Index of the sequence in the batch to visualize
            tokenizer: Tokenizer to use for decoding (if None, uses self.tokenizer)
            output_html: If True, returns HTML for display in notebooks, otherwise prints to console
        
        Returns:
            If output_html is True, returns HTML string for display in notebooks
            Otherwise, prints the visualization to console
        """
        if tokenizer is None:
            tokenizer = self.tokenizer
        
        # Get the sequence and mask for the specified index
        seq = seq_ids[idx].cpu().tolist()
        mask = sequence_loss_mask[idx].cpu().tolist()
        
        # Remove padding tokens
        pad_token_id = tokenizer.pad_token_id
        valid_indices = [i for i, token_id in enumerate(seq) if token_id != pad_token_id]
        if not valid_indices:
            return "Empty sequence (all padding tokens)"
        
        seq = [seq[i] for i in valid_indices]
        mask = [mask[i] for i in valid_indices]
        
        # Decode each token individually to see token boundaries
        tokens = []
        for token_id in seq:
            # Use decode with skip_special_tokens=False to see special tokens
            token_text = tokenizer.decode([token_id], skip_special_tokens=False)
            tokens.append(token_text)
        
        # Create visualization
        if output_html:
            html_output = ["<div style='font-family: monospace; white-space: pre-wrap; line-height: 1.5;'>"]
            html_output.append("<h3>Token-by-Token Visualization</h3>")
            html_output.append("<div style='margin-bottom: 20px;'>")
            
            for i, (token, m) in enumerate(zip(tokens, mask)):
                # Escape HTML special characters
                token_display = token.replace("<", "&lt;").replace(">", "&gt;")
                
                if m == 1:  # Included in loss
                    color = "green"
                    bg_color = "#e6ffe6"  # Light green background
                    tooltip = "Included in loss calculation"
                else:  # Masked out
                    color = "red"
                    bg_color = "#ffe6e6"  # Light red background
                    tooltip = "Masked out (not included in loss)"
                
                html_output.append(
                    f"<span title='{tooltip} (Token ID: {seq[i]})' "
                    f"style='color: {color}; background-color: {bg_color}; "
                    f"border: 1px solid {color}; border-radius: 3px; "
                    f"padding: 2px; margin: 1px; display: inline-block;'>"
                    f"{token_display}</span>"
                )
            
            html_output.append("</div>")
            
            # Add full text visualization
            html_output.append("<h3>Full Text Visualization</h3>")
            html_output.append("<div>")
            
            # Decode the full sequence
            full_text = tokenizer.decode(seq, skip_special_tokens=False)
            
            # Create character-level mask by expanding token-level mask
            char_mask = []
            current_pos = 0
            
            for token_text, m in zip(tokens, mask):
                token_len = len(token_text)
                char_mask.extend([m] * token_len)
                current_pos += token_len
            
            # Ensure char_mask is at least as long as full_text
            if len(char_mask) < len(full_text):
                char_mask.extend([0] * (len(full_text) - len(char_mask)))
            elif len(char_mask) > len(full_text):
                char_mask = char_mask[:len(full_text)]
            
            # Create HTML with character-level highlighting
            current_mask = None
            current_span = []
            
            for char, m in zip(full_text, char_mask):
                if current_mask != m:
                    if current_span:
                        color = "green" if current_mask == 1 else "red"
                        bg_color = "#e6ffe6" if current_mask == 1 else "#ffe6e6"
                        tooltip = "Included in loss" if current_mask == 1 else "Masked out"
                        span_text = "".join(current_span).replace("<", "&lt;").replace(">", "&gt;")
                        html_output.append(
                            f"<span title='{tooltip}' style='color: {color}; "
                            f"background-color: {bg_color};'>{span_text}</span>"
                        )
                    current_span = []
                    current_mask = m
                
                current_span.append(char)
            
            # Add the last span
            if current_span:
                color = "green" if current_mask == 1 else "red"
                bg_color = "#e6ffe6" if current_mask == 1 else "#ffe6e6"
                tooltip = "Included in loss" if current_mask == 1 else "Masked out"
                span_text = "".join(current_span).replace("<", "&lt;").replace(">", "&gt;")
                html_output.append(
                    f"<span title='{tooltip}' style='color: {color}; "
                    f"background-color: {bg_color};'>{span_text}</span>"
                )
            
            html_output.append("</div>")
            
            # Add statistics
            masked_count = mask.count(0)
            included_count = mask.count(1)
            total_count = len(mask)
            
            html_output.append("<h3>Statistics</h3>")
            html_output.append("<ul>")
            html_output.append(f"<li>Total tokens: {total_count}</li>")
            html_output.append(f"<li>Tokens included in loss (green): {included_count} ({included_count/total_count:.1%})</li>")
            html_output.append(f"<li>Tokens masked out (red): {masked_count} ({masked_count/total_count:.1%})</li>")
            html_output.append(f"<li>Mean advantage: {advantages[idx].mean().cpu().item()}</li>")
            html_output.append("</ul>")
            
            html_output.append("</div>")
            return "".join(html_output)
        
        else:
            # Console output
            print("\n" + "="*80)
            print(f"VISUALIZATION OF LOSS MASK FOR SEQUENCE {idx}")
            print("="*80)
            
            print("\nTOKEN-BY-TOKEN VIEW:")
            print("-"*80)
            
            for i, (token, m) in enumerate(zip(tokens, mask)):
                status = "INCLUDED" if m == 1 else "MASKED OUT"
                print(f"Token {i:4d} | ID: {seq[i]:5d} | Mask: {m} | {status:10s} | '{token}'")
            
            print("\nFULL TEXT VIEW:")
            print("-"*80)
            
            # Group consecutive tokens with the same mask value
            current_mask = None
            current_text = []
            
            for token, m in zip(tokens, mask):
                if current_mask != m:
                    if current_text:
                        status = "INCLUDED IN LOSS" if current_mask == 1 else "MASKED OUT"
                        print(f"[{status}] {''.join(current_text)}")
                    current_text = []
                    current_mask = m
                
                current_text.append(token)
            
            # Print the last group
            if current_text:
                status = "INCLUDED IN LOSS" if current_mask == 1 else "MASKED OUT"
                print(f"[{status}] {''.join(current_text)}")
            
            # Statistics
            masked_count = mask.count(0)
            included_count = mask.count(1)
            total_count = len(mask)
            
            print("\nSTATISTICS:")
            print("-"*80)
            print(f"Total tokens: {total_count}")
            print(f"Tokens included in loss: {included_count} ({included_count/total_count:.1%})")
            print(f"Tokens masked out: {masked_count} ({masked_count/total_count:.1%})")
            print("="*80 + "\n")

    def update_policy(self, data: DataProto, tokenizer=None) -> Dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
        if self.config.use_kl_loss and not self.config.disable_kl:
            select_keys.append("ref_log_probs")
        if self.config.multiturn:
            select_keys.append("response_loss_mask")
            select_keys.append("response_mask")

        if "multi_modal_inputs" in data.non_tensor_batch.keys():
            non_tensor_select_keys = ["multi_modal_inputs"]
        else:
            non_tensor_select_keys = []

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=2)

            for mini_batch in mini_batches:
                gradient_accumulation = (
                    self.config.global_batch_size_per_device // self.config.micro_batch_size_per_device_for_update
                )
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)
                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=3)

                all_unmasked_ratios = []

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    responses = model_inputs["responses"]
                    response_length = responses.size(1)
                    attention_mask = model_inputs["attention_mask"]
                    if self.config.multiturn:
                        response_mask = model_inputs["response_loss_mask"][:, -response_length:]
                    else:
                        response_mask = attention_mask[:, -response_length:]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    if self.config.mask_negative_advantage:
                        # set response_mask to 0 for negative advantages
                        response_mask[advantages < 0] = 0

                    # Collect stats for adaptive learning rate
                    if self.config.adaptive_lr:
                        unmasked_tokens_per_sample = response_mask.sum(dim=1)  # (bsz,)
                        samples_with_signal = (unmasked_tokens_per_sample > 0).float()  # (bsz,)
                        unmasked_ratio = samples_with_signal.mean().item()  # ratio between 0 and 1
                        all_unmasked_ratios.append(unmasked_ratio)

                    ###########################################################
                    # Visualize loss mask
                    ###########################################################
                    # html_text = self.visualize_loss_mask(model_inputs["input_ids"][:, -response_length:], response_mask, 0, tokenizer=tokenizer, output_html=True, advantages=advantages)
                    # import random
                    # import string
                    # random_number_letters = ''.join(random.choices(string.ascii_letters, k=10))
                    # with open(f"/home/gsarch/repo/VLM-search/src/trainer/rl/html/loss_mask_{random_number_letters}.html", "w") as f:
                    #     f.write(html_text)
                    ###########################################################

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
                    entropy_loss = -VF.masked_mean(log_probs, response_mask)  # estimator of entropy loss

                    pg_loss, pg_clipfrac_higher, pg_clipfrac_lower, ppo_kl = core_algos.compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                    )
                    if "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = core_algos.compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = VF.masked_mean(kld, response_mask)
                        pg_loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef

                    loss = pg_loss / gradient_accumulation
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_clipfrac_higher.detach().item(),
                        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        "actor/entropy_loss": entropy_loss.detach().item(),
                        "actor/ppo_kl": ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, batch_metrics)

                if self.config.adaptive_lr and all_unmasked_ratios:
                    # Average unmasked ratio across all micro-batches
                    avg_unmasked_ratio = sum(all_unmasked_ratios) / len(all_unmasked_ratios)
                    
                    # Skip optimization step if no tokens contribute to the loss
                    if avg_unmasked_ratio == 0:
                        # Zero out gradients without taking step
                        self.actor_optimizer.zero_grad()
                        # Track zero ratio in metrics
                        append_to_dict(metrics, {"actor/unmasked_ratio": 0.0, "actor/adapted_lr": 0.0, "actor/grad_norm": 0.0})
                    else:
                        # Scale learning rate by unmasked ratio with minimum bound
                        min_lr_ratio = getattr(self.config, "min_lr_ratio", 0.1)
                        current_lr = self.actor_optimizer.param_groups[0]['lr']
                        adapted_lr = current_lr * max(avg_unmasked_ratio, min_lr_ratio)
                        
                        # Save original learning rate and set the adapted one
                        original_lr = current_lr
                        self.actor_optimizer.param_groups[0]['lr'] = adapted_lr
                        
                        # Track these values in metrics
                        append_to_dict(metrics, {"actor/unmasked_ratio": avg_unmasked_ratio, "actor/adapted_lr": adapted_lr})
                        
                        # Take optimizer step with adapted learning rate
                        grad_norm = self._optimizer_step()
                        
                        # Restore original learning rate
                        self.actor_optimizer.param_groups[0]['lr'] = original_lr
                else:
                    grad_norm = self._optimizer_step()
                    append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
