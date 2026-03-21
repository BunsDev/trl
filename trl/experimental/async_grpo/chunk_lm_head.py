# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

import types

import torch
import torch.nn as nn


class _ChunkedLogProbFunction(torch.autograd.Function):
    """Compute per-token log-probs and entropy without materializing [N, V] logits.

    Processes the lm_head in chunks and uses online logsumexp
    """

    @staticmethod
    def forward(
        ctx,
        last_hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        targets: torch.Tensor,  # [N]
        temperature: float,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = last_hidden.device
        N, _ = last_hidden.shape
        vocab, _ = weight.shape
        inv_t = 1.0 / temperature

        # Online logsumexp accumulators (fp32 for stability)
        max_old = torch.full((N,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        x_sum_exp = torch.zeros((N,), device=device, dtype=torch.float32)
        target_logit = torch.zeros((N,), device=device, dtype=torch.float32)

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]
            logits_chunk = (last_hidden @ w_chunk.t()).to(torch.float32) * inv_t  # [N, C]

            # Online logsumexp update
            chunk_max = logits_chunk.amax(dim=-1)  # [N]
            max_new = torch.maximum(max_old, chunk_max)
            rescale = torch.exp(max_old - max_new)
            chunk_exp = torch.exp(logits_chunk - max_new.unsqueeze(-1))  # [N, C]

            sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)
            x_sum_exp = x_sum_exp * rescale + (chunk_exp * logits_chunk).sum(dim=-1)
            max_old = max_new

            # Gather target logits for labels in this chunk
            in_chunk_cond = (targets >= start) & (targets < end)
            local_idx = torch.clamp(targets - start, 0, end - start - 1)
            target_logit = torch.where(
                in_chunk_cond,
                # take the new logit if target_idx is in this chunk bounds
                # otherwise, keep the old value
                logits_chunk[torch.arange(N, device=device), local_idx],
                target_logit,
            )

        log_z = max_old + torch.log(sum_exp)
        logprobs = target_logit - log_z
        entropy = log_z - x_sum_exp / sum_exp

        ctx.save_for_backward(last_hidden, weight, targets, log_z)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor):  # type: ignore
        hidden, weight, labels, log_z = ctx.saved_tensors
        temperature: float = ctx.temperature
        chunk_size: int = ctx.chunk_size
        inv_t = 1.0 / temperature

        N, _ = hidden.shape
        vocab = weight.shape[0]

        # NOTE(@aminediro): always acc in fp32 even if input is not
        grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
        grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)

        g = grad_logprobs.to(torch.float32)  # [N]

        for start in range(0, vocab, chunk_size):
            end = min(start + chunk_size, vocab)
            w_chunk = weight[start:end]  # [C, H]

            logits_chunk = (hidden @ w_chunk.t()).to(torch.float32) * inv_t  # [N, C]
            probs = torch.exp(logits_chunk - log_z.unsqueeze(-1))  # [N, C]

            # dL/d(logits) = g * (1_{label} - p)
            grad_logits = (-g).unsqueeze(-1) * probs  # [N, C]
            mask = (labels >= start) & (labels < end)
            if mask.any():
                idx = (labels[mask] - start).to(torch.long)
                grad_logits[mask, idx] += g[mask]

            # Chain through temperature: scaled_logit = logit * inv_t
            grad_logits = grad_logits * inv_t

            grad_hidden.add_(grad_logits @ w_chunk.float())
            grad_weight[start:end].add_(grad_logits.t() @ hidden.float())

        return grad_hidden.to(hidden.dtype), grad_weight.to(weight.dtype), None, None, None


def patch_chunked_lm_head(model: nn.Module, chunk_size: int, temperature: float) -> None:
    def _chunked_forward(
        self: nn.Module,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        assert labels is not None, "requires labels to be not None for logprob computation"

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache, **kwargs)
        hidden_states = outputs.last_hidden_state  # [B, S+1, H]

        # Shift: predict next token
        hidden_states = hidden_states[:, :-1, :]  # [B, S-1, H]
        labels = labels[:, 1:]  # [B, S-1]

        b, s, h = hidden_states.shape
        hidden_flat = hidden_states.reshape(b * s, h).contiguous()
        targets_flat = labels.reshape(b * s).contiguous()

        logprobs, entropy = _ChunkedLogProbFunction.apply(
            hidden_flat, self.lm_head.weight, targets_flat, temperature, chunk_size
        )

        return {
            "log_probs": logprobs.reshape(b, s),
            "entropy": entropy.reshape(b, s),
        }

    model.forward = types.MethodType(_chunked_forward, model)
