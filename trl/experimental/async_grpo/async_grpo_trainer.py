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


import asyncio
import queue
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import requests
import torch

from accelerate.logging import get_logger
from datasets import Dataset
from torch.distributed._tensor import DTensor
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback
from transformers.data.data_collator import DataCollatorMixin
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
from vllm.utils.network_utils import get_ip, get_open_port

from trl.trainer.utils import pad, selective_log_softmax

from .async_grpo_config import AsyncGRPOConfig
from .async_rollout_worker import AsyncRolloutWorker


logger = get_logger(__name__)

# A reward function is a callable that returns a list of floats (the rewards). The callable receives prompts,
# completions, and additional arguments from the trainer (refer to the trainer's source for details). To ensure forward
# compatibility, it should accept **kwargs.
RewardFunc = Callable[..., list[float]]


class _SupportsReset(Protocol):
    def reset(self, **kwargs) -> str | None: ...


EnvironmentFactory = Callable[[], _SupportsReset]


class StepIntervalCallback(TrainerCallback):
    def __init__(self, fn, every_n_steps: int):
        self.fn = fn
        self.every_n_steps = every_n_steps

    def on_step_end(self, _args, state, _control, **_kwargs):
        if state.global_step % self.every_n_steps == 0:
            self.fn()


class RolloutQueueDataset(IterableDataset):
    def __init__(self, rollout_queue, model_version_fn, max_staleness=3, timeout=120.0):
        self.queue = rollout_queue
        self.model_version_fn = model_version_fn
        self.max_staleness = max_staleness
        self.timeout = timeout

    def __iter__(self):
        while True:
            try:
                sample = self.queue.get(timeout=self.timeout)
            except queue.Empty:
                logger.warning(f"Rollout queue empty for {self.timeout}s, stopping epoch")
                return  # StopIteration ends epoch

            staleness = self.model_version_fn() - sample.model_version
            if staleness > self.max_staleness:
                logger.debug(f"Dropping stale sample (staleness={staleness}, max={self.max_staleness})")
                continue  # drop stale, pull next

            yield {
                "input_ids": sample.input_ids,
                "completion_mask": sample.completion_mask,
                "old_log_probs": sample.old_log_probs,
                "advantage": sample.advantage,
                "metrics": sample.metrics,
            }


class _EmptyIterableDataset(IterableDataset):
    """Placeholder for non-rank-0 processes. Never actually iterated."""

    def __iter__(self):
        return iter([])


@dataclass
class DataCollatorForRollout(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = [torch.tensor(example["input_ids"], dtype=torch.long) for example in examples]
        attention_mask = [torch.ones(len(ids), dtype=torch.long) for ids in input_ids]
        completion_mask = [torch.tensor(example["completion_mask"], dtype=torch.float32) for example in examples]
        old_log_probs = [torch.tensor(example["old_log_probs"], dtype=torch.float32) for example in examples]
        advantages = torch.tensor([example["advantage"] for example in examples], dtype=torch.float32)

        input_ids = pad(input_ids, padding_value=self.pad_token_id)
        attention_mask = pad(attention_mask, padding_value=0)
        completion_mask = pad(completion_mask, padding_value=0)
        old_log_probs = pad(old_log_probs, padding_value=0)

        # Total valid completion tokens across all samples in the full batch.
        # Repeated per sample so that DataLoaderDispatcher (dispatch_batches=True) slices correctly on dim=0
        global_n_tokens = completion_mask.sum()
        global_n_tokens_repeated = torch.full((len(examples),), global_n_tokens.item(), dtype=torch.float32)

        # Convert per-sample metrics dicts to a dict of 1D tensors so that Accelerate's
        # recursive broadcast (dispatch_batches=True) can handle them — it traverses nested
        # dicts of tensors but chokes on plain Python floats.
        metrics_list = [example["metrics"] for example in examples]
        metrics = (
            {
                key: torch.tensor([m.get(key, 0.0) for m in metrics_list], dtype=torch.float32)
                for key in metrics_list[0]
            }
            if metrics_list and metrics_list[0]
            else {}
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "old_log_probs": old_log_probs,
            "advantages": advantages,
            "global_n_tokens": global_n_tokens_repeated,
            "metrics": metrics,
        }


class AsyncGRPOTrainer(Trainer):
    def __init__(
        self,
        model: str,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset,
        args: AsyncGRPOConfig | None = None,
        tools: list[Callable] | None = None,
        environment_factory: EnvironmentFactory | None = None,
        **kwargs,
    ):
        self.args = args or AsyncGRPOConfig()

        # Training arguments
        self.epsilon_low = self.args.epsilon
        self.epsilon_high = self.args.epsilon_high
        self.temperature = self.args.temperature

        # Model
        model_name = model
        model = AutoModelForCausalLM.from_pretrained(model, device_map=None, dtype=torch.bfloat16)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        # Initialize the Trainer
        super().__init__(
            model=model,
            args=self.args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            compute_loss_func="non-None value to disable scaling",
            **kwargs,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._last_compute_loss_time = None
        self._total_train_tokens = 0
        self._train_tokens_start_time = None
        self.model_version = 0
        # Create worker thread on rank 0
        if self.accelerator.is_main_process:
            if not self.train_dataset:
                raise ValueError("train_dataset is required for AsyncGRPOTrainer")

            self.rollout_queue = queue.Queue(maxsize=self.args.queue_maxsize)
            self.rollout_worker = AsyncRolloutWorker(
                model_name=model_name,
                dataset=train_dataset,
                rollout_buffer=self.rollout_queue,
                reward_funcs=reward_funcs,
                tools=tools,
                environment_factory=environment_factory,
                num_generations=self.args.num_generations,
                max_inflight_tasks=self.args.max_inflight_tasks,
                vllm_server_url=self.args.vllm_server_base_url,
                max_tokens=self.args.max_completion_length,
                temperature=self.args.temperature,
                request_timeout=self.args.request_timeout,
                chat_template_kwargs=self.args.chat_template_kwargs,
                log_completions=self.args.log_completions,
                num_completions_to_print=self.args.num_completions_to_print,
            )

        else:
            self.rollout_queue = None
            self.rollout_worker = None

        self.vllm_server_url = self.args.vllm_server_base_url
        self.model_update_group = None
        self._worker_loop = None
        self._worker_stop_event = None

        if self.accelerator.is_main_process:
            self._worker_thread = threading.Thread(target=self._run_worker, daemon=True)
            self._worker_thread.start()

        # Add callbacks
        self.add_callback(StepIntervalCallback(self._sync_weight, self.args.weight_sync_steps))

    def _run_worker(self):
        """Runs the AsyncRolloutWorker inside an asyncio event loop in a background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._worker_loop = loop
        self._worker_stop_event = asyncio.Event()
        try:
            loop.run_until_complete(self.rollout_worker.run(stop_event=self._worker_stop_event))
        except Exception as e:
            logger.exception(f"Worker thread failed: {e}")
            raise
        finally:
            loop.close()

    def get_train_dataloader(self) -> DataLoader:
        if self.accelerator.is_main_process:
            dataset = RolloutQueueDataset(
                rollout_queue=self.rollout_queue,
                model_version_fn=lambda: self.model_version,
                max_staleness=self.args.max_staleness,
                timeout=self.args.vllm_server_timeout,
            )
        else:
            dataset = _EmptyIterableDataset()

        return self.accelerator.prepare(
            DataLoader(
                dataset,
                batch_size=self.args.per_device_train_batch_size * self.accelerator.num_processes,
                collate_fn=DataCollatorForRollout(self.processing_class.pad_token_id),
                num_workers=0,  # MUST be 0
            )
            # NOTE(@aminediro):
            # dispatch_batches = True for DataLoader whose underlying dataset is an IterableDataset
            # dataloader prepared by the Accelerator is only iterated through on the main process a
        )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs (usually, "input_ids"
        # and "attention_mask"). In AsyncGRPOTrainer, we need additional columns ("completion_mask", "old_log_probs",
        # "advantages", "global_n_tokens") to compute the loss, hence the override.
        if self._signature_columns is None:
            self._signature_columns = [
                "input_ids",
                "attention_mask",
                "completion_mask",
                "old_log_probs",
                "advantages",
                "global_n_tokens",
                "metrics",
            ]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        old_log_probs = inputs["old_log_probs"]
        advantages = inputs["advantages"]

        # The collator pads to the global batch max length (across all ranks). After
        # DataLoaderDispatcher slices and sends rows to each rank, the local slice is
        # still padded to that global max. Truncate to the longest real sequence in
        # this rank's slice so we don't run the forward pass over pure-padding columns.
        local_max_len = attention_mask.sum(dim=1).max()
        input_ids = input_ids[:, :local_max_len]
        attention_mask = attention_mask[:, :local_max_len]
        completion_mask = completion_mask[:, :local_max_len]
        old_log_probs = old_log_probs[:, :local_max_len]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:]
        logits.div_(self.temperature)
        log_probs = selective_log_softmax(logits, targets)
        completion_mask = completion_mask[:, 1:]
        old_log_probs = old_log_probs[:, 1:]
        advantages = advantages.unsqueeze(1)
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        clipped = torch.clamp(ratio, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss = -torch.min(ratio * advantages, clipped * advantages)

        # DDP/FSDP averages gradients across ranks (world_size).
        # To get correct per-token normalization we scale by 1/tokens_per_rank
        # = world_size / global_n_tokens, so after DDP averaging the effective
        loss = (per_token_loss * completion_mask).sum()
        global_n_tokens = inputs["global_n_tokens"][0]
        world_size = self.accelerator.num_processes
        tokens_per_rank = (global_n_tokens / world_size).clamp(min=1.0)
        loss = loss / tokens_per_rank.to(torch.float32)
        # For DAPO, we would scale like this instead:
        # loss = loss / max(per_token_loss.size(0), 1)
        loss = loss / self.args.gradient_accumulation_steps

        with torch.no_grad():
            valid_mask = completion_mask > 0
            local_count = valid_mask.sum().float()

            local_ratio_sum = (
                ratio[valid_mask].sum() if valid_mask.any() else torch.zeros((), device=completion_mask.device)
            )
            # Approx KL: http://joschu.net/blog/kl-approx.html
            local_kl_sum = (
                ((ratio[valid_mask] - 1) - log_ratio[valid_mask]).sum()
                if valid_mask.any()
                else torch.zeros((), device=completion_mask.device)
            )

            probs = torch.softmax(logits, dim=-1)
            log_p = torch.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_p, dim=-1)
            local_entropy_sum = (
                entropy[valid_mask].sum() if valid_mask.any() else torch.zeros((), device=completion_mask.device)
            )

            clipped = (ratio < 1 - self.epsilon_low) | (ratio > 1 + self.epsilon_high)
            local_clip_sum = (
                clipped[valid_mask].float().sum()
                if valid_mask.any()
                else torch.zeros((), device=completion_mask.device)
            )

            # Batch all-reduce: [ratio_sum, kl_sum, entropy_sum, clip_sum, count]
            stats = torch.stack([local_ratio_sum, local_kl_sum, local_entropy_sum, local_clip_sum, local_count])
            stats = self.accelerator.reduce(stats, reduction="sum")
            global_ratio_sum, global_kl_sum, global_entropy_sum, global_clip_sum, global_count = stats.unbind(0)
            self._metrics["train"]["ratio"].append((global_ratio_sum / global_count).item())
            self._metrics["train"]["kl"].append((global_kl_sum / global_count).item())
            self._metrics["train"]["entropy"].append((global_entropy_sum / global_count).item())
            self._metrics["train"]["clip_ratio"].append((global_clip_sum / global_count).item())

            # Logging metrics from the rollout worker (reward, reward_std, etc.).
            # inputs["metrics"] is a dict of 1D tensors keyed by metric name.
            sample_metrics = inputs["metrics"]  # dict[str, Tensor(shape=[B_local])]
            keys = list(sample_metrics.keys())
            device = completion_mask.device
            n_samples = torch.tensor(completion_mask.shape[0], dtype=torch.float32, device=device)
            if keys:
                local_sums = torch.stack([sample_metrics[k].to(device).sum() for k in keys])
                stats = torch.cat([local_sums, n_samples.unsqueeze(0)])
                stats = self.accelerator.reduce(stats, reduction="sum")
                global_sums, global_n_samples = stats[:-1], stats[-1]
                for k, global_sum in zip(keys, global_sums, strict=True):
                    self._metrics["train"][k].append((global_sum / global_n_samples).item())

            completion_length = completion_mask.sum(dim=1).float()
            length_stats = torch.stack([completion_length.sum(), n_samples])
            length_stats = self.accelerator.reduce(length_stats, reduction="sum")
            self._metrics["train"]["completions/mean_length"].append((length_stats[0] / length_stats[1]).item())

            # Training-side tok/s: completion tokens consumed per second
            now = time.time()
            if self._train_tokens_start_time is None:
                self._train_tokens_start_time = now
            self._total_train_tokens += global_n_tokens.item()
            train_elapsed = now - self._train_tokens_start_time
            if train_elapsed > 0:
                self._metrics["train"]["train_tok/s"].append(self._total_train_tokens / train_elapsed)

        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}  # average the metrics
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}
        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()

    def _init_weight_transfer(self) -> None:
        if not self.accelerator.is_main_process:
            return
        response = requests.get(f"{self.vllm_server_url}/get_world_size")
        inference_world_size = response.json()["world_size"]
        world_size = inference_world_size + 1
        master_address = get_ip()
        master_port = get_open_port()

        init_info = {
            "master_address": master_address,
            "master_port": master_port,
            "rank_offset": 1,
            "world_size": world_size,
        }
        t_init = threading.Thread(
            target=requests.post,
            args=(f"{self.vllm_server_url}/init_weight_transfer_engine",),
            kwargs={"json": {"init_info": init_info}, "timeout": 120},
        )
        t_init.start()
        self.model_update_group = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
        t_init.join()

    def _sync_weight(self):
        t0 = time.time()
        logger.info("Weight sync: pausing vLLM...")
        if self.accelerator.is_main_process:
            requests.post(f"{self.vllm_server_url}/pause", params={"mode": "wait"})
        t_pause = time.time()
        logger.info(f"Weight sync: pause took {t_pause - t0:.1f}s, waiting for all ranks...")

        self.accelerator.wait_for_everyone()
        t_barrier = time.time()

        logger.info(f"Weight sync: transferring weights... (barrier took {t_barrier - t_pause:.1f}s)")
        if self.accelerator.is_main_process and self.model_update_group:
            # Build metadata without materializing tensors.
            # DTensor.shape returns the global shape; no all-gather needed here.
            names, dtype_names, shapes = [], [], []
            for name, param in self.model.named_parameters():
                names.append(name)
                dtype_names.append(str(param.dtype).split(".")[-1])
                shapes.append(list(param.shape))

            update_payload = {
                "update_info": {
                    "names": names,
                    "dtype_names": dtype_names,
                    "shapes": shapes,
                    "packed": True,
                    "is_checkpoint_format": True,
                }
            }
            t_update = threading.Thread(
                target=requests.post,
                args=(f"{self.vllm_server_url}/update_weights",),
                kwargs={"json": update_payload, "timeout": 1800},
            )
            t_update.start()

        def _streaming_iter():
            # Iterate parameters one at a time. For FSDP2 (DTensor), full_tensor()
            # all-gathers just this parameter across FSDP ranks, then frees it once
            # the generator advances — avoiding materializing the full model in memory.
            for name, param in self.model.named_parameters():
                full = param.full_tensor() if isinstance(param, DTensor) else param.detach()
                yield name, full

        if self.accelerator.is_main_process and self.model_update_group:
            NCCLWeightTransferEngine.trainer_send_weights(
                iterator=_streaming_iter(),
                trainer_args=NCCLTrainerSendWeightsArgs(group=self.model_update_group, packed=True),
            )
            t_update.join()
        else:
            # Non-rank-0 processes must still participate in full_tensor() collectives.
            for _ in _streaming_iter():
                pass
        t_transfer = time.time()

        self.accelerator.wait_for_everyone()

        logger.info(f"Weight sync: resuming vLLM... (transfer took {t_transfer - t_barrier:.1f}s)")
        if self.accelerator.is_main_process:
            requests.post(f"{self.vllm_server_url}/resume")
            self.model_version += 1
            if self.rollout_worker:
                self.rollout_worker.update_model_version(self.model_version)
        logger.info(f"Weight sync: done. Total {time.time() - t0:.1f}s")

    def _inner_training_loop(self, *args, **kwargs):
        if self.accelerator.is_main_process:
            self._init_weight_transfer()
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            if self.accelerator.is_main_process and self._worker_stop_event:
                # Schedule stop_event.set() in the worker's event loop
                logger.info("Stopping worker thread...")
                if self._worker_loop and self._worker_loop.is_running():
                    try:
                        self._worker_loop.call_soon_threadsafe(self._worker_stop_event.set)
                    except RuntimeError:
                        pass
