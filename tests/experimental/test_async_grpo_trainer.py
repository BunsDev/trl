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

import queue

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample

from ..testing_utils import TrlTestCase


def dummy_reward_func(completions, **kwargs):
    return [float(hash(c[0]["content"]) % 100) / 100.0 for c in completions]


class _StubRolloutWorker:
    """Minimal rollout worker stub for testing the trainer in isolation.

    Populates its own queue in ``start()`` using real zen prompt-completion data processed through the provided
    tokenizer.
    """

    def __init__(self, tokenizer, dataset, num_generations: int = 4):
        self.rollout_buffer = queue.Queue()
        self._tokenizer = tokenizer
        self._dataset = dataset
        self._num_generations = num_generations

    def start(self):
        for row in self._dataset:
            prompt_ids = self._tokenizer.apply_chat_template(row["prompt"], tokenize=True, add_generation_prompt=True)
            completions = [
                [{"role": "assistant", "content": f"{row['completion'][0]['content']} {idx}"}]
                for idx in range(self._num_generations)
            ]
            completion_ids = self._tokenizer.apply_chat_template(
                completions, tokenize=True, add_generation_prompt=False
            )
            rewards = np.array(dummy_reward_func(completions))
            advantages = (rewards - rewards.mean()) / rewards.std()
            for idx in range(self._num_generations):
                input_ids = prompt_ids + completion_ids[idx]
                self.rollout_buffer.put(
                    RolloutSample(
                        prompt=row["prompt"],
                        completion=completions[idx],
                        input_ids=input_ids,
                        completion_mask=[0] * len(prompt_ids) + [1] * len(completion_ids[idx]),
                        old_log_probs=[0.0] * len(prompt_ids) + [-0.5] * len(completion_ids[idx]),
                        advantage=advantages[idx],
                        model_version=0,
                        metrics={"reward": rewards[idx], "reward_std": rewards.std()},
                    )
                )

    def stop(self):
        pass

    def pause(self):
        pass

    def resume(self):
        pass

    def send_weights(self, iterator):
        pass

    def update_model_version(self, version):
        pass


class TestAsyncGRPOTrainer(TrlTestCase):
    def test_training(self):
        model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        dataset = load_dataset("trl-internal-testing/zen", "conversational_prompt_completion", split="train")

        args = AsyncGRPOConfig(
            output_dir=self.tmp_dir,
            max_steps=2,
            per_device_train_batch_size=2,
            report_to="none",
        )

        trainer = AsyncGRPOTrainer(
            model=model_id,
            reward_funcs=dummy_reward_func,
            args=args,
            train_dataset=dataset,
            rollout_worker=_StubRolloutWorker(AutoTokenizer.from_pretrained(model_id), dataset),
        )

        previous_params = {n: p.clone() for n, p in trainer.model.named_parameters()}
        trainer.train()

        assert trainer.state.log_history[-1]["train_loss"] is not None
        assert any(not torch.equal(p, previous_params[n]) for n, p in trainer.model.named_parameters()), (
            "No model parameters changed after training."
        )
