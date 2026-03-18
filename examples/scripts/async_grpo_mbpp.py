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

"""
CUDA_VISIBLE_DEVICES=2,3,4,5 VLLM_SERVER_DEV_MODE=1 vllm serve Qwen/Qwen3-4B \
    --data-parallel-size 4 \
    --weight-transfer-config '{"backend":"nccl"}' \
    --max-num-seqs 64 \
    --max-model-len 9216

CUDA_VISIBLE_DEVICES=0,1 accelerate launch  --config_file examples/accelerate_configs/fsdp2.yaml  examples/scripts/async_grpo_mbpp.py

!/! NOTE: depends on transformers > 5.0.0
"""

import logging
import os
import subprocess
import sys
import tempfile

from datasets import load_dataset

from trl.experimental.async_grpo import AsyncGRPOConfig, AsyncGRPOTrainer


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class MBPPEnvironment:
    """
    A synchronous environment class designed for `AsyncGRPOTrainer`.
    Each environment instance handles tracking test cases and exposes the `execute_python_code` tool.
    """

    def __init__(self):
        self.test_list = []
        self.done = False

    def reset(self, test_list: list[str], **kwargs):
        """
        Resets the environment with the test suite for the new problem.
        `**kwargs` ignores additional columns sent from the dataset map.
        """
        self.test_list = test_list
        self.done = False

    def execute_python_code(self, code: str) -> str:
        """Execute python code to test against the hidden test cases. Provide the complete python code.

        Args:
            code: The complete python code to execute.

        Returns:
            program stdout string
        """
        full_code = code + "\n\n" + "\n".join(self.test_list)

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(full_code)
            temp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if result.returncode == 0:
                self.done = True
                return "Tests passed."
            else:
                # Return the last 2000 characters of the stderr to fit within context length
                feedback = result.stderr[-2000:]
                return f"Execution failed with error:\n{feedback}\nPlease fix the code and try again."
        except subprocess.TimeoutExpired:
            return "Execution timeout."
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def is_done(self) -> bool:
        return self.done


def tests_passed_reward(completions, **kwargs) -> list[float]:
    """
    Reward function that checks the model's chat history for the last tool execution
    and returns 1.0 if the tests passed, 0.0 otherwise.
    """
    rewards = []
    for completion in completions:
        passed = False
        # Interrogate the completion, looping backwards to find the last tool interaction result
        for msg in reversed(completion):
            if msg["role"] == "tool" and "Tests passed." in msg.get("content", ""):
                passed = True
                break
        rewards.append(1.0 if passed else 0.0)
    return rewards


def format_sample(sample):
    """
    Format the MBPP dataset row into a prompt using OpenAI chat formatting,
    and persist `test_list` so `reset()` can inject it.
    """
    prompt_text = sample.get("text", "")
    content = (
        f"You are an expert Python programmer.\n\n"
        f"{prompt_text}\n\n"
        f"Please write a python code to solve the problem and use the execute_python_code tool to test it."
    )
    prompt = [{"role": "user", "content": content}]

    return {"prompt": prompt, "test_list": sample.get("test_list", [])}


def main() -> None:
    os.environ["WANDB_PROJECT"] = "async_grpo_trl_mbpp"
    # 1. Load dataset
    dataset = load_dataset("google-research-datasets/mbpp", split="train+test")
    dataset = dataset.map(format_sample, remove_columns=dataset.column_names)

    # 2. Config setup
    config = AsyncGRPOConfig(
        output_dir="./results",
        per_device_train_batch_size=1,
        max_completion_length=8192,
        max_seq_length=8192,
        max_tool_calling_iterations=5,
        max_steps=100,
        max_staleness=8,
        # Logging
        log_completions=True,
        num_completions_to_print=2,
        report_to="wandb",
        logging_steps=1,
        # trackio
        project="async_grpo_trl_mbpp",
        trackio_space_id=None,
    )

    # 3. Trainer initialization
    trainer = AsyncGRPOTrainer(
        model="Qwen/Qwen3-4B",
        args=config,
        train_dataset=dataset,
        reward_funcs=[tests_passed_reward],
        environment_factory=MBPPEnvironment,
    )

    # 4. Train
    trainer.train()


if __name__ == "__main__":
    main()
