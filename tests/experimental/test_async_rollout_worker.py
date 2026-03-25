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
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset
from transformers import PreTrainedTokenizerBase

from trl.experimental.async_grpo.async_rollout_worker import (
    AsyncRolloutWorker,
    RolloutCompletion,
    RolloutGroup,
    RolloutSample,
    TaggedMessage,
    ToolCallRecord,
    TurnRecord,
    _build_completion,
)


def _make_turn(
    generation_ids=None,
    generation_logprobs=None,
    tool_calls=None,
    tool_response_ids=None,
    prompt_messages=None,
    assistant_content="Sure, 2+2 is 4.",
    tool_messages=None,
    include_context=True,
):
    """Build a TurnRecord with realistic defaults."""
    if generation_ids is None:
        generation_ids = [101, 102, 103, 104]
    if generation_logprobs is None:
        generation_logprobs = [-0.5, -0.3, -0.1, -0.2]
    if tool_calls is None:
        tool_calls = []
    if tool_response_ids is None:
        tool_response_ids = []

    messages = []
    if include_context and prompt_messages is not None:
        messages.extend(TaggedMessage(m, is_completion=False) for m in prompt_messages)
    messages.append(TaggedMessage({"role": "assistant", "content": assistant_content}, is_completion=True))
    if tool_messages:
        messages.extend(TaggedMessage(m, is_completion=True) for m in tool_messages)

    return TurnRecord(
        messages=messages,
        generation_ids=generation_ids,
        generation_logprobs=generation_logprobs,
        generation_duration=0.05,
        tool_calls=tool_calls,
        tool_response_ids=tool_response_ids,
        tool_execution_duration_s=0.01 * len(tool_calls),
    )


def _make_tool_call(name="calculator", failed=False):
    return ToolCallRecord(
        tool_call_id=f"call_{name}_1",
        name=name,
        arguments={"expr": "2+2"},
        result="4" if not failed else '{"error": "division by zero"}',
        failed=failed,
        duration=0.01,
    )


PROMPT = [{"role": "user", "content": "What is 2+2?"}]
PROMPT_IDS = [1, 2, 3, 4, 5]


@pytest.fixture
def rollout_worker():
    """Create a real AsyncRolloutWorker with vLLM/NCCL side effects patched out."""
    mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.apply_chat_template.return_value = list(PROMPT_IDS)

    def dummy_reward(completions, **kw):
        return [1.0] * len(completions)

    dataset = Dataset.from_dict({"prompt": [PROMPT]})

    with (
        patch("trl.experimental.async_grpo.async_rollout_worker.is_vllm_available", return_value=True),
        patch(
            "trl.experimental.async_grpo.async_rollout_worker.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch("trl.experimental.async_grpo.async_rollout_worker.add_response_schema", return_value=mock_tokenizer),
        patch("trl.experimental.async_grpo.async_rollout_worker.get_training_chat_template", return_value=None),
        patch.object(
            __import__(
                "trl.experimental.async_grpo.async_rollout_worker", fromlist=["AsyncRolloutWorker"]
            ).AsyncRolloutWorker,
            "_wait_for_server_ready_sync",
        ),
        patch.object(
            __import__(
                "trl.experimental.async_grpo.async_rollout_worker", fromlist=["AsyncRolloutWorker"]
            ).AsyncRolloutWorker,
            "_init_weight_transfer",
        ),
    ):
        worker = AsyncRolloutWorker(
            model_name="test-model",
            dataset=dataset,
            reward_funcs=[dummy_reward],
        )
    return worker


class TestRolloutCompletion:
    def test_single_turn_no_tools(self):
        gen_ids = [10, 11, 12]
        gen_logprobs = [-0.5, -0.3, -0.1]
        turn = _make_turn(
            generation_ids=gen_ids,
            generation_logprobs=gen_logprobs,
            prompt_messages=PROMPT,
        )
        comp = _build_completion([turn], truncated=False, total_duration=0.1)

        assert comp.get_completion_ids() == gen_ids
        assert comp.get_completion_logprobs() == gen_logprobs
        assert comp.get_tool_mask() == [1, 1, 1]
        # completion messages = only is_completion=True
        msgs = comp.get_completion_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"
        # trajectory = prompt context + assistant
        traj = comp.get_trajectory()
        assert len(traj) == 2  # 1 prompt + 1 assistant
        assert traj[0].is_completion is False
        assert traj[1].is_completion is True

    def test_single_turn_with_tools(self):
        gen_ids = [10, 11]
        gen_logprobs = [-0.5, -0.3]
        tool_resp_ids = [20, 21, 22]
        tool_msg = {"role": "tool", "name": "calculator", "content": "4"}

        turn = _make_turn(
            generation_ids=gen_ids,
            generation_logprobs=gen_logprobs,
            tool_calls=[_make_tool_call()],
            tool_response_ids=tool_resp_ids,
            prompt_messages=PROMPT,
            tool_messages=[tool_msg],
        )
        comp = _build_completion([turn], truncated=False, total_duration=0.1)

        # ids = gen + tool
        assert comp.get_completion_ids() == gen_ids + tool_resp_ids
        # logprobs = gen logprobs + 0.0 for each tool token
        assert comp.get_completion_logprobs() == gen_logprobs + [0.0] * len(tool_resp_ids)
        # mask = 1 for gen, 0 for tool
        assert comp.get_tool_mask() == [1, 1] + [0, 0, 0]
        # completion messages = assistant + tool
        msgs = comp.get_completion_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[1]["role"] == "tool"

    def test_multi_turn_no_duplicate_messages(self):
        """Turn 0 has prompt context; subsequent turns have only new messages. No duplicates."""
        tool_msg = {"role": "tool", "name": "calc", "content": "4"}

        turn0 = _make_turn(
            generation_ids=[10, 11],
            generation_logprobs=[-0.5, -0.3],
            tool_calls=[_make_tool_call()],
            tool_response_ids=[20, 21],
            prompt_messages=PROMPT,
            tool_messages=[tool_msg],
        )
        # Turn 1: no context (subsequent turn), just assistant
        turn1 = _make_turn(
            generation_ids=[30, 31, 32],
            generation_logprobs=[-0.2, -0.4, -0.1],
            include_context=False,
            assistant_content="The answer is 4.",
        )
        comp = _build_completion([turn0, turn1], truncated=False, total_duration=0.2)

        traj = comp.get_trajectory()
        # Expected: prompt(1) + assistant(1) + tool(1) from turn0 + assistant(1) from turn1 = 4
        assert len(traj) == 4
        # No duplicate messages — check by identity
        assert len({id(tm) for tm in traj}) == len(traj)
        # Verify order: prompt ctx, assistant, tool, assistant
        assert traj[0].is_completion is False  # prompt
        assert traj[1].message["role"] == "assistant"
        assert traj[2].message["role"] == "tool"
        assert traj[3].message["role"] == "assistant"

    def test_multi_turn_ids_concatenation(self):
        turn0 = _make_turn(
            generation_ids=[10, 11],
            generation_logprobs=[-0.5, -0.3],
            tool_calls=[_make_tool_call()],
            tool_response_ids=[20, 21],
            prompt_messages=PROMPT,
            tool_messages=[{"role": "tool", "name": "calc", "content": "4"}],
        )
        turn1 = _make_turn(
            generation_ids=[30, 31, 32],
            generation_logprobs=[-0.2, -0.4, -0.1],
            include_context=False,
        )
        comp = _build_completion([turn0, turn1], truncated=False, total_duration=0.2)

        ids = comp.get_completion_ids()
        logprobs = comp.get_completion_logprobs()
        mask = comp.get_tool_mask()

        assert len(ids) == len(logprobs) == len(mask)
        # turn0: gen(2) + tool(2) + turn1: gen(3) = 7
        assert ids == [10, 11, 20, 21, 30, 31, 32]
        assert logprobs == [-0.5, -0.3, 0.0, 0.0, -0.2, -0.4, -0.1]
        assert mask == [1, 1, 0, 0, 1, 1, 1]


class TestScoreGroup:
    @classmethod
    def setup_class(cls):
        """Initialize accelerate PartialState required by the logger in _score_group."""
        from accelerate import PartialState

        PartialState()

    def test_score_group_produces_valid_samples(self, rollout_worker):
        # 3 completions: single-turn, single-turn-with-tools, multi-turn
        comp1 = _build_completion(
            [_make_turn(generation_ids=[10, 11], generation_logprobs=[-0.5, -0.3], prompt_messages=PROMPT)],
            truncated=False,
            total_duration=0.1,
        )
        comp2 = _build_completion(
            [
                _make_turn(
                    generation_ids=[20, 21],
                    generation_logprobs=[-0.4, -0.2],
                    tool_calls=[_make_tool_call()],
                    tool_response_ids=[30, 31],
                    prompt_messages=PROMPT,
                    tool_messages=[{"role": "tool", "name": "calc", "content": "4"}],
                )
            ],
            truncated=False,
            total_duration=0.1,
        )
        comp3 = _build_completion(
            [
                _make_turn(
                    generation_ids=[40, 41],
                    generation_logprobs=[-0.1, -0.6],
                    tool_calls=[_make_tool_call()],
                    tool_response_ids=[50],
                    prompt_messages=PROMPT,
                    tool_messages=[{"role": "tool", "name": "calc", "content": "4"}],
                ),
                _make_turn(
                    generation_ids=[60, 61, 62],
                    generation_logprobs=[-0.3, -0.2, -0.5],
                    include_context=False,
                ),
            ],
            truncated=False,
            total_duration=0.2,
        )

        group = RolloutGroup(
            prompt=PROMPT,
            prompt_ids=PROMPT_IDS,
            reward_kwargs={},
            completions=[comp1, comp2, comp3],
            model_version=1,
            queued_at=time.monotonic(),
        )

        samples = asyncio.get_event_loop().run_until_complete(rollout_worker._score_group(group))
        assert len(samples) == 3

        for sample, comp in zip(samples, [comp1, comp2, comp3], strict=True):
            assert isinstance(sample, RolloutSample)
            assert len(sample.input_ids) == len(sample.completion_mask) == len(sample.old_log_probs)
            prompt_len = len(PROMPT_IDS)
            # Prompt region: mask=0, logprobs=0.0
            assert sample.completion_mask[:prompt_len] == [0] * prompt_len
            assert sample.old_log_probs[:prompt_len] == [0.0] * prompt_len
            # Completion region matches dataclass methods
            assert sample.completion_mask[prompt_len:] == comp.get_tool_mask()
            assert sample.old_log_probs[prompt_len:] == comp.get_completion_logprobs()
            assert sample.input_ids[prompt_len:] == comp.get_completion_ids()
            assert "reward" in sample.metrics
            assert "reward_std" in sample.metrics

    def test_score_group_tool_metrics(self, rollout_worker):
        comp_with_tools = _build_completion(
            [
                _make_turn(
                    generation_ids=[10, 11],
                    generation_logprobs=[-0.5, -0.3],
                    tool_calls=[_make_tool_call("calc", failed=False), _make_tool_call("search", failed=True)],
                    tool_response_ids=[20, 21],
                    prompt_messages=PROMPT,
                    tool_messages=[
                        {"role": "tool", "name": "calc", "content": "4"},
                        {"role": "tool", "name": "search", "content": '{"error": "not found"}'},
                    ],
                )
            ],
            truncated=False,
            total_duration=0.1,
        )
        comp_no_tools = _build_completion(
            [_make_turn(generation_ids=[30, 31], generation_logprobs=[-0.2, -0.4], prompt_messages=PROMPT)],
            truncated=False,
            total_duration=0.1,
        )

        group = RolloutGroup(
            prompt=PROMPT,
            prompt_ids=PROMPT_IDS,
            reward_kwargs={},
            completions=[comp_with_tools, comp_no_tools],
            model_version=1,
            queued_at=time.monotonic(),
        )

        samples = asyncio.get_event_loop().run_until_complete(rollout_worker._score_group(group))
        # First completion has 2 tool calls (1 failed)
        assert samples[0].metrics["tools/call_frequency"] == 2.0
        assert samples[0].metrics["tools/failure_frequency"] == 0.5
        # Second completion: still has tool metrics keys since total_calls > 0
        assert samples[1].metrics["tools/call_frequency"] == 0.0
        assert samples[1].metrics["tools/failure_frequency"] == 0.0


class TestGenerateOne:
    @patch("trl.experimental.async_grpo.async_rollout_worker.parse_response")
    def test_single_turn_no_tools(self, mock_parse, rollout_worker):
        mock_parse.return_value = {"role": "assistant", "content": "4"}
        rollout_worker._generate_one_turn = AsyncMock(return_value=([10, 11, 12], [-0.5, -0.3, -0.1]))

        result = asyncio.get_event_loop().run_until_complete(rollout_worker._generate_one(PROMPT, {}))

        assert isinstance(result, RolloutCompletion)
        assert len(result.turns) == 1
        assert result.truncated is False
        # Turn 0 should have prompt context
        traj = result.get_trajectory()
        assert any(not tm.is_completion for tm in traj)  # has context
        assert result.get_completion_ids() == [10, 11, 12]

    @patch("trl.experimental.async_grpo.async_rollout_worker.parse_response")
    def test_multi_turn_tool_calling(self, mock_parse, rollout_worker):
        # Turn 0: tool call, Turn 1: no tool call (stops)
        mock_parse.side_effect = [
            {
                "role": "assistant",
                "content": "calling calc",
                "tool_calls": [{"function": {"name": "calc"}, "id": "1"}],
            },
            {"role": "assistant", "content": "The answer is 4."},
        ]
        rollout_worker._generate_one_turn = AsyncMock(
            side_effect=[([10, 11], [-0.5, -0.3]), ([20, 21, 22], [-0.2, -0.4, -0.1])]
        )
        rollout_worker._execute_tool_calls = MagicMock(
            return_value=([_make_tool_call()], [{"role": "tool", "name": "calc", "content": "4"}])
        )
        rollout_worker._build_messages_suffix_ids = MagicMock(return_value=[30, 31])

        result = asyncio.get_event_loop().run_until_complete(rollout_worker._generate_one(PROMPT, {"calc": lambda: 4}))

        assert len(result.turns) == 2
        assert result.truncated is False
        # No duplicate messages in trajectory
        traj = result.get_trajectory()
        assert len({id(tm) for tm in traj}) == len(traj)

    @patch("trl.experimental.async_grpo.async_rollout_worker.parse_response")
    def test_truncated_at_max_turns(self, mock_parse, rollout_worker):
        rollout_worker.max_tool_calling_iterations = 2

        # Model always wants to call tools
        tool_response = {
            "role": "assistant",
            "content": "calling calc",
            "tool_calls": [{"function": {"name": "calc"}, "id": "1"}],
        }
        # 3 calls: turn 0 (iter=0, tool call), turn 1 (iter=1, tool call), turn 2 (iter=2, reached_max_turns)
        mock_parse.side_effect = [tool_response, tool_response, tool_response]
        rollout_worker._generate_one_turn = AsyncMock(return_value=([10, 11], [-0.5, -0.3]))
        rollout_worker._execute_tool_calls = MagicMock(
            return_value=([_make_tool_call()], [{"role": "tool", "name": "calc", "content": "4"}])
        )
        rollout_worker._build_messages_suffix_ids = MagicMock(return_value=[30, 31])

        result = asyncio.get_event_loop().run_until_complete(rollout_worker._generate_one(PROMPT, {"calc": lambda: 4}))

        assert result.truncated is True
        # 2 tool-calling turns + 1 truncated final turn = 3 turns
        assert len(result.turns) == 3

    @patch("trl.experimental.async_grpo.async_rollout_worker.parse_response")
    def test_trajectory_no_duplicates_end_to_end(self, mock_parse, rollout_worker):
        """Multi-turn generation should produce a trajectory with no duplicate message objects."""
        mock_parse.side_effect = [
            {"role": "assistant", "content": "step1", "tool_calls": [{"function": {"name": "t"}, "id": "1"}]},
            {"role": "assistant", "content": "step2", "tool_calls": [{"function": {"name": "t"}, "id": "2"}]},
            {"role": "assistant", "content": "done"},
        ]
        rollout_worker._generate_one_turn = AsyncMock(return_value=([10, 11], [-0.5, -0.3]))
        rollout_worker._execute_tool_calls = MagicMock(
            return_value=([_make_tool_call()], [{"role": "tool", "name": "t", "content": "ok"}])
        )
        rollout_worker._build_messages_suffix_ids = MagicMock(return_value=[30])

        result = asyncio.get_event_loop().run_until_complete(rollout_worker._generate_one(PROMPT, {"t": lambda: "ok"}))

        traj = result.get_trajectory()
        # All TaggedMessage instances should be unique objects
        assert len({id(tm) for tm in traj}) == len(traj)
        # Expected: prompt(1) + turn0(assistant+tool) + turn1(assistant+tool) + turn2(assistant) = 1+2+2+1 = 6
        assert len(traj) == 6
