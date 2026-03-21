import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from trl.experimental.async_grpo.chunk_lm_head import _ChunkedLogProbFunction, patch_chunked_lm_head


N, H, V = 64, 32, 128
CHUNK_SIZE = 32


def _reference_logprobs_and_entropy(hidden, weight, labels, temperature):
    logits = (hidden @ weight.t()).to(torch.float32) / temperature  # [N, V]
    log_p = F.log_softmax(logits, dim=-1)
    logprobs = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    p = torch.softmax(logits, dim=-1)
    entropy = -(p * log_p).sum(dim=-1)
    return logprobs, entropy


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_forward(temperature):
    torch.manual_seed(42)
    hidden = torch.randn(N, H)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (N,))

    logprobs_chunked, entropy_chunked = _ChunkedLogProbFunction.apply(hidden, weight, labels, temperature, CHUNK_SIZE)
    logprobs_ref, entropy_ref = _reference_logprobs_and_entropy(hidden, weight, labels, temperature)

    torch.testing.assert_close(logprobs_chunked, logprobs_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(entropy_chunked, entropy_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_backward(temperature):
    torch.manual_seed(42)
    hidden = torch.randn(N, H, requires_grad=True)
    weight = torch.randn(V, H, requires_grad=True)
    labels = torch.randint(0, V, (N,))

    # Chunked backward
    logprobs_chunked, _ = _ChunkedLogProbFunction.apply(hidden, weight, labels, temperature, CHUNK_SIZE)
    logprobs_chunked.sum().backward()
    grad_hidden_chunked = hidden.grad.clone()
    grad_weight_chunked = weight.grad.clone()

    hidden.grad = None
    weight.grad = None

    # Reference backward
    logprobs_ref, _ = _reference_logprobs_and_entropy(hidden, weight, labels, temperature)
    logprobs_ref.sum().backward()

    torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_backward_bfloat16(temperature):
    torch.manual_seed(42)
    hidden = torch.randn(N, H, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, V, (N,))

    # Chunked backward
    logprobs_chunked, _ = _ChunkedLogProbFunction.apply(hidden, weight, labels, temperature, CHUNK_SIZE)
    logprobs_chunked.sum().backward()
    grad_hidden_chunked = hidden.grad.clone()
    grad_weight_chunked = weight.grad.clone()

    hidden.grad = None
    weight.grad = None

    # Reference backward
    logprobs_ref, _ = _reference_logprobs_and_entropy(hidden, weight, labels, temperature)
    logprobs_ref.sum().backward()

    torch.testing.assert_close(grad_hidden_chunked, hidden.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_weight_chunked, weight.grad, atol=1e-2, rtol=1e-2)


B, S = 4, 16  # batch size, sequence length (including prompt + completion)


class _FakeTransformerModel(nn.Module):
    """Minimal stand-in for a transformer body: returns random hidden states of the right shape."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Fixed hidden states so results are reproducible across calls
        self._hidden = None

    def forward(self, input_ids, attention_mask=None, use_cache=False, **kwargs):
        b, s = input_ids.shape
        if self._hidden is None or self._hidden.shape[:2] != (b, s):
            torch.manual_seed(123)
            self._hidden = torch.randn(b, s, self.hidden_size, requires_grad=True)
        return type("Out", (), {"last_hidden_state": self._hidden})()


class _FakeCausalLM(nn.Module):
    """Minimal CausalLM with .model and .lm_head, enough for patch_chunked_lm_head."""

    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.model = _FakeTransformerModel(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        raise NotImplementedError("should be monkey-patched")


def _build_model_and_inputs(temperature=1.0):
    torch.manual_seed(42)
    model = _FakeCausalLM(H, V)
    patch_chunked_lm_head(model, CHUNK_SIZE, temperature)

    input_ids = torch.randint(0, V, (B, S))
    attention_mask = torch.ones(B, S, dtype=torch.long)
    # First half of each sequence is prompt (0), second half is completion (1)
    completion_mask = torch.zeros(B, S, dtype=torch.float32)
    completion_mask[:, S // 2 :] = 1.0
    return model, input_ids, attention_mask, completion_mask


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_chunked_forward_with_completion_mask(temperature):
    """Masked forward matches unmasked forward at completion positions and is zero at prompt positions."""
    model, input_ids, attention_mask, completion_mask = _build_model_and_inputs(temperature)

    # Run WITHOUT completion_mask (baseline — computes all positions)
    out_full = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

    # Reset hidden state cache so both runs use the same hidden states
    model.model._hidden = None

    # Run WITH completion_mask
    out_masked = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, completion_mask=completion_mask
    )

    # shifted completion_mask (matching the shift in _chunked_forward)
    shifted_mask = completion_mask[:, 1:].bool()

    # At completion positions, values should match
    torch.testing.assert_close(
        out_masked["log_probs"][shifted_mask],
        out_full["log_probs"][shifted_mask],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(
        out_masked["entropy"][shifted_mask],
        out_full["entropy"][shifted_mask],
        atol=1e-5,
        rtol=1e-5,
    )

    # At prompt positions, values should be zero
    prompt_mask = ~shifted_mask
    assert (out_masked["log_probs"][prompt_mask] == 0).all()
    assert (out_masked["entropy"][prompt_mask] == 0).all()


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_chunked_forward_completion_mask_backward(temperature):
    model, input_ids, attention_mask, completion_mask = _build_model_and_inputs(temperature)

    # Full forward + backward (mask applied after, as the trainer does)
    out_full = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    shifted_mask = completion_mask[:, 1:]
    loss_full = (out_full["log_probs"] * shifted_mask).sum()
    loss_full.backward()
    grad_weight_full = model.lm_head.weight.grad.clone()

    model.lm_head.weight.grad = None
    model.model._hidden = None

    # Masked forward + backward
    out_masked = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=input_ids, completion_mask=completion_mask
    )
    loss_masked = (out_masked["log_probs"] * shifted_mask).sum()
    loss_masked.backward()
    grad_weight_masked = model.lm_head.weight.grad.clone()

    torch.testing.assert_close(grad_weight_masked, grad_weight_full, atol=1e-5, rtol=1e-5)
