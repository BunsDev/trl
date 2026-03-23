import pytest
import torch
import torch.nn.functional as F

from trl.experimental.async_grpo.chunk_lm_head import (
    _ChunkedLogProbFunction,
    _CompiledChunkedLogProbFunction,
)


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
def test_compiled_step_forward_matches_reference(temperature):
    """_CompiledChunkedLogProbFunction forward matches the reference."""
    torch.manual_seed(42)
    hidden = torch.randn(N, H)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (N,))

    logprobs, entropy = _CompiledChunkedLogProbFunction.apply(hidden, weight, labels, temperature, CHUNK_SIZE)
    logprobs_ref, entropy_ref = _reference_logprobs_and_entropy(hidden, weight, labels, temperature)

    torch.testing.assert_close(logprobs, logprobs_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(entropy, entropy_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_compiled_step_forward_matches_eager(temperature):
    """_CompiledChunkedLogProbFunction matches the eager version."""
    torch.manual_seed(42)
    hidden = torch.randn(N, H)
    weight = torch.randn(V, H)
    labels = torch.randint(0, V, (N,))

    logprobs_step, entropy_step = _CompiledChunkedLogProbFunction.apply(
        hidden, weight, labels, temperature, CHUNK_SIZE
    )
    logprobs_eager, entropy_eager = _ChunkedLogProbFunction.apply(hidden, weight, labels, temperature, CHUNK_SIZE)

    torch.testing.assert_close(logprobs_step, logprobs_eager, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(entropy_step, entropy_eager, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_compiled_step_backward(temperature):
    """_CompiledChunkedLogProbFunction backward matches the reference."""
    torch.manual_seed(42)
    hidden = torch.randn(N, H, requires_grad=True)
    weight = torch.randn(V, H, requires_grad=True)
    labels = torch.randint(0, V, (N,))

    logprobs, _ = _CompiledChunkedLogProbFunction.apply(hidden, weight, labels, temperature, CHUNK_SIZE)
    logprobs.sum().backward()
    grad_hidden_step = hidden.grad.clone()
    grad_weight_step = weight.grad.clone()

    hidden.grad = None
    weight.grad = None

    logprobs_ref, _ = _reference_logprobs_and_entropy(hidden, weight, labels, temperature)
    logprobs_ref.sum().backward()

    torch.testing.assert_close(grad_hidden_step, hidden.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_weight_step, weight.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_compiled_step_backward_matches_eager(temperature):
    """_CompiledChunkedLogProbFunction backward matches the eager backward."""
    torch.manual_seed(42)
    hidden_s = torch.randn(N, H, requires_grad=True)
    weight_s = torch.randn(V, H, requires_grad=True)
    labels = torch.randint(0, V, (N,))

    hidden_e = hidden_s.detach().clone().requires_grad_(True)
    weight_e = weight_s.detach().clone().requires_grad_(True)

    # Compiled step
    logprobs_s, _ = _CompiledChunkedLogProbFunction.apply(hidden_s, weight_s, labels, temperature, CHUNK_SIZE)
    logprobs_s.sum().backward()

    # Eager
    logprobs_e, _ = _ChunkedLogProbFunction.apply(hidden_e, weight_e, labels, temperature, CHUNK_SIZE)
    logprobs_e.sum().backward()

    torch.testing.assert_close(hidden_s.grad, hidden_e.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weight_s.grad, weight_e.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("temperature", [1.0, 0.7])
def test_compiled_step_backward_bfloat16(temperature):
    """_CompiledChunkedLogProbFunction backward works with bfloat16."""
    torch.manual_seed(42)
    hidden = torch.randn(N, H, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(V, H, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, V, (N,))

    logprobs, _ = _CompiledChunkedLogProbFunction.apply(hidden, weight, labels, temperature, CHUNK_SIZE)
    logprobs.sum().backward()
    grad_hidden_step = hidden.grad.clone()
    grad_weight_step = weight.grad.clone()

    hidden.grad = None
    weight.grad = None

    logprobs_ref, _ = _reference_logprobs_and_entropy(hidden, weight, labels, temperature)
    logprobs_ref.sum().backward()

    torch.testing.assert_close(grad_hidden_step, hidden.grad, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(grad_weight_step, weight.grad, atol=1e-2, rtol=1e-2)
