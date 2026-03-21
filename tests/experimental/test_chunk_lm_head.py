import pytest
import torch
import torch.nn.functional as F

from trl.experimental.async_grpo.chunk_lm_head import _ChunkedLogProbFunction


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
