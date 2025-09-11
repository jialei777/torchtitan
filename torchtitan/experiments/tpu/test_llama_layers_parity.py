import copy

import pytest
import torch
from torch import nn

from torchtitan.models.llama3.model.model import (
    Attention,
    FeedForward,
    precompute_freqs_cis,
    TransformerBlock,
    TransformerModelArgs,
)

DEVICE = torch.device("cuda")


def is_device_available(device):
    """
    Checks if the given torch device is available.
    For CUDA devices, it checks for CUDA availability.
    For other devices, it currently assumes they are available.
    This function can be extended to support custom device availability checks.
    """
    if device.type == "cuda":
        return torch.cuda.is_available()
    # For other device types, we assume they are available for now.
    # This can be extended for custom devices.
    return True


def _get_model_args(n_layers=2, vocab_size=32, max_seq_len=16):
    return TransformerModelArgs(
        dim=64,
        n_layers=n_layers,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        multiple_of=16,
    )


@pytest.mark.skipif(not is_device_available(DEVICE), reason=f"Device {DEVICE} not available")
def test_llama_embedding_cpu_gpu_parity():
    """
    Tests the CPU vs. GPU parity of the Embedding layer.
    """
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8

    # CPU setup
    embedding_cpu = nn.Embedding(args.vocab_size, args.dim).cpu()
    nn.init.normal_(embedding_cpu.weight)
    tokens_cpu = torch.randint(
        0, args.vocab_size, (batch, seq_len), device="cpu", requires_grad=False
    )

    # GPU setup
    embedding_gpu = copy.deepcopy(embedding_cpu).to(DEVICE)
    tokens_gpu = tokens_cpu.detach().clone().to(DEVICE).requires_grad_(False)

    # Forward pass
    out_cpu = embedding_cpu(tokens_cpu)
    out_gpu = embedding_gpu(tokens_gpu)
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5, rtol=1e-5)

    # Backward pass
    # Add requires_grad=True for output to calculate gradients
    out_cpu.requires_grad_(True)
    out_gpu.requires_grad_(True)
    loss_cpu = out_cpu.sum()
    loss_gpu = out_gpu.sum()
    loss_cpu.backward()
    loss_gpu.backward()

    assert torch.allclose(
        embedding_cpu.weight.grad,
        embedding_gpu.weight.grad.cpu(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.skipif(not is_device_available(DEVICE), reason=f"Device {DEVICE} not available")
def test_llama_attention_cpu_gpu_parity():
    """
    Tests the CPU vs. GPU parity of the Attention layer.
    """
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8

    # CPU setup
    attention_cpu = Attention(args).cpu()
    attention_cpu.init_weights(init_std=0.02)
    x_cpu = torch.randn(batch, seq_len, args.dim, device="cpu", requires_grad=True)
    freqs_cis_cpu = precompute_freqs_cis(
        args.dim // args.n_heads, seq_len, theta=args.rope_theta
    )

    # GPU setup
    attention_gpu = copy.deepcopy(attention_cpu).to(DEVICE)
    x_gpu = x_cpu.detach().clone().to(DEVICE).requires_grad_(True)
    freqs_cis_gpu = freqs_cis_cpu.to(DEVICE)

    # Forward pass
    out_cpu = attention_cpu(x_cpu, freqs_cis_cpu)
    out_gpu = attention_gpu(x_gpu, freqs_cis_gpu)
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5, rtol=1e-5)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_gpu = out_gpu.sum()
    loss_cpu.backward()
    loss_gpu.backward()

    assert torch.allclose(x_cpu.grad, x_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)
    for p_cpu, p_gpu in zip(attention_cpu.parameters(), attention_gpu.parameters()):
        assert torch.allclose(p_cpu.grad, p_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not is_device_available(DEVICE), reason=f"Device {DEVICE} not available")
def test_llama_feedforward_cpu_gpu_parity():
    """
    Tests the CPU vs. GPU parity of the FeedForward layer.
    """
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8

    # CPU setup
    feedforward_cpu = FeedForward(
        dim=args.dim,
        hidden_dim=4 * args.dim,
        multiple_of=args.multiple_of,
        ffn_dim_multiplier=args.ffn_dim_multiplier,
    ).cpu()
    feedforward_cpu.init_weights(init_std=0.02)
    x_cpu = torch.randn(batch, seq_len, args.dim, device="cpu", requires_grad=True)

    # GPU setup
    feedforward_gpu = copy.deepcopy(feedforward_cpu).to(DEVICE)
    x_gpu = x_cpu.detach().clone().to(DEVICE).requires_grad_(True)

    # Forward pass
    out_cpu = feedforward_cpu(x_cpu)
    out_gpu = feedforward_gpu(x_gpu)
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5, rtol=1e-5)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_gpu = out_gpu.sum()
    loss_cpu.backward()
    loss_gpu.backward()

    assert torch.allclose(x_cpu.grad, x_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)
    for p_cpu, p_gpu in zip(feedforward_cpu.parameters(), feedforward_gpu.parameters()):
        assert torch.allclose(p_cpu.grad, p_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not is_device_available(DEVICE), reason=f"Device {DEVICE} not available")
def test_llama_transformer_block_cpu_gpu_parity():
    """
    Tests the CPU vs. GPU parity of the TransformerBlock layer.
    """
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8
    layer_id = 0

    # CPU setup
    transformer_block_cpu = TransformerBlock(layer_id, args).cpu()
    transformer_block_cpu.init_weights()
    x_cpu = torch.randn(batch, seq_len, args.dim, device="cpu", requires_grad=True)
    freqs_cis_cpu = precompute_freqs_cis(
        args.dim // args.n_heads, seq_len, theta=args.rope_theta
    )

    # GPU setup
    transformer_block_gpu = copy.deepcopy(transformer_block_cpu).to(DEVICE)
    x_gpu = x_cpu.detach().clone().to(DEVICE).requires_grad_(True)
    freqs_cis_gpu = freqs_cis_cpu.to(DEVICE)

    # Forward pass
    out_cpu = transformer_block_cpu(x_cpu, freqs_cis_cpu)
    out_gpu = transformer_block_gpu(x_gpu, freqs_cis_gpu)
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5, rtol=1e-5)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_gpu = out_gpu.sum()
    loss_cpu.backward()
    loss_gpu.backward()

    assert torch.allclose(x_cpu.grad, x_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)
    for p_cpu, p_gpu in zip(
        transformer_block_cpu.parameters(), transformer_block_gpu.parameters()
    ):
        assert torch.allclose(p_cpu.grad, p_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)
