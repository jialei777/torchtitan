import copy
import os

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

# The default device is 'cpu', if 'TORCH_DEVICE' is not set.
# To run tests on a different device, set the 'TORCH_DEVICE' environment variable.
# For example, to run tests on a CUDA device, use:
# TORCH_DEVICE='cuda' pytest torchtitan/experiments/tpu/test_llama_layers_parity.py
DEVICE = torch.device(os.environ.get("TORCH_DEVICE", "cpu"))


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


@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_llama_embedding_cpu_device_parity():
    """
    Tests the CPU vs. DEVICE parity of the Embedding layer.
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

    # DEVICE setup
    embedding_device = copy.deepcopy(embedding_cpu).to(DEVICE)
    tokens_device = tokens_cpu.detach().clone().to(DEVICE).requires_grad_(False)

    # Forward pass
    out_cpu = embedding_cpu(tokens_cpu)
    out_device = embedding_device(tokens_device)
    assert torch.allclose(out_cpu, out_device.cpu(), atol=1e-5, rtol=1e-5)

    # Backward pass
    # Add requires_grad=True for output to calculate gradients
    out_cpu.requires_grad_(True)
    out_device.requires_grad_(True)
    loss_cpu = out_cpu.sum()
    loss_device = out_device.sum()
    loss_cpu.backward()
    loss_device.backward()

    assert torch.allclose(
        embedding_cpu.weight.grad,
        embedding_device.weight.grad.cpu(),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_llama_attention_cpu_device_parity():
    """
    Tests the CPU vs. DEVICE parity of the Attention layer.
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

    # DEVICE setup
    attention_device = copy.deepcopy(attention_cpu).to(DEVICE)
    x_device = x_cpu.detach().clone().to(DEVICE).requires_grad_(True)
    freqs_cis_device = freqs_cis_cpu.to(DEVICE)

    # Forward pass
    out_cpu = attention_cpu(x_cpu, freqs_cis_cpu)
    out_device = attention_device(x_device, freqs_cis_device)
    assert torch.allclose(out_cpu, out_device.cpu(), atol=1e-5, rtol=1e-5)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_device = out_device.sum()
    loss_cpu.backward()
    loss_device.backward()

    assert torch.allclose(x_cpu.grad, x_device.grad.cpu(), atol=1e-5, rtol=1e-5)
    for p_cpu, p_device in zip(attention_cpu.parameters(), attention_device.parameters()):
        assert torch.allclose(p_cpu.grad, p_device.grad.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_llama_feedforward_cpu_device_parity():
    """
    Tests the CPU vs. DEVICE parity of the FeedForward layer.
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

    # DEVICE setup
    feedforward_device = copy.deepcopy(feedforward_cpu).to(DEVICE)
    x_device = x_cpu.detach().clone().to(DEVICE).requires_grad_(True)

    # Forward pass
    out_cpu = feedforward_cpu(x_cpu)
    out_device = feedforward_device(x_device)
    assert torch.allclose(out_cpu, out_device.cpu(), atol=1e-5, rtol=1e-5)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_device = out_device.sum()
    loss_cpu.backward()
    loss_device.backward()

    assert torch.allclose(x_cpu.grad, x_device.grad.cpu(), atol=1e-5, rtol=1e-5)
    for p_cpu, p_device in zip(
        feedforward_cpu.parameters(), feedforward_device.parameters()
    ):
        assert torch.allclose(p_cpu.grad, p_device.grad.cpu(), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_llama_transformer_block_cpu_device_parity():
    """
    Tests the CPU vs. DEVICE parity of the TransformerBlock layer.
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

    # DEVICE setup
    transformer_block_device = copy.deepcopy(transformer_block_cpu).to(DEVICE)
    x_device = x_cpu.detach().clone().to(DEVICE).requires_grad_(True)
    freqs_cis_device = freqs_cis_cpu.to(DEVICE)

    # Forward pass
    out_cpu = transformer_block_cpu(x_cpu, freqs_cis_cpu)
    out_device = transformer_block_device(x_device, freqs_cis_device)
    assert torch.allclose(out_cpu, out_device.cpu(), atol=1e-5, rtol=1e-5)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_device = out_device.sum()
    loss_cpu.backward()
    loss_device.backward()

    assert torch.allclose(x_cpu.grad, x_device.grad.cpu(), atol=1e-5, rtol=1e-5)
    for p_cpu, p_device in zip(
        transformer_block_cpu.parameters(), transformer_block_device.parameters()
    ):
        assert torch.allclose(p_cpu.grad, p_device.grad.cpu(), atol=1e-5, rtol=1e-5)
