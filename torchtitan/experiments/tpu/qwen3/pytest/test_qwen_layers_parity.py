import copy
import os

import pytest
import torch
from torch import nn

from torchtitan.experiments.qwen3.model.model import (
    Attention,
    FeedForward,
    precompute_rope_cache,
    TransformerBlock,
    Qwen3ModelArgs,
)

# The default device is 'cpu', if 'TORCH_DEVICE' is not set.
# To run tests on a different device, set the 'TORCH_DEVICE' environment variable.
# For example, to run tests on a CUDA device, use:
# TORCH_DEVICE='cuda' pytest torchtitan/experiments/tpu/qwen3/pytest/test_qwen_layers_parity.py
DEVICE = torch.device(os.environ.get("TORCH_DEVICE", "cpu"))
# Increased tolerance to account for Q-K normalization
ATOL_FWD = 1e-4
RTOL_FWD = 1e-4
ATOL_BWD = 5e-4
RTOL_BWD = 5e-4
# Tighter tolerance for layers without Q-K normalization
ATOL_EMB = 1e-5
RTOL_EMB = 1e-5

# --- Utility funcs ---
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
    return Qwen3ModelArgs(
        dim=64,
        n_layers=n_layers,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,

        head_dim=16,
        hidden_dim=256,

        rope_theta=100000.0,
    )

def _setup_attention_tensors(args, batch, seq_len):
    """Creates inputs and caches for the Attention layer."""
    x_cpu = torch.randn(batch, seq_len, args.dim, device="cpu", requires_grad=True)
    rope_cache_cpu = precompute_rope_cache(
        args.head_dim, args.max_seq_len, base=args.rope_theta
    )
    
    x_device = x_cpu.detach().clone().to(DEVICE).requires_grad_(True)
    rope_cache_device = rope_cache_cpu.to(DEVICE)
    
    return x_cpu, x_device, rope_cache_cpu, rope_cache_device

def _setup_feedforward_tensors(args, batch, seq_len):
    """Creates input tensors for the FeedForward layer."""
    x_cpu = torch.randn(batch, seq_len, args.dim, device="cpu", requires_grad=True)
    x_device = x_cpu.detach().clone().to(DEVICE).requires_grad_(True)
    
    return x_cpu, x_device

# --- Unit Tests ---

@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_embedding_cpu_device_parity():
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
    assert torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_EMB, rtol=RTOL_EMB)

    # Backward pass
    out_cpu.requires_grad_(True)
    out_device.requires_grad_(True)
    loss_cpu = out_cpu.sum()
    loss_device = out_device.sum()
    loss_cpu.backward()
    loss_device.backward()

    # Gradient check 
    assert torch.allclose(
        embedding_cpu.weight.grad,
        embedding_device.weight.grad.cpu(),
        atol=ATOL_EMB,
        rtol=RTOL_EMB,
    )

@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_attention_forward_parity():
    """Tests the CPU vs. DEVICE parity of the Qwen3 Attention layer's forward pass."""
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8

    attention_cpu = Attention(args).cpu()
    attention_cpu.init_weights(init_std=0.02)

    attention_device = copy.deepcopy(attention_cpu).to(DEVICE)
    x_cpu, x_device, rope_cache_cpu, rope_cache_device = _setup_attention_tensors(args, batch, seq_len)
    
    with torch.no_grad():
        out_cpu = attention_cpu(x_cpu, rope_cache_cpu)
        out_device = attention_device(x_device, rope_cache_device)
    
    assert torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_FWD, rtol=RTOL_FWD)

@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_attention_backward_parity():
    """Tests the CPU vs. DEVICE parity of the Qwen3 Attention layer's backward pass."""
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8

    attention_cpu = Attention(args).cpu()
    attention_cpu.init_weights(init_std=0.02)

    attention_device = copy.deepcopy(attention_cpu).to(DEVICE)
    x_cpu, x_device, rope_cache_cpu, rope_cache_device = _setup_attention_tensors(args, batch, seq_len)

    # Forward pass
    out_cpu = attention_cpu(x_cpu, rope_cache_cpu)
    out_device = attention_device(x_device, rope_cache_device)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_device = out_device.sum()
    loss_cpu.backward()
    loss_device.backward()

    assert torch.allclose(x_cpu.grad, x_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD)
    for p_cpu, p_device in zip(attention_cpu.parameters(), attention_device.parameters()):
        assert torch.allclose(p_cpu.grad, p_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD)

@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_feedforward_forward_parity():
    """Tests the CPU vs. DEVICE parity of the FeedForward layer's forward pass."""
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8

    feedforward_cpu = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim).cpu()
    feedforward_cpu.init_weights(init_std=0.02)
    
    feedforward_device = copy.deepcopy(feedforward_cpu).to(DEVICE)
    x_cpu, x_device = _setup_feedforward_tensors(args, batch, seq_len)
    
    with torch.no_grad():
        out_cpu = feedforward_cpu(x_cpu)
        out_device = feedforward_device(x_device)
        
    assert torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_EMB, rtol=RTOL_EMB)

@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_feedforward_backward_parity():
    """Tests the CPU vs. DEVICE parity of the FeedForward layer's backward pass."""
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8

    feedforward_cpu = FeedForward(dim=args.dim, hidden_dim=args.hidden_dim).cpu()
    feedforward_cpu.init_weights(init_std=0.02)
    
    feedforward_device = copy.deepcopy(feedforward_cpu).to(DEVICE)
    x_cpu, x_device = _setup_feedforward_tensors(args, batch, seq_len)

    # Forward pass
    out_cpu = feedforward_cpu(x_cpu)
    out_device = feedforward_device(x_device)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_device = out_device.sum()
    loss_cpu.backward()
    loss_device.backward()

    # Gradient check
    assert torch.allclose(x_cpu.grad, x_device.grad.cpu(), atol=ATOL_EMB, rtol=ATOL_EMB)
    for p_cpu, p_device in zip(
        feedforward_cpu.parameters(), feedforward_device.parameters()
    ):
        assert torch.allclose(p_cpu.grad, p_device.grad.cpu(), atol=ATOL_EMB, rtol=ATOL_EMB)

@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_transformer_block_forward_parity():
    """Tests the CPU vs. DEVICE parity of the TransformerBlock layer's forward pass."""
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8
    layer_id = 0

    transformer_block_cpu = TransformerBlock(layer_id, args).cpu()
    transformer_block_cpu.init_weights()
    
    transformer_block_device = copy.deepcopy(transformer_block_cpu).to(DEVICE)
    x_cpu, x_device, rope_cache_cpu, rope_cache_device = _setup_attention_tensors(args, batch, seq_len)
    
    with torch.no_grad():
        out_cpu = transformer_block_cpu(x_cpu, rope_cache_cpu)
        out_device = transformer_block_device(x_device, rope_cache_device)
        
    assert torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_FWD, rtol=RTOL_FWD)


@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_transformer_block_backward_parity():
    """Tests the CPU vs. DEVICE parity of the TransformerBlock layer's backward pass."""
    torch.manual_seed(0)
    args = _get_model_args()
    batch, seq_len = 2, 8
    layer_id = 0

    transformer_block_cpu = TransformerBlock(layer_id, args).cpu()
    transformer_block_cpu.init_weights()
    
    transformer_block_device = copy.deepcopy(transformer_block_cpu).to(DEVICE)
    x_cpu, x_device, rope_cache_cpu, rope_cache_device = _setup_attention_tensors(args, batch, seq_len)

    x_cpu.requires_grad_(True)
    x_device.requires_grad_(True)

    # Forward pass
    out_cpu = transformer_block_cpu(x_cpu, rope_cache_cpu)
    out_device = transformer_block_device(x_device, rope_cache_device)

    # Backward pass
    loss_cpu = out_cpu.sum()
    loss_device = out_device.sum()
    loss_cpu.backward()
    loss_device.backward()

    # Gradient check 
    assert torch.allclose(x_cpu.grad, x_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD)
    for p_cpu, p_device in zip(
        transformer_block_cpu.parameters(), transformer_block_device.parameters()
    ):
        assert torch.allclose(p_cpu.grad, p_device.grad.cpu(), atol=ATOL_BWD, rtol=ATOL_BWD)    