import copy
import os

import pytest
import torch
from torch import optim
# Qwen3 model imports per experiments/qwen3/model
from torchtitan.experiments.qwen3.model.model import Qwen3Model, Qwen3ModelArgs


# The default device is 'cpu', if 'TORCH_DEVICE' is not set.
# To run tests on a different device, set the 'TORCH_DEVICE' environment variable.
# For example, to run tests on a CUDA device, use:
# TORCH_DEVICE='cuda' pytest torchtitan/experiments/tpu/qwen3/test_qwen_full_model_parity.py
DEVICE = torch.device(os.environ.get("TORCH_DEVICE", "cpu"))
# Increased tolerance to account for Q-K normalization
ATOL_FWD = 1e-4
RTOL_FWD = 1e-4
ATOL_BWD = 5e-4
RTOL_BWD = 5e-4

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

def _setup_qwen_models(args):
    """Initializes CPU and DEVICE models with identical, pre-initialized weights."""
    model_cpu = Qwen3Model(args).cpu() 
    model_cpu.init_weights()
    model_cpu.train()
    
    model_device = copy.deepcopy(model_cpu).to(DEVICE)
    model_device.train()
    
    return model_cpu, model_device

def _setup_qwen_args(seed=0):
    """Sets a fixed seed and returns standard model arguments for the test."""
    torch.manual_seed(seed)
    return Qwen3ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=32,
        max_seq_len=16,
    )

def _setup_input_tensors(args, batch_size=2, seq_len=8, device=DEVICE):
    """Creates input tensors on both CPU and target device."""
    tokens_cpu = torch.randint(0, args.vocab_size, (batch_size, seq_len), device="cpu")
    tokens_device = tokens_cpu.to(device)
    return tokens_cpu, tokens_device

# --- Unit Tests ---
@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_forward_parity():
    """
    Tests only the forward pass output parity (logits) between CPU and DEVICE.
    """
    args = _setup_qwen_args()
    model_cpu, model_device = _setup_qwen_models(args)
    tokens_cpu, tokens_device = _setup_input_tensors(args, device=DEVICE)

    # 1. Forward Pass
    with torch.no_grad(): 
        out_cpu = model_cpu(tokens_cpu)
        out_device = model_device(tokens_device)
    
    # 2. Check Forward Parity
    assert torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_FWD, rtol=RTOL_FWD)

@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_backward_parity():
    """
    Tests only the gradient parity (Backward Pass) between CPU and DEVICE.
    The gradient tolerance is slightly increased for stability.
    """
    args = _setup_qwen_args()
    model_cpu, model_device = _setup_qwen_models(args)
    tokens_cpu, tokens_device = _setup_input_tensors(args, device=DEVICE)
    
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01)
    optimizer_device = optim.SGD(model_device.parameters(), lr=0.01)

    # 1. Forward Pass 
    out_cpu = model_cpu(tokens_cpu)
    out_device = model_device(tokens_device)
    
    # 2. Backward Pass
    loss_cpu = out_cpu.sum()
    loss_device = out_device.sum()
    
    # Zero out any residual gradients
    optimizer_cpu.zero_grad()
    optimizer_device.zero_grad()
    
    loss_cpu.backward()
    loss_device.backward()

    # 3. Check Gradients Parity 
    for p_cpu, p_device in zip(model_cpu.parameters(), model_device.parameters()):
        if p_cpu.grad is not None and p_device.grad is not None:
            assert torch.allclose(
                p_cpu.grad, p_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD
            )
        elif p_cpu.grad is None and p_device.grad is None:
            pass
        else:
            pytest.fail(
                "Mismatch in gradient presence between CPU and DEVICE parameters."
            )


@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_qwen_full_model_training_step_cpu_device_parity():
    """
    Tests the CPU vs. DEVICE parity of the full Qwen model for 3 training steps.
    Each step includes a forward pass, a backward pass, and a gradient update.
    """
    args = _setup_qwen_args()
    model_cpu, model_device = _setup_qwen_models(args)
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01)
    optimizer_device = optim.SGD(model_device.parameters(), lr=0.01)
    batch, seq_len = 2, 8


    # Run 3 training steps
    for _ in range(3):
        tokens = torch.randint(0, args.vocab_size, (batch, seq_len), device="cpu")
        tokens_device = tokens.to(DEVICE)

        # Zero gradients
        optimizer_cpu.zero_grad()
        optimizer_device.zero_grad()

        # Forward pass
        out_cpu = model_cpu(tokens)
        out_device = model_device(tokens_device)
        assert torch.allclose(out_cpu, out_device.cpu(), atol=1e-4, rtol=1e-4)

        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        # Check gradients before optimizer step
        for p_cpu, p_device in zip(model_cpu.parameters(), model_device.parameters()):
            if p_cpu.grad is not None and p_device.grad is not None:
                assert torch.allclose(
                    # Increased gradient tolerance to account for Q-K normalization
                    p_cpu.grad, p_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD
                )
            elif p_cpu.grad is None and p_device.grad is None:
                pass  # Gradients are both None, which is fine
            else:
                pytest.fail(
                    "Mismatch in gradient presence between CPU and DEVICE parameters."
                )

        # Optimizer step
        optimizer_cpu.step()
        optimizer_device.step()

    # Check model parameters are in sync after training steps
    for p_cpu, p_device in zip(model_cpu.parameters(), model_device.parameters()):
        assert torch.allclose(p_cpu, p_device.cpu(), atol=1e-4, rtol=1e-4)
