import copy
import os

import pytest
import torch
from torch import optim

from torchtitan.models.llama3.model.model import Transformer, TransformerModelArgs


# The default device is 'cpu', if 'TORCH_DEVICE' is not set.
# To run tests on a different device, set the 'TORCH_DEVICE' environment variable.
# For example, to run tests on a CUDA device, use:
# TORCH_DEVICE='cuda' pytest torchtitan/experiments/tpu/test_llama_full_model_parity.py
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


@pytest.mark.skipif(
    not is_device_available(DEVICE) or DEVICE.type == "cpu",
    reason=f"Device {DEVICE} not available or is CPU",
)
def test_llama_full_model_training_step_cpu_device_parity():
    """
    Tests the CPU vs. DEVICE parity of the full Llama model for 3 training steps.
    Each step includes a forward pass, a backward pass, and a gradient update.
    """
    torch.manual_seed(0)
    args = TransformerModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=32,
        max_seq_len=16,
        multiple_of=16,
    )
    batch, seq_len = 2, 8

    # CPU setup
    model_cpu = Transformer(args).cpu()
    model_cpu.init_weights()
    optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01)

    # DEVICE setup
    model_device = copy.deepcopy(model_cpu).to(DEVICE)
    optimizer_device = optim.SGD(model_device.parameters(), lr=0.01)

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
        assert torch.allclose(out_cpu, out_device.cpu(), atol=1e-5, rtol=1e-5)

        # Backward pass
        loss_cpu = out_cpu.sum()
        loss_device = out_device.sum()
        loss_cpu.backward()
        loss_device.backward()

        # Check gradients before optimizer step
        for p_cpu, p_device in zip(model_cpu.parameters(), model_device.parameters()):
            if p_cpu.grad is not None and p_device.grad is not None:
                assert torch.allclose(
                    p_cpu.grad, p_device.grad.cpu(), atol=1e-4, rtol=1e-4
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
