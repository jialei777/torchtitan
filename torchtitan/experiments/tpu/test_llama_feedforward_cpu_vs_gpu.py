import copy

import pytest
import torch

from torchtitan.models.llama3.model.model import FeedForward


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llama_feedforward_cpu_gpu_parity():
    torch.manual_seed(0)
    dim = 64
    batch, seq_len = 2, 8
    ff_cpu = FeedForward(
        dim=dim, hidden_dim=4 * dim, multiple_of=16, ffn_dim_multiplier=None
    )
    ff_cpu.init_weights(init_std=0.02)

    ff_gpu = copy.deepcopy(ff_cpu).cuda()

    x = torch.randn(batch, seq_len, dim, device="cpu", requires_grad=True)
    x_gpu = x.detach().clone().cuda().requires_grad_(True)

    out_cpu = ff_cpu(x)
    out_gpu = ff_gpu(x_gpu)
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5, rtol=1e-5)

    loss_cpu = out_cpu.sum()
    loss_gpu = out_gpu.sum()
    loss_cpu.backward()
    loss_gpu.backward()

    assert torch.allclose(x.grad, x_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)
    for p_cpu, p_gpu in zip(ff_cpu.parameters(), ff_gpu.parameters()):
        assert torch.allclose(p_cpu.grad, p_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)
