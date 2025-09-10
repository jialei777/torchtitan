import copy

import pytest
import torch

from torchtitan.models.llama3.model.model import (
    precompute_freqs_cis,
    TransformerBlock,
    TransformerModelArgs,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llama_transformerblock_cpu_gpu_parity():
    torch.manual_seed(0)
    args = TransformerModelArgs(
        dim=64,
        n_layers=1,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=32,
        max_seq_len=16,
        multiple_of=16,
    )
    batch, seq_len = 2, 8
    block_cpu = TransformerBlock(layer_id=0, model_args=args).cpu()
    block_cpu.init_weights()

    block_gpu = copy.deepcopy(block_cpu).cuda()

    x = torch.randn(batch, seq_len, args.dim, device="cpu", requires_grad=True)
    x_gpu = x.detach().clone().cuda().requires_grad_(True)
    freqs_cis = precompute_freqs_cis(
        args.dim // args.n_heads, seq_len, theta=args.rope_theta
    )
    freqs_cis_gpu = freqs_cis.cuda()

    out_cpu = block_cpu(x, freqs_cis)
    out_gpu = block_gpu(x_gpu, freqs_cis_gpu)
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5, rtol=1e-5)

    loss_cpu = out_cpu.sum()
    loss_gpu = out_gpu.sum()
    loss_cpu.backward()
    loss_gpu.backward()

    assert torch.allclose(x.grad, x_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)
    for p_cpu, p_gpu in zip(block_cpu.parameters(), block_gpu.parameters()):
        assert torch.allclose(p_cpu.grad, p_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)
