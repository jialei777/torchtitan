import copy

import pytest
import torch

from torchtitan.models.llama3.model.model import Transformer, TransformerModelArgs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llama_model_cpu_gpu_parity():
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
    model_cpu = Transformer(args).cpu()
    model_cpu.init_weights()

    tokens = torch.randint(0, args.vocab_size, (2, 8), device="cpu")

    model_gpu = copy.deepcopy(model_cpu).cuda()
    tokens_gpu = tokens.cuda()

    out_cpu = model_cpu(tokens)
    out_gpu = model_gpu(tokens_gpu)
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5, rtol=1e-5)

    loss_cpu = out_cpu.sum()
    loss_gpu = out_gpu.sum()
    loss_cpu.backward()
    loss_gpu.backward()

    for p_cpu, p_gpu in zip(model_cpu.parameters(), model_gpu.parameters()):
        assert torch.allclose(p_cpu.grad, p_gpu.grad.cpu(), atol=1e-5, rtol=1e-5)
