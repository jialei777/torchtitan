import copy

import pytest
import torch

from torchtitan.models.llama3.model.args import TransformerModelArgs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llama_embedding_cpu_gpu_parity():
    torch.manual_seed(0)
    args = TransformerModelArgs(dim=64, vocab_size=32)
    embedding_cpu = torch.nn.Embedding(args.vocab_size, args.dim).cpu()
    torch.nn.init.normal_(embedding_cpu.weight)

    embedding_gpu = copy.deepcopy(embedding_cpu).cuda()

    tokens = torch.randint(0, args.vocab_size, (2, 8), device="cpu")
    tokens_gpu = tokens.cuda()

    out_cpu = embedding_cpu(tokens)
    out_gpu = embedding_gpu(tokens_gpu)
    assert torch.allclose(out_cpu, out_gpu.cpu(), atol=1e-5, rtol=1e-5)

    loss_cpu = out_cpu.sum()
    loss_gpu = out_gpu.sum()
    loss_cpu.backward()
    loss_gpu.backward()

    assert torch.allclose(
        embedding_cpu.weight.grad, embedding_gpu.weight.grad.cpu(), atol=1e-5, rtol=1e-5
    )
