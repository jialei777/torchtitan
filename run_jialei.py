""" Test script for LLaMA 3 model components.

python run_jialei.py 
----
Max diff for Attention: 2.9802322387695312e-08
Max diff for MLP: 7.450580596923828e-09
Max diff for TransformerBlock: 1.1920928955078125e-07

"""

import copy
import types
import torch

####################################################

from torchtitan.models.llama3.model.model import (
    Attention,
    TransformerModelArgs,
    precompute_freqs_cis,
)

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
attention_cpu = Attention(args).cpu()
attention_cpu.init_weights(init_std=0.02)

x = torch.randn(batch, seq_len, args.dim, device="cpu")
freqs_cis = precompute_freqs_cis(
    args.dim // args.n_heads, seq_len, theta=args.rope_theta
)
out_cpu = attention_cpu(x, freqs_cis)

attention_gpu = copy.deepcopy(attention_cpu).cuda()
x_gpu = x.cuda()
freqs_cis_gpu = freqs_cis.cuda()
out_gpu = attention_gpu(x_gpu, freqs_cis_gpu).cpu()

print("Max diff for Attention:", (out_cpu - out_gpu).abs().max().item())

####################################################

from torchtitan.models.llama3.model.model import FeedForward

torch.manual_seed(0)
dim = 64
batch, seq_len = 2, 8
ff_cpu = FeedForward(dim=dim, hidden_dim=4 * dim, multiple_of=16, ffn_dim_multiplier=None)
ff_cpu.init_weights(init_std=0.02)

x = torch.randn(batch, seq_len, dim, device="cpu")
out_cpu = ff_cpu(x)

ff_gpu = copy.deepcopy(ff_cpu).cuda()
x_gpu = x.cuda()
out_gpu = ff_gpu(x_gpu).cpu()

print("Max diff for MLP:", (out_cpu - out_gpu).abs().max().item())



####################################################

from torchtitan.models.llama3.model.model import (
    TransformerBlock,
    TransformerModelArgs,
    precompute_freqs_cis,
)


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

x = torch.randn(batch, seq_len, args.dim, device="cpu")
freqs_cis = precompute_freqs_cis(
    args.dim // args.n_heads, seq_len, theta=args.rope_theta
)
out_cpu = block_cpu(x, freqs_cis)

block_gpu = copy.deepcopy(block_cpu).cuda()
x_gpu = x.cuda()
freqs_cis_gpu = freqs_cis.cuda()
out_gpu = block_gpu(x_gpu, freqs_cis_gpu).cpu()

print("Max diff for TransformerBlock:", (out_cpu - out_gpu).abs().max().item())


print("\nDone!")