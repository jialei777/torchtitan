"""
This script demonstrates how to train a Qwen3 model using FSDP.

To run this script, use the following command:

.. code-block:: bash

    # To run on a specific accelerator (e.g., CUDA) with a specified backend (e.g., NCCL):
    TORCH_BACKEND=nccl torchrun --nproc_per_node=4 torchtitan/experiments/tpu/qwen3/run_qwen_training_fsdp.py

NOTE: have not tested with --nproc_per_node > 1 yet

"""

import os
import time
import torch
import torch.nn as nn
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy

from torchtitan.experiments.qwen3.model.model import Qwen3Model, TransformerBlock
from torchtitan.experiments.qwen3.model.args import Qwen3ModelArgs

def main():
    # 1. Initialize Distributed Environment
    # The default backend is 'gloo', if 'TORCH_BACKEND' is not set.
    backend = os.environ.get("TORCH_BACKEND", "gloo")
    init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])

    if backend == "nccl":
        torch.cuda.set_device(local_rank)

    device_type = "cuda" if backend == "nccl" else "cpu"
    device = torch.device(device_type, local_rank)
    world_size = int(os.environ["WORLD_SIZE"])
    mesh = init_device_mesh(device_type, (world_size,))

    # 2. Model Configuration
    # Default Qwen3-like model configuration (~1.7B params)
    # model_args = Qwen3ModelArgs()
    
    # Small config to ensure success on smaller devices.
    model_args = Qwen3ModelArgs(
        dim=1024,
        n_layers=12,
        n_heads=8,
        # n_kv_heads=4,
        # vocab_size=32,
        max_seq_len=512,
        
        # Qwen3-specific arguments required by your model implementation
        head_dim=128, # dim // n_heads         
        hidden_dim=4096, # standard FFN hidden dim (4 * dim)
        
        # Use bfloat16 for training stability and memory efficiency
        # This mirrors the setting in the MixedPrecisionPolicy below
        attn_mask_type="causal",
    )
 
    
    # 3. Instantiate and Shard the Model
    # Instantiate the model directly on CPU (Meta-device is often preferred 
    # for production, but CPU works for this simpler example).
    model = Qwen3Model(model_args).to("cpu")
    model.init_weights() # Ensure weights are materialized/initialized before FSDP

    # Manually apply FSDP configuration
    # Note: Using torch.bfloat16 for modern accelerators (A100s).
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    fsdp_config = {"mesh": mesh, "mp_policy": mp_policy, "offload_policy": CPUOffloadPolicy()}

    # --- Qwen3 FSDP Wrapping Logic ---
    if model.tok_embeddings is not None:
        fully_shard(model.tok_embeddings, **fsdp_config)
    for layer_id, transformer_block in model.layers.items():
        fully_shard(transformer_block, **fsdp_config)
    if model.norm is not None and model.output is not None:
        fully_shard([model.norm, model.output], **fsdp_config)
    fully_shard(model, **fsdp_config)

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    if local_rank == 0:
        print(f"Model created and sharded across {world_size} devices with CPU offload.")
        print(f"Total model size: {num_params:.2f}B parameters")
        # The number of parameters on a single rank will be smaller
        local_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Model size on rank {local_rank}: {local_params:.2f}B parameters")


    # 4. Dummy Dataloader
    batch_size = 1
    seq_len = model_args.max_seq_len
    vocab_size = model_args.vocab_size

    def get_dummy_batch():
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        return tokens, labels

    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 6. Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # 7. Training Loop
    num_steps = 10
    burn_in_steps = 2
    total_tokens = 0
    total_time = 0.0

    if local_rank == 0:
        print("\n--- Starting FSDP Training with CPU Offload ---")

    for step in range(num_steps):
        torch.cuda.synchronize() if device.type == "cuda" else None
        step_start_time = time.time()
        
        optimizer.zero_grad()

        tokens, labels = get_dummy_batch()
        
        output = model(tokens)

        output = output.view(-1, vocab_size)
        labels = labels.view(-1)

        loss = loss_fn(output, labels)

        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        step_time = time.time() - step_start_time

        if step >= burn_in_steps:
            total_tokens += batch_size * seq_len * world_size
            total_time += step_time

        tokens_per_sec = (batch_size * seq_len) / step_time

        if local_rank == 0:
            print(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f} | Throughput: {tokens_per_sec * world_size:.2f} tokens/sec")

    # Calculate average throughput excluding burn-in steps
    if local_rank == 0 and num_steps > burn_in_steps:
        avg_throughput = total_tokens / total_time if total_time > 0 else 0
        print(f"\nAverage throughput: {avg_throughput:.2f} tokens/sec")

    destroy_process_group()

if __name__ == "__main__":
    main()