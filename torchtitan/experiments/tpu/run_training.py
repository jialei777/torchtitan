"""
This script provides a simple example of training a Llama-like model using TorchTitan.

It demonstrates how to:
- Instantiate a Llama-style model from the `torchtitan` library.
- Create a dummy dataloader for training.
- Run a basic training loop on a specified device (CPU or other accelerators).
- Compare the throughput between CPU and the specified device.

To run this script, use the following command:

.. code-block:: bash

    # To run on CPU
    python -m torchtitan.experiments.tpu.run_training

    # To run on a different device (e.g., CUDA)
    TORCH_DEVICE=cuda python -m torchtitan.experiments.tpu.run_training

"""

import os
import time
import torch
import torch.nn as nn
from torchtitan.models.llama3.model.model import Transformer
from torchtitan.models.llama3.model.args import TransformerModelArgs


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


def train_on_device(device_str):
    """
    Runs a simple training loop on the specified device and returns the average throughput.
    """
    device = torch.device(device_str)
    if not is_device_available(device):
        print(f"--- SKIPPING TRAINING ON {device_str.upper()} ---")
        print(f"Device {device_str.upper()} not available.")
        return 0

    print(f"--- Starting training on {device_str.upper()} ---")
    # 1. Model Configuration
    try:
        # Reduced model size to fit on a 40G accelerator.
        # Original default is dim=4096, n_layers=32, n_heads=32 (approx. 7B parameters)
        model_args = TransformerModelArgs(
            dim=2048,
            n_layers=24,
            n_heads=16,
            max_seq_len=2048
        )

        # 2. Instantiate the Model
        model = Transformer(model_args).to(device)
        num_params = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"Model created and moved to {device_str}.")
        print(f"Model size: {num_params:.2f}B parameters")

        # 3. Dummy Dataloader
        batch_size = 1
        seq_len = model_args.max_seq_len
        vocab_size = model_args.vocab_size

        def get_dummy_batch():
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            return tokens, labels

        # 4. Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # 5. Loss Function
        loss_fn = nn.CrossEntropyLoss()

        # 6. Training Loop
        num_steps = 5
        # Burn-in steps to ignore for throughput calculation
        burn_in_steps = 2
        total_tokens = 0
        total_time = 0.0

        for step in range(num_steps):
            step_start_time = time.time()
            optimizer.zero_grad()

            tokens, labels = get_dummy_batch()

            output = model(tokens)

            # Reshape for CrossEntropyLoss
            output = output.view(-1, vocab_size)
            labels = labels.view(-1)

            loss = loss_fn(output, labels)

            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start_time

            if step >= burn_in_steps:
                total_tokens += batch_size * seq_len
                total_time += step_time

            tokens_per_sec = (batch_size * seq_len) / step_time

            print(f"Step {step+1}/{num_steps} | Loss: {loss.item():.4f} | Throughput: {tokens_per_sec:.2f} tokens/sec")

        # Calculate average throughput excluding burn-in steps
        avg_throughput = total_tokens / total_time if total_time > 0 else 0
        print(f"\nAverage throughput on {device_str.upper()}: {avg_throughput:.2f} tokens/sec")
        return avg_throughput

    except torch.cuda.OutOfMemoryError:
        print("\n--- OUT OF MEMORY ---")
        print(f"The model is too large to train on the {device_str.upper()} with the current configuration.")
        print("Try reducing the model size or sequence length.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

def main():
    """
    Main function to run the training on different devices and compare throughput.
    """
    # The default device is 'cpu', if 'TORCH_DEVICE' is not set.
    # To run tests on a different device, set the 'TORCH_DEVICE' environment variable.
    device_str = os.environ.get("TORCH_DEVICE", "cpu")

    cpu_throughput = train_on_device("cpu")

    device_throughput = 0
    if device_str != "cpu":
        device_throughput = train_on_device(device_str)

    print("\n--- Throughput Comparison ---")
    print(f"CPU: {cpu_throughput:.2f} tokens/sec")
    if device_throughput > 0:
        print(f"{device_str.upper()}: {device_throughput:.2f} tokens/sec")
        if cpu_throughput > 0:
            speedup = device_throughput / cpu_throughput
            print(f"\n{device_str.upper()} is approximately {speedup:.2f}x faster than CPU.")

if __name__ == "__main__":
    main()
