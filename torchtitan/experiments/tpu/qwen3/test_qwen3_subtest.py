"""Tests for Qwen3 model components on GPU and accelerator devices, using subtests."""

import copy
import os
import unittest
from absl import flags
from absl import logging
from absl.testing import absltest
import torch
from torch import nn
from torch import optim

from torchtitan.experiments.qwen3.model.model import (
    Attention,
    FeedForward,
    precompute_rope_cache,
    TransformerBlock,
    Qwen3ModelArgs,
    Qwen3Model
)

# --- Configuration Flags ---

_DEVICE = flags.DEFINE_enum(
    "device",
    "cpu",
    ["tpu", "cuda", "cpu"],
    help="Accelerator to test.",
)
    
# Tolerances
ATOL_FWD, RTOL_FWD = 1e-4, 1e-4 # Forward pass tolerances
ATOL_BWD, RTOL_BWD = 5e-4, 5e-4 # Backward pass tolerances
ATOL_EMB, RTOL_EMB = 1e-5, 1e-5 # Tighter tolerance for simple layers

# --- Utility funcs (External) ---

def _is_device_available(device):
    """Checks if the given torch device is available."""
    if device.type == "cuda":
        return torch.cuda.is_available()
    return True


class Qwen3ParityTest(absltest.TestCase):
    """Tests for Qwen3 model components on GPU and accelerator devices."""

    def setUp(self):
        super().setUp()

        # Random Seed setup
        seed = absltest.FLAGS.test_random_seed
        if seed is None or not isinstance(seed, int):
            raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)

        torch.manual_seed(seed)
        logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)
        
        # Device setup
        self.device = torch.device(_DEVICE.value)
        reason = None
        
        if self.device.type == "cpu":
            reason = "Test skipped because DEVICE is set to CPU."
        elif self.device.type == "cuda" and not torch.cuda.is_available():
            reason = "CUDA device chosen but not available on this machine."
        
        if reason:
            self.skipTest(reason)
        
        # Model Args setup
        self.args = self._get_model_args()
        self.batch, self.seq_len = 2, 8
        self.dim = self.args.dim

        
    def _get_model_args(self, n_layers=2, vocab_size=32, max_seq_len=16):
        return Qwen3ModelArgs(
            dim=64, n_layers=n_layers, n_heads=4, n_kv_heads=4, vocab_size=vocab_size,
            max_seq_len=max_seq_len, head_dim=16, hidden_dim=256, rope_theta=100000.0,
        )
    
    def _setup_model(self, model_class, *model_args):
        """Initializes and copies a model/layer."""
        model_cpu = model_class(*model_args).cpu()
        if hasattr(model_cpu, 'init_weights'):
            if isinstance(model_cpu, (FeedForward, Attention)):
                model_cpu.init_weights(init_std=0.02)
            else:
                model_cpu.init_weights()
        elif isinstance(model_cpu, nn.Embedding):
            nn.init.normal_(model_cpu.weight)
            
        model_cpu.train()
        model_device = copy.deepcopy(model_cpu).to(self.device)
        model_device.train()
        return model_cpu, model_device

    def _create_input_tensor(self, size, requires_grad=False):
        x_cpu = torch.randn(size, device="cpu", requires_grad=requires_grad)
        x_device = x_cpu.detach().clone().to(self.device).requires_grad_(requires_grad)
        return x_cpu, x_device
    
    def _create_attention_tensors(self, requires_grad=True):
        x_cpu, x_device = self._create_input_tensor((self.batch, self.seq_len, self.dim), requires_grad=requires_grad)
        rope_cache_cpu = precompute_rope_cache(
            self.args.head_dim, self.args.max_seq_len, base=self.args.rope_theta
        )
        rope_cache_device = rope_cache_cpu.to(self.device)
        return x_cpu, x_device, rope_cache_cpu, rope_cache_device

    # --- Full Model tests ---
    def test_qwen_model_parity(self):
        """
        Tests the forward/backward parity for full Qwen3Model.
        """
        args = self._get_model_args()
        model_cpu, model_device = self._setup_model(Qwen3Model, args)
        
        tokens_cpu = torch.randint(0, args.vocab_size, (self.batch, self.seq_len), device="cpu")
        tokens_device = tokens_cpu.to(self.device)
        
        optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01) 
        optimizer_device = optim.SGD(model_device.parameters(), lr=0.01)

        # Subtest 1: Full Model Forward Pass
        with self.subTest(name="FullModel_Forward"):
            with torch.no_grad():
                # Forward pass
                out_cpu = model_cpu(tokens_cpu)
                out_device = model_device(tokens_device)
            # Check forward parity
            self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_FWD, rtol=RTOL_FWD))

        # Subtest 2: Full Model Backward Pass
        with self.subTest(name="FullModel_Backward"):
            # Make sure grad tracking is enabled in this subtest
            for p in model_cpu.parameters(): p.requires_grad_(True)
            for p in model_device.parameters(): p.requires_grad_(True)
            
            # Forward Pass
            out_cpu = model_cpu(tokens_cpu) 
            out_device = model_device(tokens_device)

            # Backward pass
            loss_cpu, loss_device = out_cpu.sum(), out_device.sum()
            # Zero gradients
            optimizer_cpu.zero_grad()
            optimizer_device.zero_grad()
            loss_cpu.backward()
            loss_device.backward()

            # Gradient check
            for p_cpu, p_device in zip(model_cpu.parameters(), model_device.parameters()):
                if p_cpu.grad is not None and p_device.grad is not None:
                    self.assertTrue(torch.allclose(
                        p_cpu.grad, p_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD
                    ))
                elif p_cpu.grad is None and p_device.grad is None:
                    pass
                else:
                    self.fail("Mismatch in gradient presence between CPU and DEVICE parameters.")
   
   
   # --- Layer tests ---
    def test_qwen_layers_parity(self):
        """
        Tests forward/backward parity for all individual Qwen3 layers.
        """
        args = self.args
        
        # Subtest 1: Embedding Layer
        with self.subTest(layer="Embedding"):
            embedding_cpu, embedding_device = self._setup_model(nn.Embedding, self.args.vocab_size, self.dim)
            tokens_cpu = torch.randint(0, self.args.vocab_size, (self.batch, self.seq_len), device="cpu")
            tokens_device = tokens_cpu.to(self.device)

            # Subtest 1a: Embedding Forward Pass
            with self.subTest(name="Forward"):
                with torch.no_grad():
                    # Forward pass
                    out_cpu = embedding_cpu(tokens_cpu)
                    out_device = embedding_device(tokens_device)
                # Check forward parity
                self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_EMB, rtol=RTOL_EMB))

            # Subtest 1b: Embedding Backward Pass
            with self.subTest(name="Backward"):
                # Forward pass
                out_cpu = embedding_cpu(tokens_cpu)
                out_device = embedding_device(tokens_device)
                # Backward pass
                loss_cpu, loss_device = out_cpu.sum(), out_device.sum()
                loss_cpu.backward()
                loss_device.backward()
                # Gradient check
                self.assertTrue(torch.allclose(
                    embedding_cpu.weight.grad, 
                    embedding_device.weight.grad.cpu(), 
                    atol=ATOL_EMB, 
                    rtol=RTOL_EMB,
                ))

        # Subtest 2: Attention Layer
        with self.subTest(layer="Attention"):
            attention_cpu, attention_device = self._setup_model(Attention, args)
            x_cpu, x_device, rope_cache_cpu, rope_cache_device = self._create_attention_tensors(requires_grad=True)

            # Subtest 2a: Attention Forward Pass
            with self.subTest(name="Forward"):
                with torch.no_grad():
                    # Forward pass
                    out_cpu = attention_cpu(x_cpu, rope_cache_cpu)
                    out_device = attention_device(x_device, rope_cache_device)
                # Check forward parity
                self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_FWD, rtol=RTOL_FWD))

            # Subtest 2b: Attention Backward Pass
            with self.subTest(name="Backward"):
                # Forward pass
                out_cpu = attention_cpu(x_cpu, rope_cache_cpu)
                out_device = attention_device(x_device, rope_cache_device)
                # Backward pass
                loss_cpu, loss_device = out_cpu.sum(), out_device.sum()
                loss_cpu.backward()
                loss_device.backward()
                # Gradient check
                self.assertTrue(torch.allclose(x_cpu.grad, x_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD))
                for p_cpu, p_device in zip(attention_cpu.parameters(), attention_device.parameters()):
                    if p_cpu.grad is not None and p_device.grad is not None:
                        self.assertTrue(torch.allclose(
                            p_cpu.grad, p_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD
                        ))
                    elif p_cpu.grad is None and p_device.grad is None:
                        pass
                    else:
                        self.fail("Mismatch in gradient presence between CPU and DEVICE parameters.")

        # Subtest 3: FeedForward Layer
        with self.subTest(layer="FeedForward"):
            feedforward_cpu, feedforward_device = self._setup_model(FeedForward, args.dim, args.hidden_dim)
            x_cpu, x_device = self._create_input_tensor((self.batch, self.seq_len, self.dim), requires_grad=True)

            # Subtest 3a: FeedForward Forward Pass
            with self.subTest(name="Forward"):
                with torch.no_grad():
                    # Forward pass
                    out_cpu = feedforward_cpu(x_cpu)
                    out_device = feedforward_device(x_device)
                # Check forward parity
                self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_EMB, rtol=RTOL_EMB))

            # Subtest 3b: FeedForward Backward Pass
            with self.subTest(name="Backward"):
                # Forward pass
                out_cpu = feedforward_cpu(x_cpu)
                out_device = feedforward_device(x_device)
                # Backward pass
                loss_cpu, loss_device = out_cpu.sum(), out_device.sum()
                loss_cpu.backward()
                loss_device.backward()
                # Gradient check
                self.assertTrue(torch.allclose(x_cpu.grad, x_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD))
                for p_cpu, p_device in zip(feedforward_cpu.parameters(), feedforward_device.parameters()):
                    if p_cpu.grad is not None and p_device.grad is not None:
                        self.assertTrue(torch.allclose(
                            p_cpu.grad, p_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD
                        ))
                    elif p_cpu.grad is None and p_device.grad is None:
                        pass
                    else:
                        self.fail("Mismatch in gradient presence between CPU and DEVICE parameters.")

        # Subtest 4: TransformerBlock Layer
        with self.subTest(layer="TransformerBlock"):
            layer_id = 0
            block_cpu, block_device = self._setup_model(TransformerBlock, layer_id, args)
            x_cpu, x_device, rope_cache_cpu, rope_cache_device = self._create_attention_tensors(requires_grad=True)

            # Subtest 4a: TransformerBlock Forward Pass
            with self.subTest(name="Forward"):
                with torch.no_grad():
                    # Forward pass
                    out_cpu = block_cpu(x_cpu, rope_cache_cpu)
                    out_device = block_device(x_device, rope_cache_device)
                # Check forward parity
                self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_FWD, rtol=RTOL_FWD))

            # Subtest 4b: TransformerBlock Backward Pass
            with self.subTest(name="Backward"):
                # Forward pass
                out_cpu = block_cpu(x_cpu, rope_cache_cpu)
                out_device = block_device(x_device, rope_cache_device)
                # Backward pass
                loss_cpu, loss_device = out_cpu.sum(), out_device.sum()
                loss_cpu.backward()
                loss_device.backward()
                 # Gradient check
                self.assertTrue(torch.allclose(x_cpu.grad, x_device.grad.cpu(), atol=ATOL_BWD, rtol=RTOL_BWD))
                for p_cpu, p_device in zip(
                    block_cpu.parameters(), block_device.parameters()
                ):
                    self.assertTrue(torch.allclose(p_cpu.grad, p_device.grad.cpu(), atol=ATOL_BWD, rtol=ATOL_BWD))

if __name__ == "__main__":
    absltest.main()