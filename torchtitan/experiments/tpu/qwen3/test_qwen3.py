"""Tests for Qwen3 model components on GPU and accelerator devices."""

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

_DEVICE = flags.DEFINE_enum(
    "device",
    "cpu",
    ["tpu", "cuda", "cpu"],
    help="Accelerator to test.",
)
    
# Tolerances (tightened to account for Q-K normalization)
ATOL_FWD, RTOL_FWD = 1e-4, 1e-4 # Forward pass tolerances
ATOL_BWD, RTOL_BWD = 5e-4, 5e-4 # Backward pass tolerances
ATOL_EMB, RTOL_EMB = 1e-5, 1e-5 # Embedding layer tolerances (tighter)

def _is_device_available(device):
    """Checks if the given torch device is available."""
    if device.type == "cuda":
        return torch.cuda.is_available()
    return True

class Qwen3Test(absltest.TestCase):
    """Tests for Qwen3 model components on GPU and accelerator devices."""

    def setUp(self):
        super().setUp()

        # Random Seed setup
        seed = absltest.FLAGS.test_random_seed
        if seed is None or not isinstance(seed, int):
            raise ValueError("absltest.FLAGS.test_random_seed not an int: %s" % seed)

        torch.manual_seed(seed)
        logging.info("Using absltest.FLAGS.test_random_seed: %d", seed)
        
        # Device setup (TODO: add tpu support in g3)
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
            dim=64,
            n_layers=n_layers,
            n_heads=4,
            n_kv_heads=4,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            head_dim=16,
            hidden_dim=256,
            rope_theta=100000.0,
        )
    
    def _setup_full_model(self, args):
        """Initializes full Qwen3Model for full model tests"""
        model_cpu = Qwen3Model(args).cpu()
        model_cpu.init_weights()
        model_cpu.train()
        model_device = copy.deepcopy(model_cpu).to(self.device)
        model_device.train()
        return model_cpu, model_device
    
    def _setup_layer(self, layer_class, *layer_args):
        """Initializes layer for layer tests"""
        layer_cpu = layer_class(*layer_args).cpu()
        if hasattr(layer_cpu, 'init_weights'):
            if isinstance(layer_cpu, (FeedForward, Attention)):
                layer_cpu.init_weights(init_std=0.02)
            else:
                layer_cpu.init_weights()
        elif isinstance(layer_cpu, nn.Embedding):
            nn.init.normal_(layer_cpu.weight)
            
        layer_cpu.train()
        layer_device = copy.deepcopy(layer_cpu).to(self.device)
        layer_device.train()
        return layer_cpu, layer_device

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
    def test_qwen_full_model_forward_cpu_device_parity(self):
        """
        Tests forward output parity for full Qwen3Model.
        """
        args = self._get_model_args()
        model_cpu, model_device = self._setup_full_model(args)
        
        tokens_cpu = torch.randint(0, args.vocab_size, (self.batch, self.seq_len), device="cpu")
        tokens_device = tokens_cpu.to(self.device)

        with torch.no_grad():
            # Forward pass
            out_cpu = model_cpu(tokens_cpu)
            out_device = model_device(tokens_device)
        
        # Check forward parity
        self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_FWD, rtol=RTOL_FWD))

    def test_qwen_full_model_backward_cpu_device_parity(self):
        """
        Tests backwards gradient parity for the full Qwen3Model.
        """
        args = self._get_model_args()
        model_cpu, model_device = self._setup_full_model(args)
        
        tokens_cpu = torch.randint(0, args.vocab_size, (self.batch, self.seq_len), device="cpu")
        tokens_device = tokens_cpu.to(self.device)
        
        # Forward pass
        out_cpu = model_cpu(tokens_cpu) 
        out_device = model_device(tokens_device)

        # Backward pass
        loss_cpu, loss_device = out_cpu.sum(), out_device.sum()
        # Zero gradients
        optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01) 
        optimizer_device = optim.SGD(model_device.parameters(), lr=0.01)
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
    def test_embedding_forward_cpu_device_parity(self):
        """Tests forward output of Embedding layer."""
        embedding_cpu, embedding_device = self._setup_layer(nn.Embedding, self.args.vocab_size, self.dim)
        tokens_cpu = torch.randint(0, self.args.vocab_size, (self.batch, self.seq_len), device="cpu")
        tokens_device = tokens_cpu.to(self.device)

        with torch.no_grad():
            # Forward pass
            out_cpu = embedding_cpu(tokens_cpu)
            out_device = embedding_device(tokens_device)
        
        # Check forward parity
        self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_EMB, rtol=RTOL_EMB))

    def test_embedding_backward_cpu_device_parity(self):
        """Tests backward gradients of Embedding layer."""
        embedding_cpu, embedding_device = self._setup_layer(nn.Embedding, self.args.vocab_size, self.dim)
        tokens_cpu = torch.randint(0, self.args.vocab_size, (self.batch, self.seq_len), device="cpu")
        tokens_device = tokens_cpu.to(self.device)
        
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

    def test_attention_forward_cpu_device_parity(self):
        """Tests forward output of Attention layer."""
        attention_cpu, attention_device = self._setup_layer(Attention, self.args)
        x_cpu, x_device, rope_cache_cpu, rope_cache_device = self._create_attention_tensors(requires_grad=False)

        with torch.no_grad():
            # Forward pass
            out_cpu = attention_cpu(x_cpu, rope_cache_cpu)
            out_device = attention_device(x_device, rope_cache_device)
        
        # Check forward parity
        self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_FWD, rtol=RTOL_FWD))

    def test_attention_backward_cpu_device_parity(self):
        """Tests backward gradients of Attention layer."""
        attention_cpu, attention_device = self._setup_layer(Attention, self.args)
        x_cpu, x_device, rope_cache_cpu, rope_cache_device = self._create_attention_tensors(requires_grad=True)
        
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
    
    def test_feedforward_forward_cpu_device_parity(self):
        """Tests forward output of FeedForward layer."""
        feedforward_cpu, feedforward_device = self._setup_layer(
            FeedForward, self.args.dim, self.args.hidden_dim
        )
        x_cpu, x_device = self._create_input_tensor((self.batch, self.seq_len, self.dim), requires_grad=False)

        with torch.no_grad():
            # Forward pass
            out_cpu = feedforward_cpu(x_cpu)
            out_device = feedforward_device(x_device)
            
        self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_EMB, rtol=ATOL_EMB))

    def test_feedforward_backward_cpu_device_parity(self):
        """Tests backward gradients of FeedForward layer."""
        feedforward_cpu, feedforward_device = self._setup_layer(
            FeedForward, self.args.dim, self.args.hidden_dim
        )
        x_cpu, x_device = self._create_input_tensor((self.batch, self.seq_len, self.dim), requires_grad=True)

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
    
    def test_transformer_block_forward_cpu_device_parity(self):
        """Tests forward output of TransformerBlock layer."""
        layer_id = 0
        block_cpu, block_device = self._setup_layer(TransformerBlock, layer_id, self.args)
        x_cpu, x_device, rope_cache_cpu, rope_cache_device = self._create_attention_tensors(requires_grad=False)
        
        with torch.no_grad():
            # Forward pass
            out_cpu = block_cpu(x_cpu, rope_cache_cpu)
            out_device = block_device(x_device, rope_cache_device)
        
        # Check forward parity
        self.assertTrue(torch.allclose(out_cpu, out_device.cpu(), atol=ATOL_FWD, rtol=RTOL_FWD))

    def test_transformer_block_backward_cpu_device_parity(self):
        """Tests backward gradients of TransformerBlock layer."""
        layer_id = 0
        block_cpu, block_device = self._setup_layer(TransformerBlock, layer_id, self.args)
        x_cpu, x_device, rope_cache_cpu, rope_cache_device = self._create_attention_tensors(requires_grad=True)
        
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