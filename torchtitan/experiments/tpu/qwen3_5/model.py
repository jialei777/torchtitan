"""Qwen3.5 attention and model modules for TPU.

Upstream ``Qwen35Attention.forward`` does not pass ``positions`` to its
``inner_attention``, so neither SDPA nor Splash / Tokamax attention sees
packed-document boundaries. This module provides:
1. ``PackedScaledDotProductInnerAttention`` and ``TpuQwen35Attention``: forward
   ``positions`` to the inner attention and apply same-document causal masking
   on packed sequences.
2. ``TpuQwen35Model``: the Qwen3.5 model accepting ``positions``, initializing
   SimpleFSDP-sharded parameters in place.
"""

from dataclasses import dataclass

import torch
from torch.nn.attention import sdpa_kernel
import torch.nn.functional as F
from torchtitan.experiments.graph_trainer.simple_fsdp import disable_active_parametrization

# Installs the `attn_gym` stub before `torchtitan.models.qwen3_5` is imported.
from torchtitan.experiments.tpu.qwen3_5 import gated_delta
from torchtitan.models.common.attention import (
    AttentionMasksType,
    ScaledDotProductInnerAttention,
    segment_ids_from_positions,
)
from torchtitan.models.qwen3_5.model import Qwen35Attention, Qwen35Model


def packed_causal_mask(
    seq_len: int,
    device: torch.device,
    *,
    segment_ids_BL: torch.Tensor | None = None,
) -> torch.Tensor:
  """Causal mask restricted to each packed document within a row."""
  q_idx = torch.arange(seq_len, device=device).unsqueeze(1)
  kv_idx = torch.arange(seq_len, device=device).unsqueeze(0)
  causal_LL = q_idx >= kv_idx
  if segment_ids_BL is None:
    return causal_LL
  same_doc_BLL = segment_ids_BL.unsqueeze(2) == segment_ids_BL.unsqueeze(1)
  return (causal_LL.unsqueeze(0) & same_doc_BLL).unsqueeze(1)


class PackedScaledDotProductInnerAttention(ScaledDotProductInnerAttention):
  """Document-masked causal SDPA compatible with Splash / Tokamax patches."""

  @dataclass(kw_only=True, slots=True)
  class Config(ScaledDotProductInnerAttention.Config):
    pass

  def forward(
      self,
      q_BLHK: torch.Tensor,
      k_BLHK: torch.Tensor,
      v_BLHV: torch.Tensor,
      *,
      attention_masks: AttentionMasksType | None = None,
      scale: float | None = None,
      enable_gqa: bool = False,
      is_causal: bool = True,
      positions: torch.Tensor | None = None,
      segment_ids: torch.Tensor | None = None,
      **kwargs,
  ) -> torch.Tensor:
    if segment_ids is None and positions is not None:
      pos_1d = positions[:, 0] if positions.ndim == 2 else positions
      segment_ids = segment_ids_from_positions(pos_1d)
    if segment_ids is None:
      return super().forward(
          q_BLHK,
          k_BLHK,
          v_BLHV,
          attention_masks=attention_masks,
          scale=scale,
          enable_gqa=enable_gqa,
          is_causal=is_causal,
          **kwargs,
      )
    if attention_masks is not None:
      raise ValueError(
          "PackedScaledDotProductInnerAttention does not support runtime "
          "attention_masks."
      )
    if not is_causal:
      raise ValueError("Packed-document SDPA attention is always causal.")

    was_3d = q_BLHK.ndim == 3
    if was_3d:
      batch_size, seq_len = gated_delta.resolve_batch_and_seq_len(
          q_BLHK.shape[0]
      )
      q_BLHK = q_BLHK.reshape(batch_size, seq_len, *q_BLHK.shape[1:])
      k_BLHK = k_BLHK.reshape(batch_size, seq_len, *k_BLHK.shape[1:])
      v_BLHV = v_BLHV.reshape(batch_size, seq_len, *v_BLHV.shape[1:])
    else:
      batch_size, seq_len = q_BLHK.shape[:2]

    q_BHLK, k_BHLK, v_BHLV = (
        q_BLHK.transpose(1, 2),
        k_BLHK.transpose(1, 2),
        v_BLHV.transpose(1, 2),
    )
    mask = packed_causal_mask(
        seq_len,
        q_BHLK.device,
        segment_ids_BL=segment_ids.reshape(batch_size, seq_len),
    )
    with sdpa_kernel(self.sdpa_backends, set_priority=True):
      out_BHLV = F.scaled_dot_product_attention(
          q_BHLK,
          k_BHLK,
          v_BHLV,
          attn_mask=mask,
          scale=scale,
          enable_gqa=enable_gqa,
      )
    out_BLHV = out_BHLV.transpose(1, 2)
    if was_3d:
      return out_BLHV.reshape(-1, *out_BLHV.shape[2:])
    return out_BLHV


class TpuQwen35Attention(Qwen35Attention):
  """Qwen35Attention that forwards ``positions`` to ``inner_attention``.

  ``forward`` is upstream ``Qwen35Attention.forward`` plus ``positions`` for the
  inner attention, and default per-row positions when the token stream holds
  more than one ``[B, S]`` row.
  """

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35Attention.Config):
    pass

  def forward(
      self,
      x_TD: torch.Tensor,
      attention_masks: AttentionMasksType | None,
      positions: torch.Tensor | None = None,
  ) -> torch.Tensor:
    num_tokens = x_TD.shape[0]
    if positions is None:
      batch_size, seq_len = gated_delta.resolve_batch_and_seq_len(num_tokens)
      if batch_size > 1:
        positions = torch.arange(
            seq_len, dtype=torch.long, device=x_TD.device
        ).repeat(batch_size)

    xq_gate_THC = self.wq(x_TD).view(num_tokens, -1, self.head_dim * 2)
    xq_THK, gate_THV = xq_gate_THC.chunk(2, dim=-1)
    xk_THK = self.wk(x_TD).view(num_tokens, -1, self.head_dim)
    xv_THV = self.wv(x_TD).view(num_tokens, -1, self.head_dim)

    xq_THK = self.q_norm(xq_THK)
    xk_THK = self.k_norm(xk_THK)

    assert self.rotary_dim <= self.head_dim
    xq_THR, xq_THP = (
        xq_THK[..., : self.rotary_dim],
        xq_THK[..., self.rotary_dim :],
    )
    xk_THR, xk_THP = (
        xk_THK[..., : self.rotary_dim],
        xk_THK[..., self.rotary_dim :],
    )
    xq_THR, xk_THR = self.rope(xq_THR, xk_THR, positions)
    xq_THK = torch.cat([xq_THR, xq_THP], dim=-1)
    xk_THK = torch.cat([xk_THR, xk_THP], dim=-1)

    out_THV = self.inner_attention(
        xq_THK,
        xk_THK,
        xv_THV,
        attention_masks=attention_masks,
        scale=self.scaling,
        enable_gqa=self.enable_gqa,
        positions=positions,
    ).contiguous()

    out_THV = out_THV * torch.sigmoid(gate_THV)
    out_TD = out_THV.view(num_tokens, -1)
    return self.wo(out_TD)


class TpuQwen35Model(Qwen35Model):
  """Qwen35Model accepting ``positions`` positionally or by keyword."""

  @dataclass(kw_only=True, slots=True)
  class Config(Qwen35Model.Config):
    pass

  def init_states(
      self,
      *,
      buffer_device: torch.device | None = None,
  ) -> None:
    # Under SimpleFSDP, parameter reads return all-gathered copies;
    # initialize the sharded parameters themselves.
    with disable_active_parametrization():
      super().init_states(buffer_device=buffer_device)

  def forward(  # pyrefly: ignore[bad-override]
      self,
      tokens: torch.Tensor,
      positions: torch.Tensor | None = None,
      attention_masks: AttentionMasksType | None = None,
      **kwargs,
  ) -> torch.Tensor:
    return super().forward(
        tokens,
        attention_masks=attention_masks,  # pyrefly: ignore[bad-argument-type]
        positions=positions,
        **kwargs,
    )
