"""TPU GatedDeltaNet for Qwen3.5: chunked gated delta rule and causal conv1d.

Upstream ``torchtitan.models.qwen3_5.gdn`` imports the CUDA/Triton-only
``attn_gym`` package at top level and runs its non-varlen depthwise causal
``Conv1d`` across the folded ``[T] = [B * S]`` axis as a single stream.

This module provides:
1. ``_install_attn_gym_stub()``: installs a stub ``attn_gym`` module in
   ``sys.modules`` when ``attn_gym`` is unavailable so
   ``torchtitan.models.qwen3_5`` imports cleanly on CPU and TPU. Import this
   module before ``torchtitan.models.qwen3_5``.
2. ``chunked_gated_delta``: pure-PyTorch/XLA chunked gated delta rule matching
   ``attn_gym.linear.chunk_gdn`` / HF ``torch_chunk_gated_delta_rule``.
3. ``TpuGatedDeltaKernel`` and ``TpuInnerGatedDeltaNet``: drop-in ``Module``
   replacements that unflatten ``[T]`` to ``[B, S]`` via
   ``set_unflatten_seq_len`` for both the depthwise causal ``conv1d`` and the
   gated delta rule (``chunked_gated_delta`` or Tokamax's Mosaic kernels), and
   expand grouped Q/K heads to match ``n_value_heads``. Each ``[B, S]`` row is
   one recurrence: documents packed into a row share the convolution and the
   delta-rule state (no varlen ``cu_seqlens``).
"""

from dataclasses import dataclass, field
import importlib.util
import sys
import types

import torch
import torch.nn.functional as F


def _install_attn_gym_stub() -> None:
  """Registers a stub ``attn_gym`` package if ``attn_gym`` is not installed."""
  if "attn_gym" in sys.modules:
    return
  if importlib.util.find_spec("attn_gym") is not None:
    return

  def _stub_op(name: str):
    def _unavailable(*args, **kwargs):
      del args, kwargs
      raise RuntimeError(
          f"attn_gym.linear.{name} is CUDA/Triton-only; use the TPU "
          "TpuInnerGatedDeltaNet / TpuGatedDeltaKernel modules instead."
      )

    return _unavailable

  def _getattr(name: str):
    if name.startswith("__"):
      raise AttributeError(name)
    return _stub_op(name)

  for mod_name in ("attn_gym", "attn_gym.linear"):
    module = types.ModuleType(mod_name)
    module.__path__ = []  # type: ignore[attr-defined]
    module.__getattr__ = _getattr  # type: ignore[method-assign]
    sys.modules[mod_name] = module


_install_attn_gym_stub()

# pylint: disable=g-import-not-at-top,g-bad-import-order
from torchtitan.experiments.tpu.unflatten_shim import get_unflatten_seq_len
from torchtitan.models.qwen3_5.gdn import (
    GatedDeltaKernel,
    InnerGatedDeltaNet,
)

# pylint: enable=g-import-not-at-top,g-bad-import-order


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def resolve_batch_and_seq_len(num_tokens: int) -> tuple[int, int]:
  """Resolves ``(batch_size, seq_len)`` from ``get_unflatten_seq_len()``."""
  seq_len = get_unflatten_seq_len()
  if seq_len is None or num_tokens <= seq_len:
    return 1, num_tokens
  if num_tokens % seq_len != 0:
    raise ValueError(
        f"num_tokens ({num_tokens}) must be divisible by seq_len ({seq_len})"
    )
  return num_tokens // seq_len, seq_len


def _causal_conv(x_BSC: torch.Tensor, weight_C1W: torch.Tensor) -> torch.Tensor:
  """Depthwise causal ``conv1d`` on ``[B, S, C]`` via shifted slices.

  ``y[s] = sum_k w[k] * x[s + k - (W - 1)]``, with zeros before the sequence.
  Slices of the front-padded input avoid transposing ``[B, S, C]`` to
  ``[B, C, S]`` and torch_tpu's grouped 1D ``convolution_backward`` shape bug
  under GraphTrainer.

  Args:
    x_BSC: ``[B, S, C]`` input.
    weight_C1W: ``[C, 1, W]`` depthwise ``Conv1d`` weight.

  Returns:
    ``[B, S, C]`` convolution output.
  """
  seq_len = x_BSC.shape[1]
  kernel_size = weight_C1W.shape[-1]
  x_pad = F.pad(x_BSC, (0, 0, kernel_size - 1, 0))
  out_BSC = x_pad[:, :seq_len, :] * weight_C1W[:, 0, 0]
  for k in range(1, kernel_size):
    out_BSC = out_BSC + x_pad[:, k : k + seq_len, :] * weight_C1W[:, 0, k]
  return out_BSC


class _CausalConvSilu(torch.autograd.Function):
  """``silu(_causal_conv(x, w))`` with a backward built from slices too.

  Autograd of ``_causal_conv`` accumulates the ``W`` shifted input gradients
  into the padded buffer with in-place adds at sublane-misaligned sequence
  offsets, which XLA emits as ``dynamic-update-slice`` adds at ~0.19 TB/s on
  TPU7x (12-layer 35B-A3B proxy at 32K tokens/rank: 5.4% of device time). The
  input gradient of a causal convolution is an anti-causal convolution of the
  output gradient, ``dx[t] = sum_k w[k] * dy[t + W - 1 - k]``, so it is computed
  from shifted slices of the end-padded output gradient instead, like the
  forward. The convolution output is recomputed for ``silu``'s derivative.
  """

  @staticmethod
  def forward(ctx, x_BSC: torch.Tensor, weight_C1W: torch.Tensor):
    ctx.save_for_backward(x_BSC, weight_C1W)
    return F.silu(_causal_conv(x_BSC, weight_C1W))

  @staticmethod
  def backward(ctx, grad_BSC: torch.Tensor):
    x_BSC, weight_C1W = ctx.saved_tensors
    seq_len = x_BSC.shape[1]
    kernel_size = weight_C1W.shape[-1]
    grad_y_BSC = torch.ops.aten.silu_backward(
        grad_BSC, _causal_conv(x_BSC, weight_C1W)
    )
    grad_pad = F.pad(grad_y_BSC, (0, 0, 0, kernel_size - 1))
    grad_x_BSC = None
    grad_w = []
    for k in range(kernel_size):
      shift = kernel_size - 1 - k
      term = grad_pad[:, shift : shift + seq_len, :] * weight_C1W[:, 0, k]
      grad_x_BSC = term if grad_x_BSC is None else grad_x_BSC + term
      if shift == 0:
        grad_w.append((x_BSC * grad_y_BSC).sum(dim=(0, 1)))
      elif shift < seq_len:
        grad_w.append(
            (x_BSC[:, : seq_len - shift, :] * grad_y_BSC[:, shift:, :]).sum(
                dim=(0, 1)
            )
        )
      else:
        grad_w.append(grad_y_BSC.new_zeros(weight_C1W.shape[0]))
    grad_w_C1W = torch.stack(grad_w, dim=-1).unsqueeze(1)
    return grad_x_BSC, grad_w_C1W.to(weight_C1W.dtype)


def causal_conv_silu(
    x_BSC: torch.Tensor, weight_C1W: torch.Tensor
) -> torch.Tensor:
  """``silu`` of a depthwise causal ``conv1d`` over ``[B, S, C]``."""
  return _CausalConvSilu.apply(x_BSC, weight_C1W)


def _matmul_bf16x3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Matmul of fp32 operands with near-fp32 accuracy from three bf16 passes.

  torch_tpu runs fp32 matmuls at ``Precision.DEFAULT`` (one bf16 MXU pass) by
  default. Splitting each operand into a bf16-exact high part and a residual
  keeps ``hi @ hi + hi @ lo + lo @ hi`` accurate to about 2^-16 regardless of
  the matmul precision setting.
  """
  a_hi = a.to(torch.bfloat16).float()
  b_hi = b.to(torch.bfloat16).float()
  return a_hi @ b_hi + (a_hi @ (b - b_hi) + (a - a_hi) @ b_hi)


def _unit_lower_triangular_inverse(strict_lower: torch.Tensor) -> torch.Tensor:
  """Returns ``(I + strict_lower)^-1`` for ``[..., C, C]`` with C a power of 2.

  Recursive doubling over diagonal blocks: a unit lower-triangular matrix split
  as ``[[A, 0], [L, D]]`` has inverse ``[[A^-1, 0], [-D^-1 L A^-1, D^-1]]``.
  With ``T_s`` the block-diagonal matrix of the inverses of the ``s x s``
  diagonal blocks and ``L_s`` the lower-left ``s x s`` quadrants of the
  ``2s x 2s`` diagonal blocks of ``strict_lower``, this reads
  ``T_2s = T_s - T_s L_s T_s``. Starting from ``T_2 = I - L_1``,
  ``log2(C) - 1`` levels of two batched (bf16x3) ``C x C`` matmuls build the
  inverse. Unlike a Neumann series, every intermediate is an entry of the
  inverse itself, so there is no growth and cancellation.

  Every intermediate stays ``[..., C, C]`` (masked, mostly-zero blocks are
  multiplied too), so no ``s x s`` block is a trailing dimension that TPU would
  pad to its ``(8, 128)`` tiles. On TPU7x this measured the same as a blockwise
  ``[..., C/s, s, s]`` version (49 vs 50 ms forward + backward for 4 x 2048
  tokens with 32 heads and 128-token chunks): both are bound by HBM traffic of
  the ``[C, C]`` FP32 intermediates, which is what the ``tokamax`` backend of
  ``TpuGatedDeltaKernel`` avoids.

  Args:
    strict_lower: ``[..., C, C]`` strictly lower-triangular matrices (the
      diagonal and upper triangle must be zero).

  Returns:
    ``[..., C, C]`` inverses of ``I + strict_lower``.
  """
  c = strict_lower.shape[-1]
  if c == 1:
    return torch.ones_like(strict_lower)
  idx = torch.arange(c, device=strict_lower.device)
  row, col = idx.unsqueeze(-1), idx
  inv = torch.eye(c, dtype=strict_lower.dtype, device=strict_lower.device)
  s = 1
  while s < c:
    # Lower-left s x s quadrant of each diagonal 2s x 2s block.
    quadrant = (
        (row // (2 * s) == col // (2 * s))
        & (row % (2 * s) >= s)
        & (col % (2 * s) < s)
    )
    lower_left = strict_lower * quadrant
    if s == 1:
      # T_1 = I, so T_2 = I - L_1.
      inv = inv - lower_left
    else:
      inv = inv - _matmul_bf16x3(inv, _matmul_bf16x3(lower_left, inv))
    s *= 2
  return inv


def chunked_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
  """Chunked gated delta rule in pure PyTorch/XLA (FLA ``chunk_gdn`` math).

  Work within each chunk is expressed as batched matrix multiplications over
  all chunks at once (including the UT transform's unit lower-triangular
  inverse, see ``_unit_lower_triangular_inverse``); only the inter-chunk state
  recurrence is sequential over ``seq_len / chunk_size`` steps, and each step
  is three matmuls. Applies Q/K L2 normalization and ``1 / sqrt(key_head_dim)``
  query scaling in fp32.

  Args:
    q: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized queries.
    k: ``(bs, seq_len, n_heads, key_head_dim)`` un-normalized keys.
    v: ``(bs, seq_len, n_heads, value_head_dim)`` values.
    g: ``(bs, seq_len, n_heads)`` log-space decay (negative).
    beta: ``(bs, seq_len, n_heads)`` update gate in ``(0, 1)``.
    chunk_size: Tokens per chunk, a power of 2.

  Returns:
    ``(bs, seq_len, n_heads, value_head_dim)`` output in ``q.dtype``.
  """
  out_dtype = q.dtype
  d_k = q.shape[-1]
  q = _l2norm(q.float()) * (d_k**-0.5)
  k = _l2norm(k.float())
  q, k, v = [x.transpose(1, 2).float() for x in (q, k, v)]
  g, beta = [x.transpose(1, 2).float() for x in (g, beta)]

  bs, n_heads, seq_len, _ = k.shape
  d_v = v.shape[-1]
  chunk = chunk_size
  if 0 < seq_len < chunk_size:
    # Short sequences use one chunk, rounded up to a power of 2.
    chunk = 1 << (seq_len - 1).bit_length()
  if chunk & (chunk - 1):
    raise ValueError(f"chunk_size must be a power of 2: {chunk}")
  pad = (chunk - seq_len % chunk) % chunk
  if pad:
    # Padded tokens have beta = 0 (no state update) and g = 0 (no decay).
    q, k, v = [F.pad(x, (0, 0, 0, pad)) for x in (q, k, v)]
    g, beta = [F.pad(x, (0, pad)) for x in (g, beta)]
  num_chunks = (seq_len + pad) // chunk

  q, k, v = [
      x.reshape(bs, n_heads, num_chunks, chunk, x.shape[-1]) for x in (q, k, v)
  ]
  beta = beta.reshape(bs, n_heads, num_chunks, chunk, 1)
  g = g.reshape(bs, n_heads, num_chunks, chunk).cumsum(dim=-1)

  # decay[i, j] = exp(g_i - g_j) for i >= j and 0 otherwise. Masking before
  # exp keeps the positive upper-triangle exponents from overflowing.
  decay = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().tril()

  # UT transform for all chunks at once:
  #   [W | U] = T [diag(beta) diag(exp(g)) K | diag(beta) V],
  #   T = (I + tril_{-1}(diag(beta) (K K^T * decay)))^-1.
  k_beta = k * beta
  a = (k_beta @ k.transpose(-1, -2) * decay).tril(-1)
  rhs = torch.cat([k_beta * g.exp().unsqueeze(-1), v * beta], dim=-1)
  wu = _unit_lower_triangular_inverse(a) @ rhs
  w, u = wu[..., :d_k], wu[..., d_k:]

  # State-independent per-chunk terms, batched over all chunks.
  qk = q @ k.transpose(-1, -2) * decay  # Causal, including the diagonal.
  g_last = g[..., -1:]
  # Rows [:chunk] give W S (the state's prediction of each value) and rows
  # [chunk:] give exp(g) Q S (the inter-chunk output), in one matmul.
  wq = torch.cat([w, q * g.exp().unsqueeze(-1)], dim=-2)
  k_decayed_t = (k * (g_last - g).exp().unsqueeze(-1)).transpose(-1, -2)
  chunk_decay = g_last.exp().unsqueeze(-1)

  state = torch.zeros(bs, n_heads, d_k, d_v, dtype=q.dtype, device=q.device)
  outs = []
  for i in range(num_chunks):
    wq_state = wq[:, :, i] @ state
    v_new = u[:, :, i] - wq_state[..., :chunk, :]
    outs.append(wq_state[..., chunk:, :] + qk[:, :, i] @ v_new)
    state = state * chunk_decay[:, :, i] + k_decayed_t[:, :, i] @ v_new

  out = torch.stack(outs, dim=2).reshape(bs, n_heads, -1, d_v)[:, :, :seq_len]
  return out.transpose(1, 2).contiguous().to(out_dtype)


class TpuGatedDeltaKernel(GatedDeltaKernel):
  """TPU-compatible ``GatedDeltaKernel`` using ``chunked_gated_delta``."""

  @dataclass(kw_only=True, slots=True)
  class Config(GatedDeltaKernel.Config):
    # "xla": `chunked_gated_delta` (pure PyTorch, any device). "tokamax":
    # Tokamax's Mosaic KDA kernels on TPU (`tokamax_gated_delta`), which keep
    # each chunk in VMEM; `chunk_size` then does not apply.
    backend: str = "xla"
    # "xla" only. On TPU7x (4 x 2048 tokens, 32 heads x 128 dims), compiled
    # forward + backward took 65 ms with 64-token chunks, 49 ms with 128, and
    # 82 ms with 256.
    chunk_size: int = 128
    # "tokamax" only: per-token log decays below this are clamped to it (see
    # `tokamax_gated_delta.DEFAULT_MIN_LOG_DECAY`).
    tokamax_min_log_decay: float = -5.0

  def __init__(self, config: Config):
    super().__init__(config)
    if config.backend not in ("xla", "tokamax"):
      raise ValueError(f"unknown TpuGatedDeltaKernel backend: {config.backend}")
    self.backend = config.backend
    self.chunk_size = config.chunk_size
    self.tokamax_min_log_decay = config.tokamax_min_log_decay

  def forward(
      self,
      xq_THK: torch.Tensor,
      xk_THK: torch.Tensor,
      xv_THV: torch.Tensor,
      g_TH: torch.Tensor,
      beta_TH: torch.Tensor,
      *,
      cu_seqlens: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if cu_seqlens is not None:
      raise NotImplementedError(
          "TpuGatedDeltaKernel does not support varlen cu_seqlens."
      )
    repeats = xv_THV.shape[1] // xq_THK.shape[1]
    if repeats > 1:
      xq_THK = xq_THK.repeat_interleave(repeats, dim=1)
      xk_THK = xk_THK.repeat_interleave(repeats, dim=1)
    num_tokens = xq_THK.shape[0]
    batch_size, seq_len = resolve_batch_and_seq_len(num_tokens)
    inputs = (
        xq_THK.reshape(batch_size, seq_len, *xq_THK.shape[1:]),
        xk_THK.reshape(batch_size, seq_len, *xk_THK.shape[1:]),
        xv_THV.reshape(batch_size, seq_len, *xv_THV.shape[1:]),
        g_TH.reshape(batch_size, seq_len, *g_TH.shape[1:]),
        beta_TH.reshape(batch_size, seq_len, *beta_TH.shape[1:]),
    )
    if self.backend == "tokamax":
      # pylint: disable-next=g-import-not-at-top
      from torchtitan.experiments.tpu.qwen3_5 import tokamax_gated_delta

      out_BSHV = tokamax_gated_delta.tokamax_gated_delta(
          *inputs, min_log_decay=self.tokamax_min_log_decay
      )
    else:
      out_BSHV = chunked_gated_delta(*inputs, chunk_size=self.chunk_size)
    return out_BSHV.reshape(num_tokens, *out_BSHV.shape[2:])


class TpuInnerGatedDeltaNet(InnerGatedDeltaNet):
  """TPU-compatible ``InnerGatedDeltaNet`` isolating ``[B, S]`` conv streams."""

  @dataclass(kw_only=True, slots=True)
  class Config(InnerGatedDeltaNet.Config):
    kernel: GatedDeltaKernel.Config = field(
        default_factory=TpuGatedDeltaKernel.Config
    )

  def forward(
      self,
      query_TC: torch.Tensor,
      key_TC: torch.Tensor,
      value_TC: torch.Tensor,
      a_TH: torch.Tensor,
      b_TH: torch.Tensor,
      conv_q_weight_C1W: torch.Tensor,
      conv_k_weight_C1W: torch.Tensor,
      conv_v_weight_C1W: torch.Tensor,
      A_log_H: torch.Tensor,
      dt_bias_H: torch.Tensor,
      cu_seqlens: torch.Tensor,
      *,
      key_head_dim: int,
      value_head_dim: int,
  ) -> torch.Tensor:
    del cu_seqlens  # Fixed-shape [B, S] unflattening via set_unflatten_seq_len.
    num_tokens = query_TC.shape[0]
    batch_size, seq_len = resolve_batch_and_seq_len(num_tokens)

    def causal_conv(
        x_TC: torch.Tensor,
        weight_C1W: torch.Tensor,
    ) -> torch.Tensor:
      channels = weight_C1W.shape[0]
      # Each [B, S] stream is convolved on its own (see `causal_conv_silu`).
      return causal_conv_silu(
          x_TC.reshape(batch_size, seq_len, channels), weight_C1W
      ).reshape(num_tokens, channels)

    xq_THK = causal_conv(query_TC, conv_q_weight_C1W).reshape(
        num_tokens, -1, key_head_dim
    )
    xk_THK = causal_conv(key_TC, conv_k_weight_C1W).reshape(
        num_tokens, -1, key_head_dim
    )
    xv_THV = causal_conv(value_TC, conv_v_weight_C1W).reshape(
        num_tokens, -1, value_head_dim
    )
    g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
    beta_TH = torch.sigmoid(b_TH)
    return self.kernel(
        xq_THK,
        xk_THK,
        xv_THV,
        g_TH,
        beta_TH,
        cu_seqlens=None,
    )
