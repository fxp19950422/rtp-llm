"""PPU DSV4 compressor core compute (pure torch).

Reproduces the compress math of the fused triton kernel
``_compressor_vllm_triton._fused_kv_compress_norm_rope_insert_*_attn`` (which
cannot compile on PPU: triton lacks ``fp8e4nv``). Per boundary token:

  1. per-dim softmax over the gathered window positions of ``gate_score`` (the
     caller must have already added the APE table, ``ape[pos % compress_ratio]``);
  2. weighted sum of the window KV -> compressed KV ``[head_dim]``;
  3. RMSNorm over the full head with ``rms_weight``;
  4. interleaved even/odd partial RoPE on the last ``rope_head_dim`` dims
     (``new_even = even*cos - odd*sin``, ``new_odd = odd*cos + even*sin``);

then packs the full head (Mapping A) into the PPU 656-byte FP8 KV slot via
``pack_ppu_kv_656``. The gather (state-pool / raw dispatch) is done by the
caller; this is the numeric core. Verified on ZW810E: RoPE matches a complex-
exponential reference (bf16 floor); softmax/RMSNorm are standard torch; pack is
covered by ``_ppu_kv_layout`` tests.
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.fp8._ppu_kv_layout import pack_ppu_kv_656


def compress_windows_ppu(
    kv_windows: torch.Tensor,     # [T, W, head_dim] bf16 — gathered KV per boundary token
    score_windows: torch.Tensor,  # [T, W, head_dim] fp32 — gate score + APE
    cos: torch.Tensor,            # [T, rope_head_dim // 2] fp32
    sin: torch.Tensor,            # [T, rope_head_dim // 2] fp32
    rms_weight: torch.Tensor,     # [head_dim] bf16
    eps: float,
    rope_head_dim: int,
) -> torch.Tensor:
    """Compress + RMSNorm + partial RoPE -> ``[T, head_dim]`` bf16."""
    T, W, head_dim = kv_windows.shape
    wts = torch.softmax(score_windows.float(), dim=1)            # softmax over W, per dim
    compressed = (kv_windows.float() * wts).sum(dim=1)           # [T, head_dim]
    var = (compressed * compressed).sum(dim=-1, keepdim=True) / head_dim
    normed = compressed * torch.rsqrt(var + eps) * rms_weight.float()  # [T, head_dim]

    nope_dim = head_dim - rope_head_dim
    nope = normed[:, :nope_dim]
    rope = normed[:, nope_dim:]                                  # [T, rope_head_dim]
    even = rope[:, 0::2]                                         # [T, rope//2]
    odd = rope[:, 1::2]
    new_even = even * cos - odd * sin
    new_odd = odd * cos + even * sin
    roped = torch.empty_like(rope)
    roped[:, 0::2] = new_even
    roped[:, 1::2] = new_odd
    return torch.cat([nope, roped], dim=-1).to(torch.bfloat16)   # [T, head_dim]


def compress_and_pack_ppu_656(
    kv_windows: torch.Tensor,
    score_windows: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    rope_head_dim: int,
) -> torch.Tensor:
    """``compress_windows_ppu`` then pack to ``[T, 656]`` uint8 (Mapping A)."""
    normed = compress_windows_ppu(
        kv_windows, score_windows, cos, sin, rms_weight, eps, rope_head_dim
    )
    return pack_ppu_kv_656(normed)
