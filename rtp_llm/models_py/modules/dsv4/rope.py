"""DeepSeek-V4 partial RoPE with YaRN scaling.

Direct port of `inference/model.py:precompute_freqs_cis / apply_rotary_emb`.
V4 applies RoPE only to the LAST `rope_head_dim` dims of each head; the
non-RoPE dims pass through unchanged.

Two RoPE bases per model:
  - rope_theta = 10000          (main, used by SWA-only layers)
  - compress_rope_theta = 160000 (used by CSA/HCA layers' compressed branch)
"""

import math
from typing import Dict, Optional, Tuple

import torch

from rtp_llm.models_py.modules.dsv4.runtime_config import get_switch, parse_bool

# --- DSV4_FUSED_ROPE switch (Task #23) ------------------------------------
# When ON, both apply_rotary_emb and apply_rotary_emb_batched use a single
# fused CUDA kernel (ppu_fused_rope_apply) that replaces the entire
# complex-number pipeline:
#   x.float() -> unflatten -> view_as_complex -> conj -> mul -> view_as_real
#   -> flatten -> copy_
# Default OFF: existing complex path, byte-identical to before.
_DSV4_FUSED_ROPE: bool = get_switch("DSV4_FUSED_ROPE", False, parse_bool)

_ppu_fused_rope_apply = None  # lazy import; only resolved when switch is ON


def _get_ppu_fused_rope_apply():
    global _ppu_fused_rope_apply
    if _ppu_fused_rope_apply is not None:
        return _ppu_fused_rope_apply
    from rtp_llm.ops.compute_ops import rtp_llm_ops
    _ppu_fused_rope_apply = rtp_llm_ops.ppu_fused_rope_apply
    return _ppu_fused_rope_apply


def _fused_rope_inplace(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Apply RoPE in-place via the fused CUDA kernel.

    Handles both ``apply_rotary_emb`` and ``apply_rotary_emb_batched``
    shapes.  ``freqs_cis`` is a complex64 tensor whose leading dims are a
    prefix of x's leading dims; the kernel broadcasts across heads.
    """
    RD = x.size(-1)
    assert RD % 2 == 0
    N = x.numel() // RD
    row_stride = x.stride(-2)

    # flatten freqs to [N_freq, k] and convert to [N_freq, k, 2] float32
    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()
    freqs_flat = freqs_cis.view(-1, freqs_cis.shape[-1])
    N_freq = freqs_flat.shape[0]
    assert N % N_freq == 0, f"N_rows={N} not divisible by N_freq={N_freq}"
    freq_stride_n = N // N_freq
    # view_as_real: [N_freq, k] complex64 -> [N_freq, k, 2] float32
    cos_sin = torch.view_as_real(freqs_flat).contiguous()

    fn = _get_ppu_fused_rope_apply()
    fn(x, cos_sin, freq_stride_n, row_stride, inverse)
    return x

# Process-local memoization keyed by (params, device). All DSV4 compressor
# layers compute identical freqs_cis (they share rope params), so a single
# shared tensor replaces what was 61 distinct CPU + 61 distinct GPU copies
# during model init. Cascade: identical id(freqs_cis) → `_ensure_cos_sin_cache`
# in compressor.py dedupes the derived 256 MiB cos_sin_cache (91× → 1×).
_FREQS_CIS_CACHE: Dict[
    Tuple[int, int, int, float, float, int, int, Optional[str]], torch.Tensor
] = {}


def precompute_freqs_cis(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Returns complex cis tensor `[seqlen, dim/2]`.
    When `original_seq_len > 0`, applies YaRN frequency interpolation.
    When `device` is given, returns the tensor on that device and shares
    it across all callers with the same params+device tuple."""
    key = (
        dim,
        seqlen,
        original_seq_len,
        base,
        factor,
        beta_fast,
        beta_slow,
        str(device) if device is not None else None,
    )
    cached = _FREQS_CIS_CACHE.get(key)
    if cached is not None:
        return cached

    def find_correction_dim(num_rotations, dim_, base_, max_seq_len_):
        return (
            dim_
            * math.log(max_seq_len_ / (num_rotations * 2 * math.pi))
            / (2 * math.log(base_))
        )

    def find_correction_range(low_rot, high_rot, dim_, base_, max_seq_len_):
        low = math.floor(find_correction_dim(low_rot, dim_, base_, max_seq_len_))
        high = math.ceil(find_correction_dim(high_rot, dim_, base_, max_seq_len_))
        return max(low, 0), min(high, dim_ - 1)

    def linear_ramp_factor(min_, max_, dim_):
        if min_ == max_:
            max_ = max_ + 0.001
        linear_func = (torch.arange(dim_, dtype=torch.float32) - min_) / (max_ - min_)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    if device is not None and freqs_cis.device != torch.device(device):
        freqs_cis = freqs_cis.to(device)
    _FREQS_CIS_CACHE[key] = freqs_cis
    return freqs_cis


def apply_rotary_emb_batched(
    x: torch.Tensor, freqs_cis_per_b: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place partial RoPE with PER-REQUEST freqs_cis (Stage 3B).

    Identical to :func:`apply_rotary_emb` except ``freqs_cis_per_b`` is
    ``[B, freqs_dim]`` (one row per batch entry) rather than ``[S, freqs_dim]``
    (broadcast across batch). Used by the CUDA-graph decode path where each
    request has its own ``start_pos`` and therefore its own RoPE row.

    Shapes:
        x                : ``[B, S, ..., 2k]`` (S typically 1 for decode)
        freqs_cis_per_b  : ``[B, k]`` complex64
    """
    if x.numel() == 0 or freqs_cis_per_b.numel() == 0:
        return x
    if _DSV4_FUSED_ROPE:
        return _fused_rope_inplace(x, freqs_cis_per_b, inverse)
    y = x
    B = x.size(0)
    S = x.size(1)
    last = x.size(-1)
    x = torch.view_as_complex(x.float().unflatten(-1, (last // 2, 2)))
    if inverse:
        freqs_cis_per_b = freqs_cis_per_b.conj()
    # freqs_cis_per_b: [B, k] -> broadcastable shape [B, S, ..., k]
    if x.ndim == 3:
        freqs_cis = freqs_cis_per_b.view(B, 1, last // 2).expand(B, S, last // 2)
    else:
        # x.ndim == 4 (B, S, H, k)
        freqs_cis = freqs_cis_per_b.view(B, 1, 1, last // 2).expand(
            B, S, x.size(2), last // 2
        )
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """In-place partial RoPE. x: [..., S, (..., 2k)]; rotates last dim only.
    `inverse=True` applies the conjugate rotation (used on attention output).

    Empty-batch safe: if either ``x`` or the sliced ``freqs_cis`` has zero
    elements (DP rank with no local tokens, or start_pos past max_seq_len
    during warmup), return ``x`` unchanged rather than crashing in ``.view``.
    """
    if x.numel() == 0 or freqs_cis.numel() == 0:
        return x
    if _DSV4_FUSED_ROPE:
        return _fused_rope_inplace(x, freqs_cis, inverse)
    y = x
    # Use explicit size (last_dim // 2, 2) rather than (-1, 2).  Some
    # torch paths (dynamo/fakemode, certain warmup shapes) reject the
    # ``-1`` inferred dim to ``unflatten`` with "unknown parameter type".
    x = torch.view_as_complex(x.float().unflatten(-1, (x.size(-1) // 2, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(x.size(0), x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(x.size(0), x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y
