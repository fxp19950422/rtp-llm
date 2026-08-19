"""Shared W8A8-INT8 dense GEMM wiring for DeepSeek-V4 (PPU).

Bridges per-channel INT8 weights to a *true* W8A8 matmul: quantize the
BF16 activation to per-token INT8, then run DeepGEMM's dense
``gemm_int8_int8_bf16_nt`` (PPU wheel) instead of dequantizing the whole
weight to BF16 on every forward.

Why: the dequant path (``w.to(fp32) * scale -> bf16`` then ``F.linear``)
touches ~19x the weight bytes per call (int8->fp32 write, mul read/write,
fp32->bf16 read/write) while a real INT8 GEMM reads the weight once. On the
wr5 timeline that dequant is 62% of decode GPU time versus 6.5% for the
actual matmul.

Numeric contract: the activation goes BF16 -> per-token INT8, a lossy step
the dequant path does not have (it keeps activations in BF16 and only
dequantizes the weight exactly). This is the standard W8A8 contract shared
with the routed-expert masked-GEMM path in
``moe/strategies/deepep_low_latency.py``. Callers MUST gate on
:func:`has_int8_dense_gemm` and keep the dequant fallback for platforms
without the PPU int8 kernels -- the CUDA/ROCm DeepGEMM wheels do not export
the ``gemm_int8_int8_bf16_nt`` family.

Layout (mirrors ``deepgemm_warmup.warmup_dense_int8_gemm``):
    lhs = (x_i8   [M, K] int8, x_scale [M, 1] fp32)   -- per-token
    rhs = (w_i8   [N, K] int8, w_scale [N, 1] fp32)   -- per-out-channel
    out =          [M, N] bf16                          -- x @ w^T dequantized
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import torch

# Cached capability probe: True/False once resolved, None before first probe.
_PROBED_CAPABILITY: Optional[bool] = None


def _import_ppu_int8() -> Tuple[Callable[[], bool], Callable[..., Any], Callable[..., Any]]:
    """Lazily import the PPU INT8 kernels.

    They live under ``internal_source`` and are absent from CUDA/ROCm
    wheels, so the import is deferred and mirrors the internal-import
    pattern used by ``moe/strategies/deepep_low_latency.py``.
    """
    from internal_source.rtp_llm.models_py.kernels.ppu.deepgemm_wrapper import (
        has_deep_gemm_int8_dense,
        int8_gemm_nt,
    )
    from internal_source.rtp_llm.models_py.kernels.ppu.int8_quant import (
        per_token_quant_int8,
    )

    return has_deep_gemm_int8_dense, int8_gemm_nt, per_token_quant_int8


def has_int8_dense_gemm() -> bool:
    """Whether a true per-token W8A8 INT8 dense GEMM is available.

    Result is cached: a missing kernel (non-PPU wheel) resolves to ``False``
    once and never retries the import.
    """
    global _PROBED_CAPABILITY
    if _PROBED_CAPABILITY is not None:
        return _PROBED_CAPABILITY
    try:
        has_dense, _, _ = _import_ppu_int8()
        _PROBED_CAPABILITY = bool(has_dense())
    except Exception:
        _PROBED_CAPABILITY = False
    return _PROBED_CAPABILITY


def quantize_per_token_int8(x_2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D BF16 activation to per-token INT8.

    Returns ``(x_i8 [M, K] int8, x_scale [M, 1] fp32)`` -- the ``lhs`` layout
    expected by :func:`int8_dense_gemm`.
    """
    _, _, per_token_quant_int8 = _import_ppu_int8()
    return per_token_quant_int8(x_2d)


def int8_dense_gemm(
    x_i8: torch.Tensor,
    x_scale: torch.Tensor,
    w_i8: torch.Tensor,
    w_scale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run ``out = (x_i8 @ w_i8^T) * x_scale * w_scale`` in BF16.

    ``w_i8`` is ``[N, K]`` int8 with per-out-channel ``w_scale`` ``[N, 1]``
    fp32; ``x_i8`` is ``[M, K]`` int8 with per-token ``x_scale`` ``[M, 1]``
    fp32. Allocates a ``[M, N]`` BF16 output when ``out`` is None.
    """
    _, int8_gemm_nt, _ = _import_ppu_int8()
    M = x_i8.shape[0]
    N = w_i8.shape[0]
    if out is None:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=x_i8.device)
    int8_gemm_nt((x_i8, x_scale), (w_i8, w_scale), out)
    return out
