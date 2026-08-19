"""Triton implementation for DeepSeek-V4 Hyper-Connections.

Subclasses the PyTorch fallback so that everything the fused kernels do not
cover -- ``_post_impl``, the huge-prefill chunked path, and any kernel/JIT
failure -- degrades to the exact reference implementation. The launch-bound
pre-mixer tail (RMS + rescale + Sinkhorn) and the head readout move to
Triton; the numerically-sensitive ``fn`` matmuls stay in cuBLAS/torch bf16.

Selected by ``hc/factory.py`` when ``tilelang`` is unavailable but ``triton``
is importable (e.g. PPU), or explicitly via ``DSV4_HC_IMPL=triton``.
"""

from __future__ import annotations

import logging

import torch

from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import (
    FallbackHCHead,
    FallbackHCUnit,
    _hc_fallback_chunk_tokens,
)

logger = logging.getLogger(__name__)

_TRITON_FAIL_LOGGED = False


def _log_fallback_once(where: str, exc: Exception) -> None:
    global _TRITON_FAIL_LOGGED
    if not _TRITON_FAIL_LOGGED:
        logger.warning(
            "DSV4 mHC Triton %s failed (%s); using PyTorch fallback for this "
            "and subsequent calls in this process where the kernel errors.",
            where,
            exc,
        )
        _TRITON_FAIL_LOGGED = True


class TritonHCUnit(FallbackHCUnit):
    def _pre_impl(self, x: torch.Tensor, dbg_tag=None):
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(-2)  # [..., hc*dim]
        # Huge flat prefill: the fallback chunks the fp32 RMS intermediate to
        # bound memory. Defer to it there -- decode (the bottleneck) never hits
        # this, and the Sinkhorn win is a decode-phase win.
        if x_flat.dim() == 2 and x_flat.shape[0] > _hc_fallback_chunk_tokens():
            return super()._pre_impl(x, dbg_tag=dbg_tag)

        try:
            from rtp_llm.models_py.modules.dsv4.hc._mhc_pre_triton import (
                mhc_pre_sinkhorn,
                mhc_rms_mul,
                mhc_readout,
            )

            # ``mixes``: the bf16 ``fn`` matmul stays in cuBLAS/torch, then ONE
            # fused Triton kernel (mhc_rms_mul) does RMS rsqrt + ``* rsqrt`` +
            # ``.float()`` -- replacing the 7-launch torch tail (~600
            # elementwise kernels/step at decode: 43 layers x 2 HC units).
            gemm_out = torch.nn.functional.linear(x_flat, self._fn_bf16())
            mixes = mhc_rms_mul(x_flat, gemm_out, norm_eps=self.norm_eps)

            leading = tuple(int(v) for v in shape[:-2])
            mixes_2d = mixes.reshape(-1, mixes.shape[-1])
            pre, post, comb = mhc_pre_sinkhorn(
                mixes_2d,
                self.scale,
                self.base,
                hc_mult=self.hc_mult,
                eps=self.hc_eps,
                sinkhorn_iters=self.hc_sinkhorn_iters,
            )
            x_view = x.reshape(-1, self.hc_mult, shape[-1])
            y2d = mhc_readout(pre, x_view)

            y = y2d.reshape(*leading, shape[-1]).to(dtype)
            post = post.reshape(*leading, self.hc_mult).unsqueeze(-1)
            comb = comb.reshape(*leading, self.hc_mult, self.hc_mult)
            return y, post, comb
        except Exception as exc:  # pragma: no cover - defensive fallback
            _log_fallback_once("pre", exc)
            return super()._pre_impl(x, dbg_tag=dbg_tag)


class TritonHCHead(FallbackHCHead):
    def _head_impl(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        T = shape[0] if x.dim() == 3 else int(torch.tensor(shape[:-2]).prod().item())
        if x.dim() == 3 and T > _hc_fallback_chunk_tokens():
            return super()._head_impl(x)

        try:
            from rtp_llm.models_py.modules.dsv4.hc._mhc_pre_triton import mhc_head

            # Head runs in fp32 (matches fallback: fp32 x, fp32 ``fn``).
            x_f32 = x.float()
            x_flat = x_f32.flatten(-2)  # [..., hc*dim]
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.norm_eps
            )
            mixes = torch.nn.functional.linear(x_flat, self.fn) * rsqrt  # [..., hc]

            leading = tuple(int(v) for v in shape[:-2])
            mixes_2d = mixes.reshape(-1, self.hc_mult)
            x_view = x_f32.reshape(-1, self.hc_mult, shape[-1])
            y2d = mhc_head(
                mixes_2d,
                self.scale,
                self.base,
                x_view,
                hc_mult=self.hc_mult,
                eps=self.hc_eps,
            )
            return y2d.reshape(*leading, shape[-1]).to(dtype)
        except Exception as exc:  # pragma: no cover - defensive fallback
            _log_fallback_once("head", exc)
            return super()._head_impl(x)
