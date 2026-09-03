"""Operator-level alignment test: fused RoPE kernel vs original complex path.

Tests that ppu_fused_rope_apply (the C++ CUDA kernel registered in
rtp_llm_ops) produces output numerically equivalent to the original
complex-number RoPE implementation in rope.py.

These tests run on *pure CUDA* (torch.cuda) and do NOT require the full
rtp_llm wheel -- they exercise the kernel via a Python-level reimplementation
of _fused_rope_inplace that directly calls the kernel.

When the kernel is not available (e.g. CPU-only CI), all tests are skipped.
"""
from __future__ import annotations

import os
import sys
import unittest
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Reference: original complex-number RoPE (copied from rope.py verbatim)
# ---------------------------------------------------------------------------

def _eager_apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Reference complex-path RoPE (in-place)."""
    if x.numel() == 0 or freqs_cis.numel() == 0:
        return x
    y = x
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


def _eager_apply_rotary_emb_batched(
    x: torch.Tensor, freqs_cis_per_b: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Reference complex-path batched RoPE (in-place)."""
    if x.numel() == 0 or freqs_cis_per_b.numel() == 0:
        return x
    y = x
    B, S = x.size(0), x.size(1)
    last = x.size(-1)
    x = torch.view_as_complex(x.float().unflatten(-1, (last // 2, 2)))
    if inverse:
        freqs_cis_per_b = freqs_cis_per_b.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis_per_b.view(B, 1, last // 2).expand(B, S, last // 2)
    else:
        freqs_cis = freqs_cis_per_b.view(B, 1, 1, last // 2).expand(
            B, S, x.size(2), last // 2
        )
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


# ---------------------------------------------------------------------------
# Fused kernel wrapper (mirrors _fused_rope_inplace in rope.py)
# ---------------------------------------------------------------------------

def _try_load_kernel():
    """Return ppu_fused_rope_apply or None if unavailable."""
    try:
        from rtp_llm.ops.compute_ops import rtp_llm_ops
        return rtp_llm_ops.ppu_fused_rope_apply
    except Exception:
        pass
    # Fallback: try direct import
    try:
        import librtp_compute_ops.rtp_llm_ops as ops
        return ops.ppu_fused_rope_apply
    except Exception:
        return None


def _fused_rope_call(kernel_fn, x, freqs_cis, inverse=False):
    """Python wrapper that mirrors _fused_rope_inplace."""
    RD = x.size(-1)
    assert RD % 2 == 0
    N = x.numel() // RD
    row_stride = x.stride(-2)

    if not freqs_cis.is_contiguous():
        freqs_cis = freqs_cis.contiguous()
    freqs_flat = freqs_cis.view(-1, freqs_cis.shape[-1])
    N_freq = freqs_flat.shape[0]
    if N == 0 or N_freq == 0:
        return x
    assert N % N_freq == 0
    freq_stride_n = N // N_freq
    cos_sin = torch.view_as_real(freqs_flat).contiguous()

    kernel_fn(x, cos_sin, freq_stride_n, row_stride, inverse)
    return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_freqs(rows: int, rd_half: int, device="cuda") -> torch.Tensor:
    """Random complex64 freqs_cis [rows, rd_half]."""
    angle = torch.rand(rows, rd_half, device=device) * 6.28
    return torch.polar(torch.ones_like(angle), angle).to(torch.complex64).contiguous()


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

_kernel_fn = None


def setUpModule():
    global _kernel_fn
    _kernel_fn = _try_load_kernel()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class PpuFusedRopeAlignmentTest(unittest.TestCase):
    """Verify fused kernel output matches the original complex path."""

    def setUp(self):
        if _kernel_fn is None:
            self.skipTest(
                "ppu_fused_rope_apply not available "
                "(kernel not built or wheel not installed)"
            )

    # -- apply_rotary_emb shapes -------------------------------------------

    def _check_apply(self, shape, rd, inverse=False, dtype=torch.bfloat16):
        torch.manual_seed(42 + sum(shape) + rd + int(inverse))
        base = torch.randn(*shape, dtype=dtype, device="cuda")
        ref = base.clone()
        cand = base.clone()

        # Build freqs_cis [S, rd//2] where S = shape[1] (for 4D) or shape[0] (for 3D)
        if len(shape) == 4:
            S = shape[0] * shape[1]
        else:
            S = shape[0] * shape[1] if len(shape) >= 2 else shape[0]
        freqs = _make_freqs(S, rd // 2)

        _eager_apply_rotary_emb(ref[..., -rd:], freqs, inverse=inverse)
        _fused_rope_call(_kernel_fn, cand[..., -rd:], freqs, inverse=inverse)
        torch.cuda.synchronize()

        torch.testing.assert_close(
            cand, ref,
            rtol=0, atol=3e-2,
            msg=f"Mismatch: shape={shape}, rd={rd}, inverse={inverse}, dtype={dtype}",
        )

    def test_4d_prefill_bf16(self):
        """[B=1, S=128, H=64, head_dim=128], rope_dim=64, BF16."""
        self._check_apply((1, 128, 64, 128), rd=64)

    def test_4d_decode_bf16(self):
        """[B=4, S=1, H=8, head_dim=128], rope_dim=64, BF16."""
        self._check_apply((4, 1, 8, 128), rd=64)

    def test_4d_inverse_bf16(self):
        """Inverse rotation (used on attention output)."""
        self._check_apply((2, 16, 8, 128), rd=64, inverse=True)

    def test_3d_shape_bf16(self):
        """[T=96, H=7, head_dim=128], rope_dim=64, BF16."""
        self._check_apply((96, 7, 128), rd=64)

    def test_4d_fp16(self):
        """FP16 variant."""
        self._check_apply((2, 32, 8, 128), rd=64, dtype=torch.float16)

    def test_4d_fp32(self):
        """FP32 variant -- should be near-exact."""
        torch.manual_seed(99)
        shape = (2, 16, 4, 128)
        rd = 64
        base = torch.randn(*shape, dtype=torch.float32, device="cuda")
        ref = base.clone()
        cand = base.clone()
        freqs = _make_freqs(shape[0] * shape[1], rd // 2)

        _eager_apply_rotary_emb(ref[..., -rd:], freqs)
        _fused_rope_call(_kernel_fn, cand[..., -rd:], freqs)
        torch.cuda.synchronize()

        torch.testing.assert_close(cand, ref, rtol=1e-5, atol=1e-5)

    # -- apply_rotary_emb_batched shapes -----------------------------------

    def test_batched_4d_bf16(self):
        """Batched: [B=4, S=1, H=8, head_dim=128], per-request freqs [B, k]."""
        torch.manual_seed(200)
        B, S, H, D, rd = 4, 1, 8, 128, 64
        base = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        ref = base.clone()
        cand = base.clone()
        freqs = _make_freqs(B, rd // 2)  # [B, k]

        _eager_apply_rotary_emb_batched(ref[..., -rd:], freqs)
        _fused_rope_call(_kernel_fn, cand[..., -rd:], freqs)
        torch.cuda.synchronize()

        torch.testing.assert_close(cand, ref, rtol=0, atol=3e-2)

    def test_batched_4d_inverse(self):
        """Batched + inverse."""
        torch.manual_seed(201)
        B, S, H, D, rd = 3, 2, 16, 128, 64
        base = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        ref = base.clone()
        cand = base.clone()
        freqs = _make_freqs(B, rd // 2)

        _eager_apply_rotary_emb_batched(ref[..., -rd:], freqs, inverse=True)
        _fused_rope_call(_kernel_fn, cand[..., -rd:], freqs, inverse=True)
        torch.cuda.synchronize()

        torch.testing.assert_close(cand, ref, rtol=0, atol=3e-2)

    # -- dtype/shape/inplace checks ----------------------------------------

    def test_inplace_semantics(self):
        """Fused path modifies the input tensor, just like the original."""
        torch.manual_seed(300)
        x = torch.randn(2, 4, 8, 64, dtype=torch.bfloat16, device="cuda")
        orig_ptr = x.data_ptr()
        freqs = _make_freqs(2 * 4, 32)
        _fused_rope_call(_kernel_fn, x, freqs)
        self.assertEqual(x.data_ptr(), orig_ptr, "data_ptr changed -- not in-place!")

    def test_output_dtype_preserved(self):
        """Output dtype must equal input dtype."""
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            x = torch.randn(4, 8, 64, dtype=dtype, device="cuda")
            freqs = _make_freqs(4, 32)
            _fused_rope_call(_kernel_fn, x, freqs)
            self.assertEqual(x.dtype, dtype)

    def test_output_shape_preserved(self):
        """Output shape must equal input shape."""
        shape = (2, 4, 8, 128)
        x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")
        freqs = _make_freqs(2 * 4, 64)
        _fused_rope_call(_kernel_fn, x, freqs)
        self.assertEqual(x.shape, torch.Size(shape))

    def test_empty_noop(self):
        """Empty tensors must be handled gracefully."""
        x = torch.empty(0, 64, dtype=torch.bfloat16, device="cuda")
        freqs = torch.empty(0, 32, dtype=torch.complex64, device="cuda")
        # Should not crash
        _fused_rope_call(_kernel_fn, x, freqs)
        self.assertEqual(x.numel(), 0)

    def test_non_contiguous_tail_view(self):
        """Tail view: x is [B,S,H,head_dim][..., -rd:], stride(-2) > rd."""
        torch.manual_seed(400)
        B, S, H, D, rd = 2, 8, 4, 192, 64
        full = torch.randn(B, S, H, D, dtype=torch.bfloat16, device="cuda")
        ref_full = full.clone()
        cand_full = full.clone()
        freqs = _make_freqs(B * S, rd // 2)

        _eager_apply_rotary_emb(ref_full[..., -rd:], freqs)
        _fused_rope_call(_kernel_fn, cand_full[..., -rd:], freqs)
        torch.cuda.synchronize()

        torch.testing.assert_close(cand_full, ref_full, rtol=0, atol=3e-2)
        # Non-RoPE dims must be untouched
        torch.testing.assert_close(
            cand_full[..., :-rd], ref_full[..., :-rd], rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
