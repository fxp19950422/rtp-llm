"""Tests for the PPU compressor core compute (_ppu_compressor).

Verifies the compress math (per-dim softmax-over-window pooling + RMSNorm +
interleaved even/odd partial RoPE) against an independent reference:
  * softmax-pool + RMSNorm: standard torch;
  * interleaved RoPE: a complex-exponential rotation reference (cross-check of
    the even/odd formulation used by the fused triton kernel).
"""
from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._ppu_compressor import (
    compress_and_pack_ppu_656,
    compress_windows_ppu,
)
from rtp_llm.models_py.modules.dsv4.fp8._ppu_kv_layout import PPU_KV_ENTRY_BYTES


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device (fp8 pack)")
class TestPpuCompressorCompute(unittest.TestCase):
    def test_compress_matches_complex_exp_rope_reference(self):
        torch.manual_seed(0)
        dev = "cuda"
        T, W, HEAD, ROPE = 5, 8, 512, 64
        NOPE = HEAD - ROPE
        eps = 1e-6
        kv = torch.randn(T, W, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        score = torch.randn(T, W, HEAD, dtype=torch.float32, device=dev) * 0.5
        cos = torch.randn(T, ROPE // 2, device=dev)
        sin = torch.randn(T, ROPE // 2, device=dev)
        rms_w = torch.randn(HEAD, dtype=torch.bfloat16, device=dev) * 0.1 + 1.0

        out = compress_windows_ppu(kv, score, cos, sin, rms_w, eps, ROPE)
        self.assertEqual(tuple(out.shape), (T, HEAD))

        wts = torch.softmax(score.float(), dim=1)
        comp = (kv.float() * wts).sum(1)
        var = (comp * comp).sum(-1, keepdim=True) / HEAD
        nrm = comp * torch.rsqrt(var + eps) * rms_w.float()
        ref = nrm.clone()
        rp = nrm[:, NOPE:]
        c = torch.complex(rp[:, 0::2], rp[:, 1::2])
        rot = c * torch.complex(cos, sin)
        ref_rope = torch.empty_like(rp)
        ref_rope[:, 0::2] = rot.real
        ref_rope[:, 1::2] = rot.imag
        ref[:, NOPE:] = ref_rope
        err = ((out.float() - ref).norm() / ref.norm()).item()
        self.assertLess(err, 5e-3, f"compress compute rel err {err}")  # bf16 floor

    def test_compress_and_pack_shape(self):
        torch.manual_seed(0)
        dev = "cuda"
        T, W, HEAD, ROPE = 3, 8, 512, 64
        kv = torch.randn(T, W, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        score = torch.randn(T, W, HEAD, dtype=torch.float32, device=dev) * 0.5
        cos = torch.randn(T, ROPE // 2, device=dev)
        sin = torch.randn(T, ROPE // 2, device=dev)
        rms_w = torch.ones(HEAD, dtype=torch.bfloat16, device=dev)
        pool = compress_and_pack_ppu_656(kv, score, cos, sin, rms_w, 1e-6, ROPE)
        self.assertEqual(tuple(pool.shape), (T, PPU_KV_ENTRY_BYTES))
        self.assertEqual(pool.dtype, torch.uint8)


if __name__ == "__main__":
    unittest.main()
