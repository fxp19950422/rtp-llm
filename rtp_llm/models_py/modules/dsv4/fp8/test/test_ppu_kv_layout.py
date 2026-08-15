"""Tests for the PPU 656-byte FP8 KV pool layout (_ppu_kv_layout).

  * round-trip: pack_ppu_kv_656 -> dequant_ppu_kv_656 ~= input (fp8 floor).
  * real kernel: a pool packed by pack_ppu_kv_656 is read correctly by the PPU
    flash_mla_with_kvcache (matches a bf16 reference), proving the produced
    byte layout is exactly what the kernel expects. Skipped off-PPU.
"""
from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._ppu_kv_layout import (
    PPU_KV_ENTRY_BYTES,
    dequant_ppu_kv_656,
    pack_ppu_kv_656,
)

try:
    import flash_mla  # noqa: F401
    from rtp_llm.models_py.modules.dsv4.fp8.decode import fp8_sparse_attn_decode_op as _dm

    _PPU_REAL = (
        torch.cuda.is_available()
        and _dm._FLASH_MLA_AVAILABLE
        and not _dm._flash_mla_with_kvcache_supports_attn_sink()
    )
except Exception:  # noqa: BLE001
    _PPU_REAL = False


@unittest.skipUnless(torch.cuda.is_available(), "fp8 pack/dequant needs a CUDA device")
class TestPpuKvLayoutRoundTrip(unittest.TestCase):
    def test_pack_dequant_roundtrip(self):
        torch.manual_seed(0)
        k = torch.randn(200, 512, dtype=torch.bfloat16, device="cuda") * 0.2
        pool = pack_ppu_kv_656(k)
        self.assertEqual(pool.shape, (200, PPU_KV_ENTRY_BYTES))
        self.assertEqual(pool.dtype, torch.uint8)
        back = dequant_ppu_kv_656(pool)
        err = ((back.float() - k.float()).norm() / k.float().norm()).item()
        self.assertLess(err, 0.06, f"fp8 round-trip rel err {err}")


@unittest.skipUnless(_PPU_REAL, "requires PPU flash_mla wheel (reduced sig) + CUDA")
class TestPpuKvLayoutRealKernel(unittest.TestCase):
    def test_packed_pool_read_by_kernel(self):
        from flash_mla import flash_mla_with_kvcache, get_mla_metadata

        torch.manual_seed(0)
        dev = "cuda"
        H, T, topk, HEAD, D_KER, DV = 64, 128, 128, 512, 576, 512
        kv = torch.randn(T, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        q = torch.randn(1, 1, H, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        idx = torch.arange(topk, device=dev, dtype=torch.int32).view(1, 1, topk)

        # bf16 reference (value = full 512)
        w = torch.softmax(torch.einsum("bqhd,kd->bqhk", q.float(), kv.float()) * (HEAD ** -0.5), -1)
        o_ref = torch.einsum("bqhk,kd->bqhd", w, kv.float())

        pool = pack_ppu_kv_656(kv).view(1, T, 1, PPU_KV_ENTRY_BYTES)  # uint8
        q_pad = torch.cat([q, torch.zeros(1, 1, H, D_KER - HEAD, dtype=q.dtype, device=dev)], -1)
        cs = torch.full((1,), T, dtype=torch.int32, device=dev)
        bt = torch.zeros(1, 1, dtype=torch.int32, device=dev)
        sm, ns = get_mla_metadata(cs, H, 1, num_heads_q=H, is_fp8_kvcache=True, topk=topk)
        out, _ = flash_mla_with_kvcache(
            q=q_pad, k_cache=pool, block_table=bt, cache_seqlens=cs, head_dim_v=DV,
            tile_scheduler_metadata=sm, num_splits=ns, softmax_scale=HEAD ** -0.5,
            is_fp8_kvcache=True, indices=idx,
        )
        self.assertTrue(torch.isfinite(out).all().item())
        err = ((out.float() - o_ref).norm() / o_ref.norm()).item()
        self.assertLess(err, 0.1, f"packed-pool kernel rel err {err}")


if __name__ == "__main__":
    unittest.main()
