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



@unittest.skipUnless(_PPU_REAL, "requires PPU flash_mla wheel (reduced sig) + CUDA")
class TestSwaInsert656WriteReadCycle(unittest.TestCase):
    """SWA-insert 656 branch -> decode op read (real kernel) == bf16 ref.

    The GPU 584-byte triton insert kernel cannot compile on PPU (triton lacks
    fp8e4nv), so the PPU path packs per-token in pure torch. This exercises the
    full write->read cycle: quantize_and_insert_k_cache into a multi-block 656
    pool, then SparseAttnV4DecodeFp8Op.forward through the real kernel.
    """

    def test_insert_then_decode(self):
        from rtp_llm.models_py.modules.dsv4.fp8._swa_kv_insert_triton import (
            quantize_and_insert_k_cache,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
            SparseAttnV4DecodeFp8Op,
        )

        torch.manual_seed(0)
        dev = "cuda"
        H, HEAD, topk = 64, 512, 128
        num_blocks, block_size = 2, 64
        N = num_blocks * block_size
        k = torch.randn(N, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        pool = torch.zeros(num_blocks, block_size, PPU_KV_ENTRY_BYTES, dtype=torch.uint8, device=dev)
        slot_mapping = torch.arange(N, dtype=torch.int64, device=dev)
        quantize_and_insert_k_cache(k, pool, slot_mapping)  # -> 656 pure-torch branch
        self.assertEqual(int(pool.reshape(N, -1).any(-1).sum()), N)

        q = torch.randn(1, 1, H, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        sink = torch.rand(H, dtype=torch.float32, device=dev) * 0.5 + 0.2
        idx = torch.randint(0, N, (1, 1, topk), dtype=torch.int32, device=dev)
        op = SparseAttnV4DecodeFp8Op(n_heads=H, head_dim=HEAD, softmax_scale=HEAD ** -0.5)
        out = op.forward(q, pool, sink, idx, sched_meta=None)
        self.assertEqual(tuple(out.shape), (1, 1, H, HEAD))
        self.assertTrue(torch.isfinite(out).all().item())

        sel = k[idx[0, 0].long()].float()
        sc = torch.einsum("hd,kd->hk", q[0, 0].float(), sel) * (HEAD ** -0.5)
        m = sc.amax(-1, keepdim=True)
        e = torch.exp(sc - m)
        denom = e.sum(-1, keepdim=True) + torch.exp(sink.view(H, 1) - m)
        oref = torch.einsum("hk,kd->hd", e, sel) / denom
        err = ((out[0, 0].float() - oref).norm() / oref.norm()).item()
        self.assertLess(err, 0.1, f"insert->decode rel err {err}")



@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device (triton import + fp8)")
class TestSwaDequantGather656(unittest.TestCase):
    """SWA dequant+gather 656 PPU branch round-trip: pack K -> 656 pool ->
    dequantize_and_gather_k_cache (pure-torch PPU path) -> ~= K (fp8 floor).
    The GPU 584 triton dequant kernel can't compile on PPU (fp8e4nv)."""

    def test_dequant_gather_roundtrip(self):
        try:
            from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
                dequantize_and_gather_k_cache,
            )
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"_swa_dequant_triton import failed: {e}")

        torch.manual_seed(0)
        dev = "cuda"
        HEAD = 512
        num_blocks, block_size = 2, 64
        N = num_blocks * block_size
        k = torch.randn(N, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        pool = pack_ppu_kv_656(k).view(num_blocks, block_size, PPU_KV_ENTRY_BYTES)
        out = torch.zeros(1, N, HEAD, dtype=torch.bfloat16, device=dev)
        seq_lens = torch.tensor([N], dtype=torch.int32, device=dev)
        gather_lens = torch.tensor([N], dtype=torch.int32, device=dev)
        block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
        dequantize_and_gather_k_cache(out, pool, seq_lens, gather_lens, block_table, block_size, 0)
        err = ((out[0].float() - k.float()).norm() / k.float().norm()).item()
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertLess(err, 0.06, f"dequant+gather rel err {err}")

    def test_dequant_gather_slots_roundtrip(self):
        try:
            from rtp_llm.models_py.modules.dsv4.fp8._swa_dequant_triton import (
                dequantize_and_gather_k_cache_slots,
            )
        except Exception as e:  # noqa: BLE001
            self.skipTest(f"_swa_dequant_triton import failed: {e}")
        torch.manual_seed(0)
        dev = "cuda"
        HEAD = 512
        num_blocks, block_size = 2, 64
        N = num_blocks * block_size
        k = torch.randn(N, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        pool = pack_ppu_kv_656(k).view(num_blocks, block_size, PPU_KV_ENTRY_BYTES)
        slot = torch.arange(N, dtype=torch.int32, device=dev).view(1, N).clone()
        slot[0, 5] = -1  # -1 => zero-fill + skip
        out = torch.zeros(1, N, HEAD, dtype=torch.bfloat16, device=dev)
        gl = torch.tensor([N], dtype=torch.int32, device=dev)
        dequantize_and_gather_k_cache_slots(out, pool, slot, gl, 0)
        valid = torch.arange(N, device=dev) != 5
        err = ((out[0][valid].float() - k[valid].float()).norm() / k[valid].float().norm()).item()
        self.assertLess(err, 0.06, f"slots dequant rel err {err}")
        self.assertEqual(float(out[0, 5].abs().max()), 0.0)  # -1 row zeroed


if __name__ == "__main__":
    unittest.main()
