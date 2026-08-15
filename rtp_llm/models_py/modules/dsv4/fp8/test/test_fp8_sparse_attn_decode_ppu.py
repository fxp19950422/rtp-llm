"""Unit tests for the PPU flash_mla_with_kvcache decode path.

The PPU flash_mla wheel exposes only the reduced sparse signature (no
``attn_sink`` / ``topk_length`` / ``extra_*`` kwargs) and returns a 2-based
(log2) softmax_lse. ``SparseAttnV4DecodeFp8Op`` reproduces the GPU kernel's
dual-pool + attn-sink semantics in Python. These tests lock in that logic:

  * the reduced-signature dispatch (no attn_sink/extra kwargs; block_table and
    cache_seqlens are real tensors, not None) via a mocked kernel;
  * the pure post-processing math (topk_length idx-mask, base-e sink from a
    2-based lse, 2-based online-softmax dual-pool merge).

The math primitives were validated against the real PPU kernel in the
standalone decode verification (single-pool rel err ~0.002, dual-pool merge vs
combined-call rel err ~0.003, sink ln2-correction rel err ~0.003).
"""

from __future__ import annotations

import math
import sys
import types
import unittest

import torch

import rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op as decode_mod
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)


def _fake_flash_mla_module(calls):
    """flash_mla stub with the *reduced* PPU signature (no attn_sink)."""
    mod = types.ModuleType("flash_mla")

    def flash_mla_with_kvcache(
        q,
        k_cache,
        block_table,
        cache_seqlens,
        head_dim_v,
        tile_scheduler_metadata,
        num_splits,
        softmax_scale=None,
        causal=False,
        is_fp8_kvcache=False,
        indices=None,
    ):
        calls.append(
            {
                "q": q,
                "k_cache": k_cache,
                "block_table": block_table,
                "cache_seqlens": cache_seqlens,
                "indices": indices,
                "is_fp8_kvcache": is_fp8_kvcache,
            }
        )
        B, q_len, H, _ = q.shape
        out = torch.full((B, q_len, H, head_dim_v), float(len(calls)), dtype=q.dtype)
        lse = torch.full((B, H, q_len), 0.5 * len(calls), dtype=torch.float32)
        return out, lse

    def get_mla_metadata(
        cache_seqlens,
        num_heads_per_head_k,
        num_heads_k,
        num_heads_q=None,
        is_fp8_kvcache=False,
        topk=None,
    ):
        return object(), torch.zeros(1, dtype=torch.int32)

    mod.flash_mla_with_kvcache = flash_mla_with_kvcache
    mod.get_mla_metadata = get_mla_metadata
    return mod


class TestDecodePpuMathHelpers(unittest.TestCase):
    def setUp(self):
        self.op = SparseAttnV4DecodeFp8Op(n_heads=4, head_dim=8, softmax_scale=1.0)

    def test_mask_topk_length(self):
        idx = torch.arange(6, dtype=torch.int32).view(1, 1, 6).expand(2, 1, 6).contiguous()
        tl = torch.tensor([4, 2], dtype=torch.int32)
        masked = self.op._ppu_mask_topk_length(idx, tl)
        # request 0: keep cols < 4; request 1: keep cols < 2
        self.assertEqual(masked[0, 0].tolist(), [0, 1, 2, 3, -1, -1])
        self.assertEqual(masked[1, 0].tolist(), [0, 1, -1, -1, -1, -1])
        # None -> unchanged
        self.assertTrue(torch.equal(self.op._ppu_mask_topk_length(idx, None), idx))

    def test_apply_sink_ln2_base(self):
        B, q_len, H, D = 1, 1, 4, 8
        out = torch.randn(B, q_len, H, D)
        lse2 = torch.randn(B, q_len, H, 1).abs() + 1.0  # 2-based lse
        sink = torch.rand(H) * 0.5 + 0.2
        got = self.op._ppu_apply_sink(out.clone(), lse2, sink)
        # reference: sink is base-e (utils._sparse_attn) -> convert lse log2->ln
        factor = torch.sigmoid(lse2 * math.log(2.0) - sink.view(1, 1, H, 1))
        exp = out * factor
        torch.testing.assert_close(got, exp, rtol=1e-5, atol=1e-5)
        # None sink -> passthrough
        self.assertTrue(torch.equal(self.op._ppu_apply_sink(out, lse2, None), out))

    def test_merge_pools_matches_combined_softmax(self):
        # Build two partial softmax states from raw scores; the 2-based merge
        # must equal a single softmax over the concatenated scores.
        torch.manual_seed(0)
        B, q_len, H, D, K = 1, 1, 4, 8, 5
        v1 = torch.randn(K, D)
        v2 = torch.randn(K, D)
        s1 = torch.randn(B, q_len, H, K)
        s2 = torch.randn(B, q_len, H, K)

        def partial(scores, v):
            e = torch.exp(scores)
            o = torch.einsum("bqhk,kd->bqhd", e, v) / e.sum(-1, keepdim=True)
            lse2 = torch.log2(e.sum(-1, keepdim=True))  # 2-based
            return o, lse2

        o1, l1 = partial(s1, v1)
        o2, l2 = partial(s2, v2)
        merged, merged_lse = self.op._ppu_merge_pools(o1, l1, o2, l2)

        alls = torch.cat([s1, s2], dim=-1)
        allv = torch.cat([v1, v2], dim=0)
        e = torch.exp(alls)
        ref = torch.einsum("bqhk,kd->bqhd", e, allv) / e.sum(-1, keepdim=True)
        torch.testing.assert_close(merged, ref, rtol=1e-4, atol=1e-4)
        ref_lse2 = torch.log2(e.sum(-1, keepdim=True))
        torch.testing.assert_close(merged_lse, ref_lse2, rtol=1e-4, atol=1e-4)


class TestDecodePpuForwardDispatch(unittest.TestCase):
    def _run(self, extra=False):
        calls = []
        fake = _fake_flash_mla_module(calls)
        old_mod = sys.modules.get("flash_mla")
        old_flag = decode_mod._FLASH_MLA_KVCACHE_ATTN_SINK
        sys.modules["flash_mla"] = fake
        decode_mod._FLASH_MLA_KVCACHE_ATTN_SINK = False  # force PPU path
        try:
            op = SparseAttnV4DecodeFp8Op(n_heads=4, head_dim=8, softmax_scale=1.0)
            B, q_len, H = 2, 1, 4
            q = torch.randn(B, q_len, H, 8, dtype=torch.bfloat16)
            kv = torch.zeros(4, 16, 584, dtype=torch.uint8)
            topk = torch.randint(0, 64, (B, q_len, 32), dtype=torch.int32)
            sink = torch.rand(H, dtype=torch.float32)
            extra_kv = torch.zeros(2, 16, 584, dtype=torch.uint8) if extra else None
            extra_topk = (
                torch.randint(0, 32, (B, q_len, 16), dtype=torch.int32) if extra else None
            )
            out = op._forward_flash_mla(
                q=q,
                kv_cache=kv,
                attn_sink=sink,
                topk_idxs=topk,
                sched_meta=object(),
                cache_seqlens=None,
                block_table=None,
                extra_k_cache=extra_kv,
                extra_topk_idxs=extra_topk,
            )
            return out, calls
        finally:
            decode_mod._FLASH_MLA_KVCACHE_ATTN_SINK = old_flag
            if old_mod is None:
                sys.modules.pop("flash_mla", None)
            else:
                sys.modules["flash_mla"] = old_mod

    def test_single_pool_reduced_signature(self):
        out, calls = self._run(extra=False)
        self.assertEqual(tuple(out.shape), (2, 1, 4, 8))
        self.assertEqual(len(calls), 1)  # single pool -> one kernel call
        c = calls[0]
        # PPU requires real tensors (not None) for block_table + cache_seqlens
        self.assertIsInstance(c["block_table"], torch.Tensor)
        self.assertIsInstance(c["cache_seqlens"], torch.Tensor)
        self.assertTrue(c["is_fp8_kvcache"])

    def test_dual_pool_two_calls(self):
        out, calls = self._run(extra=True)
        self.assertEqual(tuple(out.shape), (2, 1, 4, 8))
        self.assertEqual(len(calls), 2)  # dual pool -> two kernel calls + merge


if __name__ == "__main__":
    unittest.main()
