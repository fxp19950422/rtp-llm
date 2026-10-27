"""Hardware diagnostic: unused paged KV tokens must not affect decode."""

import unittest

import torch

from rtp_llm.models_py.modules.factory.attention.ppu_impl.fa3_mha import (
    _import_fa3_interface,
)


class TestUnusedCache(unittest.TestCase):
    def test_unused_tokens(self):
        fa3 = _import_fa3_interface()
        torch.manual_seed(42)
        for dtype in (torch.float16, torch.bfloat16):
            for length in (17, 63, 65, 129):
                for splits in (0, 1):
                    with self.subTest(dtype=dtype, length=length, splits=splits):
                        blocks = (length + 63) // 64
                        q = torch.randn(1, 16, 128, device="cuda", dtype=dtype)
                        packed = torch.randn(blocks, 2, 16, 64, 128, device="cuda", dtype=dtype)
                        lens = torch.tensor([length], device="cuda", dtype=torch.int32)
                        cu = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
                        table = torch.arange(blocks, device="cuda", dtype=torch.int32)[None, :]

                        def run():
                            metadata = fa3.get_scheduler_metadata(
                                batch_size=1, max_seqlen_q=1, max_seqlen_k=8192,
                                num_heads_q=16, num_heads_kv=16, headdim=128,
                                cache_seqlens=lens, qkv_dtype=torch.bfloat16,
                                cu_seqlens_q=cu, page_size=64, causal=True,
                                num_splits=splits,
                            )
                            return fa3.flash_attn_with_kvcache(
                                q=q, k_cache=packed[:, 0].transpose(1, 2),
                                v_cache=packed[:, 1].transpose(1, 2),
                                cache_seqlens=lens, page_table=table, cu_seqlens_q=cu,
                                softmax_scale=128 ** -0.5, causal=True,
                                max_seqlen_q=1, max_seqlen_k=8192,
                                scheduler_metadata=metadata, num_splits=splits,
                            )

                        expected = run().clone()
                        packed[-1, :, :, length % 64:, :] = float("nan")
                        actual = run()
                        print(f"dtype={dtype} length={length} splits={splits} "
                              f"baseline_finite={torch.isfinite(expected).all().item()} "
                              f"poisoned_finite={torch.isfinite(actual).all().item()}", flush=True)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
