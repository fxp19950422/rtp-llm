"""Bounded prefill normalization must preserve the original per-row values."""

import os
import unittest
from unittest.mock import patch

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class PrefillNormChunksTest(unittest.TestCase):
    @torch.inference_mode()
    def test_replicated_tp_matches_original(self):
        from rtp_llm.models_py.modules.dsv4 import block

        torch.manual_seed(167)
        for length in (16383, 16384, 16385, 65537):
            for tp_size in (1, 4):
                with self.subTest(length=length, tp_size=tp_size):
                    x = torch.randn(length, 4096, device="cuda", dtype=torch.bfloat16)
                    norm = block.RMSNorm(
                        torch.randn(4096, device="cuda", dtype=torch.bfloat16), 1e-6
                    )
                    with patch.dict(os.environ, DSV4_PREFILL_NORM_CHUNK_TOKENS="0"):
                        expected = block._prefill_fast_norm(
                            norm, x.clone(), tp_size=tp_size
                        )
                    with patch.dict(os.environ, DSV4_PREFILL_NORM_CHUNK_TOKENS="16384"):
                        actual = block._prefill_fast_norm(norm, x, tp_size=tp_size)
                    torch.cuda.synchronize()
                    self.assertTrue(torch.equal(actual, expected))
                    if length > 16384:
                        self.assertEqual(actual.data_ptr(), x.data_ptr())
                    del expected, actual, x, norm


if __name__ == "__main__":
    unittest.main()
