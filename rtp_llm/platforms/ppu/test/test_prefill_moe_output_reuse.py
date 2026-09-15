import os
from types import SimpleNamespace
from unittest.mock import patch
import torch
from rtp_llm.platforms.ppu.models.dsv4 import ppu_tp_moe
import unittest


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class PrefillMoEReuseTest(unittest.TestCase):
    def test_completed_chunks_replace_disposable_input(self):
        with torch.inference_mode():
            for n in [32768, 32769, 65537]:
                x = torch.randn(n, 64, device="cuda", dtype=torch.bfloat16)
                ids = torch.arange(n, device="cuda")
                seen = []

                def local(x, ids):
                    seen.extend(ids.cpu().tolist())
                    return x * 2

                obj = SimpleNamespace(
                    dim=64, max_tokens_per_rank=2048, _forward_local_chunk=local
                )
                expected = x * 2
                with patch.dict(os.environ, DSV4_PREFILL_REUSE_MOE_INPUT="1"):
                    actual = ppu_tp_moe.PpuTPMoE.forward(obj, x, ids)
                assert torch.equal(actual, expected) and seen == list(range(n))
                assert (actual.data_ptr() == x.data_ptr()) == (n > 32768)
                print("PASS", n, flush=True)


if __name__ == "__main__":
    unittest.main()
