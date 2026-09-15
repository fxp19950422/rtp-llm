import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import flash_mla
from rtp_llm.models_py.modules.dsv4.fp8 import attention as mod


@unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
class PrefillAttentionOutputReuseTest(unittest.TestCase):
    def test_chunk_output_uses_supplied_buffer(self):
        with torch.inference_mode():
            for rows in [16383, 16384, 16385, 32769]:
                q = torch.randn(rows, 1, 16, device="cuda", dtype=torch.bfloat16)
                x = torch.randn(rows, 16, device="cuda", dtype=torch.bfloat16)

                def project(o, f, out):
                    out.copy_(o.flatten(1))

                obj = SimpleNamespace(
                    n_heads=1,
                    head_dim=16,
                    dim=16,
                    softmax_scale=1.0,
                    attn_sink=None,
                    compress_ratio=0,
                    _prefill_output_proj_into=project,
                    _prefill_output_all_reduce=lambda out: None,
                )
                kw = dict(
                    q=q,
                    kv=q[:1],
                    indices=torch.zeros(rows, 1, 1, device="cuda", dtype=torch.int32),
                    topk_length=torch.ones(rows, device="cuda", dtype=torch.int32),
                    freqs_cis=torch.ones(rows, 1, device="cuda"),
                    prefill_workspace=SimpleNamespace(prefill_q=lambda n: q),
                    profile_name="test",
                )
                with patch.object(
                    flash_mla,
                    "flash_mla_sparse_fwd",
                    side_effect=lambda **k: (k["q"].clone(), None, None),
                ):
                    original = mod.AttentionFP8._flash_mla_sparse_fwd_chunked_projected(
                        obj, **kw
                    )
                    reused = mod.AttentionFP8._flash_mla_sparse_fwd_chunked_projected(
                        obj, **kw, out=x
                    )
                assert (
                    torch.equal(original, reused) and reused.data_ptr() == x.data_ptr()
                )
                print("PASS", rows, flush=True)


if __name__ == "__main__":
    unittest.main()
