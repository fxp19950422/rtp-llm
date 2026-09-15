import inspect
import textwrap
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import deep_gemm
from rtp_llm.platforms.ppu.models.dsv4 import ppu_fp4_indexer as baseline

candidate = baseline
source = textwrap.dedent(inspect.getsource(baseline.PpuFP4Indexer.forward))
source = source.replace("stream_queries = meta.M > 32768", "stream_queries = False")
namespace = dict(vars(baseline))
exec(compile(source, "<full-query-reference>", "exec"), namespace)
reference = SimpleNamespace(PpuFP4Indexer=SimpleNamespace(forward=namespace["forward"]))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires PPU M890P",
)
class PrefillQueryChunksTest(unittest.TestCase):
    def test_full_query_logits_and_topk_sets(self):
        torch.manual_seed(1094)
        M, K, H = 32769, 1024, 64
        x = torch.randn(M, 16, device="cuda", dtype=torch.bfloat16)
        qr = torch.randn(M, H * 128, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(H, 16, device="cuda", dtype=torch.bfloat16)
        freqs = torch.polar(
            torch.ones(M + 17, 32, device="cuda"),
            torch.randn(M + 17, 32, device="cuda"),
        )
        k = torch.randint(-127, 127, (K, 64), device="cuda", dtype=torch.int8)
        ks = torch.full((K, 1), 0x7F7F7F7F, device="cuda", dtype=torch.int32)
        meta = SimpleNamespace(
            M=M,
            T=K,
            sp_int=17,
            freqs_cis_slice=None,
            compressor_meta=SimpleNamespace(
                positions=torch.arange(17, M + 17, device="cuda")
            ),
            block_table_i32=None,
            cu_kv_seqlens=None,
            ks=torch.zeros(M, device="cuda", dtype=torch.int32),
            ke=torch.full((M,), K, device="cuda", dtype=torch.int32),
        )

        class Compressor:
            _profile_label = ""

            def __init__(self):
                self.calls = []

            def __call__(self, x, start, **kw):
                self.calls.append((len(x), start))

        def run(mod):
            snapshots = []
            rows = []
            cleaned = []
            compressor = Compressor()

            def project(q, freqs, apply_rope):
                assert not apply_rope
                rows.append(len(q))
                return q

            def score(qv, kv, w, lo, hi, **kw):
                result = deep_gemm.fp8_fp4_mqa_logits(qv, kv, w, lo, hi, **kw)
                snapshots.append(result.clone())
                return result

            obj = SimpleNamespace(
                _cp_ctx=None,
                _kv_pool_view=k,
                _kv_block_table=object(),
                _kv_eb=64,
                compressor=compressor,
                freqs_cis=freqs,
                n_heads=H,
                index_topk=512,
                _propagate_pool_to_nested=lambda: None,
                _clear_nested_pool=lambda: cleaned.append(True),
                _compute_indexer_q=project,
                weights_proj=weight,
                weight_scale=0.01,
                _prefill_score_chunk_rows=2048,
                _score=score,
            )
            with patch.object(baseline, "gather_k", return_value=(k, ks)), patch.dict(
                namespace, gather_k=lambda *a: (k, ks)
            ):
                out = mod.PpuFP4Indexer.forward(obj, x, qr, meta, workspace=None)
            torch.cuda.synchronize()
            assert compressor.calls == [(M, 17)] and cleaned == [True]
            return out, rows, snapshots

        with torch.inference_mode():
            expected, old_rows, old_scores = run(reference)
            actual, new_rows, new_scores = run(candidate)
            for a, b in zip(old_scores, new_scores):
                assert torch.equal(a, b), "score difference"
            assert torch.equal(
                expected.sort(-1).values, actual.sort(-1).values
            ), "TopK set difference"
            assert old_rows == [M] and max(new_rows) <= 2048 and sum(new_rows) == M
            print(
                json.dumps(
                    dict(
                        passed=True,
                        rows=M,
                        heads=H,
                        columns=K,
                        query_batches=new_rows,
                        topk_sets_equal=True,
                        all_logits_bitwise_equal=True,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
