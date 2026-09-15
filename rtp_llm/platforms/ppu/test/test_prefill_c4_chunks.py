"""C4 compression must preserve causal overlap and write each slot once."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class PrefillC4ChunksTest(unittest.TestCase):
    def metadata(self, lengths, starts):
        positions = torch.cat(
            [
                torch.arange(s, s + n, device="cuda", dtype=torch.int64)
                for n, s in zip(lengths, starts)
            ]
        )
        requests = torch.repeat_interleave(
            torch.arange(len(lengths), device="cuda"),
            torch.tensor(lengths, device="cuda"),
        )
        state_slots = torch.full_like(positions, -1)
        offset = 0
        for request, length in enumerate(lengths):
            count = min(length, 8)
            rows = slice(offset + length - count, offset + length)
            state_slots[rows] = request * 8 + positions[rows] % 8
            offset += length
        return SimpleNamespace(
            positions=positions,
            b_idx=requests,
            state_slots=state_slots,
            kv_slots=torch.where(
                (positions + 1) % 4 == 0,
                torch.arange(sum(lengths), device="cuda"),
                -1,
            ),
            is_batched=True,
            seq_start_per_req=torch.tensor(starts, device="cuda"),
        )

    def execute(self, meta, fused, ape, initial, use_model):
        from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import compress4
        from rtp_llm.platforms.ppu.kernels.ppu_fp4_indexer_cache import build_plans
        from rtp_llm.platforms.ppu.models.dsv4 import ppu_fp4_indexer as model

        state = initial.clone()
        table = torch.arange(len(initial), device="cuda", dtype=torch.int32)
        table = table.view(-1, 1)
        count = len(fused)
        output = torch.zeros((count, 128), device="cuda")
        seen = torch.zeros(count, device="cuda", dtype=torch.int32)

        def record(compressed, plans, weight, eps, freqs, slots, cache, page_size):
            valid = slots >= 0
            output[slots[valid]] = compressed[valid]
            seen[slots[valid]] += 1

        if use_model:
            compressor = SimpleNamespace(
                _state_pool_3d=state,
                _kv_pool_view=torch.empty(0, device="cuda", dtype=torch.uint8),
                _cp_ctx=None,
                _state_block_table=table,
                _state_eb=8,
                _state_tokens_per_block=256,
                ape=ape,
                norm=SimpleNamespace(weight=torch.ones(128, device="cuda")),
                norm_eps=1e-6,
                freqs_cis=torch.empty(0, device="cuda", dtype=torch.complex64),
                _kv_eb=64,
            )
            with patch.object(model, "norm_rope_store", record):
                model.PpuFP4Compressor._launch(
                    compressor, fused[:, :256], fused[:, 256:], meta
                )
        else:
            # Within 64K this is the original single-call path. Above 64K,
            # a different partition checks that boundaries do not affect values.
            step = count if count <= 65536 else 4096
            for begin in range(0, count, step):
                lo, hi = max(0, begin - 8), min(count, begin + step)
                plans, writes, slots = build_plans(
                    meta,
                    table,
                    8,
                    256,
                    None,
                    row_start=lo,
                    row_end=hi,
                    skip_rows=begin - lo,
                )
                values = compress4(state, fused[lo:hi], ape, plans, writes)
                record(values, plans, None, None, None, slots, None, None)
        torch.cuda.synchronize()
        return output, state, seen

    @torch.inference_mode()
    def test_compression_state_and_slot_coverage(self):
        torch.manual_seed(73019)
        cases = (
            ([65536], [0]),
            ([65521], [7]),
            ([4093, 123, 2107, 5731], [0, 3, 257, 15]),
            ([131073], [0]),
            ([65529, 65544], [5, 19]),
            ([1048576], [0]),
        )
        for lengths, starts in cases:
            with self.subTest(lengths=lengths, starts=starts):
                meta = self.metadata(lengths, starts)
                fused = torch.randn(sum(lengths), 512, device="cuda")
                ape = torch.randn(8, 128, device="cuda")
                initial = torch.randn(len(lengths), 8, 512, device="cuda")
                actual = self.execute(meta, fused, ape, initial, True)
                expected = self.execute(meta, fused, ape, initial, False)
                for got, want in zip(actual, expected):
                    self.assertTrue(torch.equal(got, want))
                self.assertTrue(
                    torch.equal(actual[2], (meta.kv_slots >= 0).to(torch.int32))
                )
                del actual, expected, meta, fused, ape, initial


if __name__ == "__main__":
    unittest.main()
