"""Long FP4 indexer selection preserves BF16 values and request-local ranges."""

import unittest

import torch


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_name() == "ZW-M890P",
    "requires a PPU M890P",
)
class LongBF16TopKTest(unittest.TestCase):
    def test_ranges_padding_and_selected_values(self):
        from rtp_llm.platforms.ppu.kernels.cuda.ppu_fp4_indexer import topk_bf16

        torch.manual_seed(1109)
        cases = (
            (16384, 8, "random"),
            (16385, 8, "random"),
            (32768, 8, "ties"),
            (262144, 257, "random"),
            (262144, 8, "ties"),
            (262144, 8, "negative"),
        )
        for cols, rows, kind in cases:
            with self.subTest(columns=cols, rows=rows, kind=kind):
                scores = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16)
                if kind == "ties":
                    scores.zero_()
                elif kind == "negative":
                    scores = -scores.abs() - 1
                starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
                starts[1::2] = 17
                ends = torch.full_like(starts, cols)
                for row, length in enumerate([0, 17, 511, 512, 513]):
                    ends[row] = starts[row] + length
                out = torch.full((rows, 512), -2, device="cuda", dtype=torch.int32)
                topk_bf16(scores, starts, ends, out)
                torch.cuda.synchronize()
                for row in range(rows):
                    lo, hi = int(starts[row]), int(ends[row])
                    length = max(0, hi - lo)
                    count = min(512, length)
                    selected = out[row]
                    valid = selected[selected >= 0].long()
                    self.assertEqual(valid.numel(), count)
                    self.assertEqual(valid.unique().numel(), count)
                    self.assertTrue(bool((valid < length).all()))
                    self.assertTrue(bool((selected[count:] == -1).all()))
                    actual = (
                        scores[row, lo + valid].float().sort(descending=True).values
                    )
                    expected = scores[row, lo:hi].float().topk(count).values
                    # Tied candidates may have different indices, but all selected
                    # values must match exact TopK and indices must be unique.
                    self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()
