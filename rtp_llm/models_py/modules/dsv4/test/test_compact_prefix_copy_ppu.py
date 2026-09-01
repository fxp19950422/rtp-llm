import unittest

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies._compact_prefix_copy import (
    compact_to_strided_prefix,
    prepare_compact_prefix_copy,
)


def _is_m890p() -> bool:
    try:
        return torch.cuda.is_available() and "M890P" in torch.cuda.get_device_name(0)
    except Exception:
        return False


@unittest.skipUnless(_is_m890p(), "requires an M890P PPU")
class CompactPrefixCopyPPUTest(unittest.TestCase):
    EXPERTS = 4
    COMPACT_ROWS = 7
    PADDED_ROWS = 19
    WIDTH = 64
    SENTINEL = 13.0

    @classmethod
    def setUpClass(cls) -> None:
        torch.cuda.set_device(0)
        prepare_compact_prefix_copy()

    def _source(self) -> torch.Tensor:
        values = torch.arange(
            self.EXPERTS * self.COMPACT_ROWS * self.WIDTH,
            dtype=torch.float32,
            device="cuda",
        )
        return (values.remainder(97) - 48).to(torch.bfloat16).view(
            self.EXPERTS, self.COMPACT_ROWS, self.WIDTH
        )

    def _destination(self) -> torch.Tensor:
        return torch.full(
            (self.EXPERTS, self.PADDED_ROWS, self.WIDTH),
            self.SENTINEL,
            dtype=torch.bfloat16,
            device="cuda",
        )

    def _assert_result(self, source: torch.Tensor, destination: torch.Tensor) -> None:
        self.assertTrue(
            torch.equal(destination[:, : self.COMPACT_ROWS, :], source)
        )
        self.assertTrue(
            bool(
                torch.all(
                    destination[:, self.COMPACT_ROWS :, :] == self.SENTINEL
                ).item()
            )
        )

    def test_eager_copy_preserves_entire_tail(self) -> None:
        source = self._source()
        destination = self._destination()
        compact_to_strided_prefix(source, destination)
        torch.cuda.synchronize()
        self._assert_result(source, destination)

    def test_non_default_stream_obeys_current_stream(self) -> None:
        source = self._source()
        destination = self._destination()
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            compact_to_strided_prefix(source, destination)
        side_stream.synchronize()
        self._assert_result(source, destination)

    def test_graph_replay_reads_updated_source_and_preserves_tail(self) -> None:
        source = self._source()
        destination = self._destination()
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                compact_to_strided_prefix(source, destination)
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compact_to_strided_prefix(source, destination)

        for value in (2.0, -3.0, 5.0):
            source.fill_(value)
            destination.fill_(self.SENTINEL)
            graph.replay()
            torch.cuda.synchronize()
            self._assert_result(source, destination)

    def test_contract_rejects_invalid_layouts(self) -> None:
        source = self._source()
        destination = self._destination()

        with self.assertRaisesRegex(TypeError, "BF16"):
            compact_to_strided_prefix(source.to(torch.float16), destination)

        noncontiguous = torch.empty(
            (self.EXPERTS, self.COMPACT_ROWS, self.WIDTH * 2),
            dtype=torch.bfloat16,
            device="cuda",
        )[:, :, : self.WIDTH]
        self.assertFalse(noncontiguous.is_contiguous())
        with self.assertRaisesRegex(ValueError, "contiguous"):
            compact_to_strided_prefix(noncontiguous, destination)

        too_small = torch.empty(
            (self.EXPERTS, self.COMPACT_ROWS - 1, self.WIDTH),
            dtype=torch.bfloat16,
            device="cuda",
        )
        with self.assertRaisesRegex(ValueError, "must not exceed"):
            compact_to_strided_prefix(source, too_small)

        storage = torch.empty(
            self.EXPERTS * self.PADDED_ROWS * self.WIDTH,
            dtype=torch.bfloat16,
            device="cuda",
        )
        overlapping_source = storage[
            : self.EXPERTS * self.COMPACT_ROWS * self.WIDTH
        ].view(self.EXPERTS, self.COMPACT_ROWS, self.WIDTH)
        overlapping_destination = storage.view(
            self.EXPERTS, self.PADDED_ROWS, self.WIDTH
        )
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            compact_to_strided_prefix(overlapping_source, overlapping_destination)


if __name__ == "__main__":
    unittest.main()
