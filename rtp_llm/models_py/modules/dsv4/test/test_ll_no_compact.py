"""Aliasing checks for the DeepEP LL no-compact path.

Under ``DSV4_MOE_LL_NO_COMPACT`` the down-projection GEMM writes straight into
the combine buffer, skipping a full-slot staging copy.  That is only sound while
the buffer's bytes are disjoint from the dispatch payload the first GEMM still
reads, and both can live in one RDMA arena.  A false "disjoint" here does not
raise -- it silently corrupts activations -- so the span arithmetic is what this
file exists to hold down.
"""

import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies import deepep


class TensorByteSpanTest(unittest.TestCase):
    def test_contiguous_span_is_exactly_its_bytes(self):
        t = torch.zeros(10, 4, dtype=torch.bfloat16)
        start, end = deepep._tensor_byte_span(t)
        self.assertEqual(start, t.data_ptr())
        self.assertEqual(end - start, t.numel() * t.element_size())

    def test_narrowed_view_spans_less_than_its_base(self):
        base = torch.zeros(100, dtype=torch.float32)
        view = base[10:20]
        _, base_end = deepep._tensor_byte_span(base)
        view_start, view_end = deepep._tensor_byte_span(view)
        self.assertEqual(view_start, base.data_ptr() + 10 * 4)
        self.assertEqual(view_end - view_start, 10 * 4)
        self.assertLess(view_end, base_end)

    def test_strided_view_spans_its_reach_not_its_element_count(self):
        # Every other row: the view holds 20 elements but reaches across 36.
        # Sizing the guard by numel would declare a real overlap disjoint.
        base = torch.zeros(10, 4, dtype=torch.float32)
        view = base[::2]
        start, end = deepep._tensor_byte_span(view)
        reach = ((view.size(0) - 1) * view.stride(0) + (view.size(1) - 1) + 1) * 4
        self.assertEqual(end - start, reach)
        self.assertGreater(end - start, view.numel() * 4)

    def test_empty_tensor_has_no_span(self):
        self.assertEqual(deepep._tensor_byte_span(torch.zeros(0)), (0, 0))


class SpansOverlapTest(unittest.TestCase):
    def test_tensor_overlaps_itself(self):
        t = torch.zeros(8, dtype=torch.float32)
        self.assertTrue(deepep._spans_overlap(t, t))

    def test_separate_allocations_are_disjoint(self):
        a = torch.zeros(64, dtype=torch.float32)
        b = torch.zeros(64, dtype=torch.float32)
        self.assertFalse(deepep._spans_overlap(a, b))

    def test_partially_overlapping_views_of_one_storage(self):
        base = torch.zeros(100, dtype=torch.float32)
        self.assertTrue(deepep._spans_overlap(base[0:10], base[5:15]))

    def test_touching_views_are_not_overlapping(self):
        # The end is exclusive: b starts exactly where a stops.  An inclusive
        # comparison here would reject every legitimately adjacent buffer and
        # fall back to the copy on every step.
        base = torch.zeros(100, dtype=torch.float32)
        self.assertFalse(deepep._spans_overlap(base[0:10], base[10:20]))

    def test_disjoint_views_of_one_storage(self):
        base = torch.zeros(100, dtype=torch.float32)
        self.assertFalse(deepep._spans_overlap(base[0:10], base[50:60]))

    def test_interleaved_strided_views_are_treated_as_overlapping(self):
        # base[::2] and base[1::2] share no element, but their byte ranges
        # interleave.  The guard is deliberately conservative: it costs a copy,
        # where being wrong costs correctness.
        base = torch.zeros(20, dtype=torch.float32)
        self.assertTrue(deepep._spans_overlap(base[::2], base[1::2]))

    def test_empty_tensor_never_overlaps(self):
        base = torch.zeros(10, dtype=torch.float32)
        self.assertFalse(deepep._spans_overlap(base, base[0:0]))
        self.assertFalse(deepep._spans_overlap(base[0:0], base))

    def test_differing_dtypes_compare_by_bytes(self):
        # A bf16 view and an fp32 view of one buffer overlap even though neither
        # element count nor index range says so.
        base = torch.zeros(64, dtype=torch.float32)
        as_bytes = base.view(torch.bfloat16)
        self.assertTrue(deepep._spans_overlap(base, as_bytes))


class NoCompactDefaultTest(unittest.TestCase):
    def test_off_unless_asked_for(self):
        if os.environ.get("DSV4_MOE_LL_NO_COMPACT"):
            self.skipTest("DSV4_MOE_LL_NO_COMPACT set in this environment")
        self.assertFalse(deepep._LL_NO_COMPACT)


if __name__ == "__main__":
    unittest.main()
