"""Capture-safety and counting checks for the DeepEP LL capacity overflow monitor.

The monitor's reader syncs device->host and sits one call away from the forward
path.  Production decode captures CUDA graphs, where that sync aborts the whole
engine during warmup -- so the guards, not the arithmetic, are what this file
exists to hold down.
"""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4.moe.strategies import deepep

CAP = 128
_MOD = "rtp_llm.models_py.modules.dsv4.moe.strategies.deepep"


def _expected_dropped(counts, capacity):
    return sum(max(0, c - capacity) for c in counts)


class _ExplodingStats:
    """Stand-in for the stats buffer that fails loudly on any host access.

    Asserting on a return value cannot tell "guarded" from "synced and happened
    to return None".  Raising here makes an unguarded sync a test failure with a
    stack that points at the caller.
    """

    def tolist(self):
        raise AssertionError("device->host sync reached during graph capture")

    def zero_(self):
        raise AssertionError("device write reached during graph capture")


def _capturing():
    return patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.is_current_stream_capturing", return_value=True
    )


class CapacityOverflowCountingTest(unittest.TestCase):
    """The numbers the monitor reports must be exact, not approximate."""

    def test_counts_below_capacity_drop_nothing(self):
        mon = deepep._CapacityOverflowMonitor()
        counts = [7, 8, 6, 9, 7, 8, 7, 6]
        mon.record(torch.tensor(counts, dtype=torch.int32), CAP)
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(mon.read(), (0, max(counts)))

    def test_capacity_boundary_is_not_an_overflow(self):
        mon = deepep._CapacityOverflowMonitor()
        mon.record(torch.tensor([CAP] * 4, dtype=torch.int32), CAP)
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(mon.read(), (0, CAP))

    def test_records_accumulate_and_reset(self):
        mon = deepep._CapacityOverflowMonitor()
        seq = [[CAP + 5, 1], [CAP + 50, 2], [3, 3]]
        for counts in seq:
            mon.record(torch.tensor(counts, dtype=torch.int32), CAP)
        want = (
            sum(_expected_dropped(c, CAP) for c in seq),
            max(max(c) for c in seq),
        )
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(mon.read(reset=True), want)
            self.assertEqual(mon.read(reset=False), (0, 0))

    def test_empty_counts_allocate_nothing(self):
        mon = deepep._CapacityOverflowMonitor()
        mon.record(torch.tensor([], dtype=torch.int32), CAP)
        with patch("torch.cuda.is_available", return_value=False):
            self.assertIsNone(mon.read())


class CapacityOverflowCaptureSafetyTest(unittest.TestCase):
    """Nothing on the forward path may sync while a graph is being captured."""

    def test_read_refuses_to_sync_during_capture(self):
        mon = deepep._CapacityOverflowMonitor()
        mon._stats = _ExplodingStats()
        a, b = _capturing()
        with a, b:
            self.assertIsNone(mon.read(reset=True))

    def test_record_stays_available_during_capture(self):
        # record() is pure device arithmetic, so capture must not disable it --
        # losing the counts would defeat the monitor on the only path that runs
        # under a graph.
        mon = deepep._CapacityOverflowMonitor()
        a, b = _capturing()
        with a, b:
            mon.record(torch.tensor([CAP + 3, 1], dtype=torch.int32), CAP)
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(mon.read(), (3, CAP + 3))

    def test_logger_is_silent_during_capture(self):
        mon = deepep._CapacityOverflowMonitor()
        mon._stats = _ExplodingStats()
        counts = torch.tensor([CAP + 1, 2], dtype=torch.int32)
        a, b = _capturing()
        with a, b, patch.object(deepep, "_OVF_LOG_EVERY", 1), patch.object(
            deepep, "_CAPACITY_OVERFLOW_INSTANCES", [mon]
        ):
            deepep._maybe_log_capacity_overflow(CAP, counts)

    def test_logger_period_ignores_capture_passes(self):
        # The counter must only advance on calls that could log.  Otherwise the
        # capture passes burn the period and the first eager line lands at an
        # arbitrary step.
        mon = deepep._CapacityOverflowMonitor()
        mon.record(torch.tensor([CAP + 1, 2], dtype=torch.int32), CAP)
        counts = torch.tensor([CAP + 1, 2], dtype=torch.int32)
        with patch.object(deepep, "_OVF_LOG_EVERY", 2), patch.object(
            deepep, "_CAPACITY_OVERFLOW_INSTANCES", [mon]
        ), patch.object(deepep, "_OVF_LOG_CALLS", [0]):
            a, b = _capturing()
            with a, b:
                for _ in range(5):
                    deepep._maybe_log_capacity_overflow(CAP, counts)
            self.assertEqual(deepep._OVF_LOG_CALLS[0], 0)

            with patch("torch.cuda.is_available", return_value=False), patch.object(
                deepep.logging, "info"
            ) as info:
                deepep._maybe_log_capacity_overflow(CAP, counts)
                self.assertEqual(deepep._OVF_LOG_CALLS[0], 1)
                info.assert_not_called()
                deepep._maybe_log_capacity_overflow(CAP, counts)
                self.assertEqual(deepep._OVF_LOG_CALLS[0], 2)
                self.assertEqual(info.call_count, 1)

    def test_logger_off_by_default(self):
        mon = deepep._CapacityOverflowMonitor()
        mon._stats = _ExplodingStats()
        with patch.object(deepep, "_OVF_LOG_EVERY", 0), patch.object(
            deepep, "_CAPACITY_OVERFLOW_INSTANCES", [mon]
        ), patch("torch.cuda.is_available", return_value=False):
            deepep._maybe_log_capacity_overflow(CAP, None)

    def test_capture_probe_matches_the_dispatch_site(self):
        # The dispatch path already gates on is_available()+is_current_stream_
        # capturing(); a second idiom here would drift from it.
        with patch("torch.cuda.is_available", return_value=False), patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ):
            self.assertFalse(deepep._graph_capture_active())
        a, b = _capturing()
        with a, b:
            self.assertTrue(deepep._graph_capture_active())


if __name__ == "__main__":
    unittest.main()
