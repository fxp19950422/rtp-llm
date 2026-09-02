"""Behaviour tests for the outside-graph capacity-overflow readout.

This host has no torch, so the file loads ``deepep.py`` with a minimal
list-backed torch stub plus stub siblings (``warmup_sync`` / ``base`` /
``local_loop``), following the repo's sanctioned monkeypatch-isolation
style.  Everything the code under test needs from torch is integer tensor
arithmetic; the stub fails with ``AttributeError`` on anything fancier, so
an accidental device-sync path shows up as a loud test failure rather than
a silent pass.

Covers the three P0 items from ROADMAP task #10:
* ``runtime_config`` switch registration with legacy defaults;
* ``drain_capacity_overflow_outside_graph`` (replay-side readout hook,
  wired into both decode fmha impls' ``prepare_cuda_graph``);
* per-expert routing percentiles (p50/p99) on both readout paths.

Run: ``python3 test_capacity_overflow_outside_graph.py`` (no bazel).
"""

import ast
import importlib.util
import logging
import os
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

_DSV4_DIR = Path(__file__).resolve().parents[1]
RC_PATH = _DSV4_DIR / "runtime_config.py"
DEEPEP_PATH = _DSV4_DIR / "moe" / "strategies" / "deepep.py"
BF16_IMPL_PATH = _DSV4_DIR / "decode" / "decode_fmha_impl.py"
FP8_IMPL_PATH = _DSV4_DIR / "fp8" / "decode" / "decode_fmha_impl.py"

_PKG = "rtp_llm.models_py.modules.dsv4"
_STRATS = _PKG + ".moe.strategies"

_TOUCHED_ENV = (
    "DSV4_MOE_CAPACITY_OVERFLOW_MONITOR",
    "DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY",
    "DSV4_MOE_ROUTER_PERCENTILES",
    "DSV4_MOE_ROUTER_PERCENTILE_LOG_EVERY",
)


class _StubScalar:
    def __init__(self, v):
        self._v = int(v)

    def to(self, dtype):
        return self

    def __index__(self):
        return self._v

    def __int__(self):
        return self._v

    def __repr__(self):
        return f"_StubScalar({self._v})"


class _StubTensor:
    """List-backed tensor: exactly the ops the monitor/log path touches."""

    device = "stub:cuda"

    def __init__(self, data):
        self._data = list(data)

    def numel(self):
        return len(self._data)

    def to(self, dtype):
        return _StubTensor(self._data)

    def sum(self):
        return sum(self._data)

    def max(self):
        return max(self._data) if self._data else 0

    def min(self):
        return min(self._data) if self._data else 0

    def __getitem__(self, i):
        return self._data[i]

    def __setitem__(self, i, v):
        self._data[i] = _unwrap(v) if isinstance(v, _StubScalar) else v

    def argmax(self):
        return _StubScalar(self._data.index(max(self._data)))

    def clamp(self, max=None, min=None):
        data = self._data
        if min is not None:
            data = [v if v >= min else min for v in data]
        if max is not None:
            data = [v if v <= max else max for v in data]
        return _StubTensor(data)

    def __gt__(self, other):
        return _StubTensor([1 if v > other else 0 for v in self._data])

    def __iadd__(self, other):
        other_data = other._data if isinstance(other, _StubTensor) else other
        for i in range(len(self._data)):
            self._data[i] += other_data[i]
        return self

    def tolist(self):
        return list(self._data)

    def zero_(self):
        self._data = [0] * len(self._data)


def _unwrap(x):
    return x._v if isinstance(x, _StubScalar) else x


def _fmt_info_call_like(call):
    """Replay one element of a mock's call list into its final string."""
    fmt, args = call.args[0], call.args[1:]
    return fmt % args


def _fmt_info_call(info):
    """Replay a ``logging.info(fmt, *args)`` mock call into its final string."""
    return _fmt_info_call_like(info.call_args)


def _install_stub_environment():
    # Never trample a real torch that an outer harness (e.g. a bazel
    # py_test with the full runtime on the path) may already have loaded.
    real = sys.modules.get("torch")
    if real is not None and not getattr(real, "__t10_stub__", False):
        raise unittest.SkipTest("real torch already loaded; stub install refused")
    if getattr(sys.modules.get("torch"), "__t10_stub__", False):
        return
    torch_mod = types.ModuleType("torch")
    torch_mod.__t10_stub__ = True
    torch_mod.int64 = "stub:int64"
    torch_mod.int32 = "stub:int32"
    torch_mod.float32 = "stub:float32"
    torch_mod.Tensor = _StubTensor
    torch_mod.zeros = lambda n, dtype=None, device=None: _StubTensor([0] * n)
    torch_mod.tensor = lambda data, dtype=None: _StubTensor(data)
    torch_mod.maximum = lambda a, b: max(_unwrap(a), _unwrap(b))
    torch_mod.stack = lambda items: _StubTensor([_unwrap(i) for i in items])
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.__t10_stub__ = True
    cuda_mod.is_available = lambda: False
    cuda_mod.is_current_stream_capturing = lambda: False
    torch_mod.cuda = cuda_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.cuda"] = cuda_mod

    # Package chain + deepep's sibling imports, as inert stubs.  The real
    # chain (models_py.modules.__init__ etc.) needs the serving container.
    for name in (
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.modules",
        _PKG,
        _PKG + ".moe",
        _STRATS,
    ):
        mod = types.ModuleType(name)
        mod.__t10_stub__ = True
        sys.modules[name] = mod

    warm = types.ModuleType(_PKG + ".moe.warmup_sync")
    warm.__t10_stub__ = True
    warm.cuda_graph_warmup_forward_enabled = lambda: False
    warm.sync_cuda_graph_warmup_ranks = lambda *a, **k: None
    base = types.ModuleType(_STRATS + ".base")
    base.__t10_stub__ = True
    base.MoeCfg = type("MoeCfg", (), {})
    base.RoutedExpertsStrategy = type("RoutedExpertsStrategy", (), {})
    base.register_strategy = lambda cls: cls
    loop = types.ModuleType(_STRATS + ".local_loop")
    loop.__t10_stub__ = True
    loop.LocalLoopStrategy = type("LocalLoopStrategy", (), {})
    for mod in (warm, base, loop):
        sys.modules[mod.__name__] = mod
    sys.modules[_PKG + ".moe"].warmup_sync = warm
    sys.modules[_STRATS].base = base
    sys.modules[_STRATS].local_loop = loop


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_deepep(env=None):
    """Fresh deepep (and runtime_config) with the given env snapshot."""
    _install_stub_environment()
    saved = {k: os.environ.get(k) for k in _TOUCHED_ENV}
    for k, v in (env or {}).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        rc = _load_module(RC_PATH, _PKG + ".runtime_config")
        sys.modules[_PKG].runtime_config = rc
        dp = _load_module(DEEPEP_PATH, _STRATS + ".deepep")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return dp


class _ExplodingStats:
    """Fails loudly on any host access (capture-safety canary)."""

    def tolist(self):
        raise AssertionError("device->host sync reached during graph capture")

    def zero_(self):
        raise AssertionError("device write reached during graph capture")


def _capturing():
    return patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.is_current_stream_capturing", return_value=True
    )


CAP = 128


class OutsideGraphDefaultsTest(unittest.TestCase):
    """The new switches must default to byte-identical legacy behaviour."""

    def test_defaults_match_legacy(self):
        dp = _load_deepep()
        self.assertEqual(dp._OVF_OUTSIDE_EVERY, 0)
        self.assertEqual(dp._OVF_ROUTER_PERCENTILES, ())
        self.assertEqual(dp._OVF_ROUTER_PCTL_EVERY, 0)

    def test_disabled_drain_never_touches_the_monitor(self):
        dp = _load_deepep()
        mon = dp._CapacityOverflowMonitor()
        mon._stats = _ExplodingStats()
        with patch.object(dp, "_CAPACITY_OVERFLOW_INSTANCES", [mon]):
            # Default EVERY=0: early return before any read/reset attempt.
            dp.drain_capacity_overflow_outside_graph()

    def test_per_expert_tracking_stays_off_by_default(self):
        dp = _load_deepep()
        mon = dp._CapacityOverflowMonitor()
        mon.record(dp.torch.tensor([CAP + 1, 2]), CAP)
        self.assertIsNone(mon._per_expert)  # no extra device op, no memory

    def test_env_values_reach_the_module_constants(self):
        dp = _load_deepep(
            {
                "DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY": "7",
                "DSV4_MOE_ROUTER_PERCENTILES": "p50,p99",
                "DSV4_MOE_ROUTER_PERCENTILE_LOG_EVERY": "10",
            }
        )
        self.assertEqual(dp._OVF_OUTSIDE_EVERY, 7)
        self.assertEqual(dp._OVF_ROUTER_PERCENTILES, (50, 99))
        self.assertEqual(dp._OVF_ROUTER_PCTL_EVERY, 10)

    def test_invalid_switch_value_fails_loud_at_import(self):
        with self.assertRaises(ValueError) as ctx:
            _load_deepep({"DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY": "abc"})
        self.assertIn("DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY", str(ctx.exception))

    def test_negative_switch_value_is_rejected(self):
        with self.assertRaises(ValueError):
            _load_deepep({"DSV4_MOE_ROUTER_PERCENTILE_LOG_EVERY": "-2"})

    def test_switch_resolution_audits_through_dsv4_config(self):
        with self.assertLogs(level=logging.INFO) as logs:
            _load_deepep()
        audited = [r for r in logs.output if "[DSV4_CONFIG]" in r]
        names = {
            "DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY",
            "DSV4_MOE_ROUTER_PERCENTILES",
            "DSV4_MOE_ROUTER_PERCENTILE_LOG_EVERY",
        }
        for name in names:
            self.assertTrue(
                any(name in r for r in audited), f"missing audit line for {name}"
            )

    def test_module_files_live_on_the_wheel_import_path(self):
        dp = _load_deepep()
        for path in (dp.__file__, sys.modules[_PKG + ".runtime_config"].__file__):
            resolved = str(Path(path).resolve())
            self.assertIn(
                os.path.join("rtp_llm", "models_py", "modules", "dsv4"), resolved
            )
            self.assertNotIn("internal_source", resolved)

    def test_stub_install_refuses_to_trample_a_real_torch(self):
        """A harness that already loaded the real torch must not be
        overwritten by the stub (review item: silent import trampling)."""
        real = types.ModuleType("torch")
        real.__version__ = "2.x-real"  # NOT marked __t10_stub__
        saved = sys.modules.get("torch")
        sys.modules["torch"] = real
        try:
            with self.assertRaises(unittest.SkipTest):
                _install_stub_environment()
        finally:
            if saved is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = saved


class PerExpertTrackingTest(unittest.TestCase):
    def setUp(self):
        self.dp = _load_deepep()

    def test_record_accumulates_per_expert_and_read_resets(self):
        dp = self.dp
        mon = dp._CapacityOverflowMonitor()
        with patch.object(dp, "_OVF_OUTSIDE_EVERY", 5):
            mon.record(dp.torch.tensor([CAP + 2, 1]), CAP)
            mon.record(dp.torch.tensor([CAP + 2, 3]), CAP)
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(mon.read_per_expert(reset=True), [2 * (CAP + 2), 4])
            self.assertEqual(mon.read_per_expert(reset=False), [0, 0])

    def test_legacy_two_scalar_read_is_unchanged(self):
        dp = self.dp
        mon = dp._CapacityOverflowMonitor()
        with patch.object(dp, "_OVF_OUTSIDE_EVERY", 5):
            mon.record(dp.torch.tensor([CAP + 2, 1]), CAP)
            mon.record(dp.torch.tensor([2, 3]), CAP)
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(mon.read(reset=True), (2, CAP + 2))
            self.assertEqual(mon.read(reset=False), (0, 0))

    def test_read_per_expert_refuses_to_sync_during_capture(self):
        dp = self.dp
        mon = dp._CapacityOverflowMonitor()
        mon._per_expert = _ExplodingStats()
        a, b = _capturing()
        with a, b:
            self.assertIsNone(mon.read_per_expert(reset=True))

    def test_percentile_math(self):
        dp = self.dp
        counts = [7, 8, 6, 9, 7, 8, 7, 6]
        self.assertEqual(dp._percentiles_from_counts(counts, (50, 99)), [7, 9])
        self.assertEqual(dp._percentiles_from_counts([], (50, 99)), [])
        self.assertEqual(dp._percentiles_from_counts([260, 2], (50, 99)), [2, 260])
        self.assertEqual(dp._percentiles_from_counts([5], (0, 100)), [5, 5])


class OutsideDrainTest(unittest.TestCase):
    def setUp(self):
        self.dp = _load_deepep()
        self.mon = self.dp._CapacityOverflowMonitor()

    def _patches(self, every, percentiles=(50, 99)):
        return [
            patch.object(self.dp, "_OVF_OUTSIDE_EVERY", every),
            patch.object(self.dp, "_OVF_ROUTER_PERCENTILES", percentiles),
            patch.object(self.dp, "_CAPACITY_OVERFLOW_INSTANCES", [self.mon]),
            patch.object(self.dp, "_OVF_OUTSIDE_CALLS", [0]),
        ]

    def test_drain_logs_every_k_calls_and_resets_window(self):
        dp = self.dp
        with patch.object(dp, "_OVF_OUTSIDE_EVERY", 5):
            self.mon.record(dp.torch.tensor([CAP + 2, 1]), CAP)
            self.mon.record(dp.torch.tensor([CAP + 2, 1]), CAP)
        patches = self._patches(3)
        with patches[0], patches[1], patches[2], patches[3]:
            with patch("torch.cuda.is_available", return_value=False):
                with patch.object(dp.logging, "info") as info:
                    # 2 of 3 calls: throttled, no readout, no reset.
                    dp.drain_capacity_overflow_outside_graph()
                    dp.drain_capacity_overflow_outside_graph()
                    self.assertEqual(info.call_count, 0)
                    # 3rd call: readout fires.
                    dp.drain_capacity_overflow_outside_graph()
                    self.assertEqual(info.call_count, 1)
                    line = _fmt_info_call(info)
                    for needle in (
                        "[CAPACITY_OVERFLOW]",
                        "ts=",
                        "dropped_routes=4",
                        "max_count_seen=%d" % (CAP + 2),
                        "active_experts=2",
                        "p50=2",
                        "p99=%d" % (2 * (CAP + 2)),
                    ):
                        self.assertIn(needle, line)
                    # Window was reset: next window reports only new records.
                    self.mon.record(dp.torch.tensor([CAP + 1, 0]), CAP)
                    dp.drain_capacity_overflow_outside_graph()
                    dp.drain_capacity_overflow_outside_graph()
                    dp.drain_capacity_overflow_outside_graph()
                    self.assertEqual(info.call_count, 2)
                    line2 = _fmt_info_call(info)
                    self.assertIn("dropped_routes=1", line2)
                    self.assertIn("active_experts=1", line2)

    def test_drain_is_silent_during_capture(self):
        dp = self.dp
        self.mon.record(dp.torch.tensor([CAP + 1, 1]), CAP)
        patches = self._patches(1)
        a, b = _capturing()
        with patches[0], patches[1], patches[2], patches[3], a, b:
            with patch.object(dp.logging, "info") as info:
                dp.drain_capacity_overflow_outside_graph()
                info.assert_not_called()
            self.assertEqual(dp._OVF_OUTSIDE_CALLS[0], 0)

    def test_drain_without_percentile_switch_has_no_pctl_fields(self):
        dp = self.dp
        self.mon.record(dp.torch.tensor([CAP + 1, 1]), CAP)
        patches = self._patches(1, percentiles=())
        with patches[0], patches[1], patches[2], patches[3]:
            with patch("torch.cuda.is_available", return_value=False):
                with patch.object(dp.logging, "info") as info:
                    dp.drain_capacity_overflow_outside_graph()
                    line = _fmt_info_call(info)
                    self.assertIn("dropped_routes=1", line)
                    self.assertNotIn("p50=", line)
                    self.assertNotIn("p99=", line)

    def test_drain_with_no_stats_is_silent(self):
        dp = self.dp
        patches = self._patches(1)
        with patches[0], patches[1], patches[2], patches[3]:
            with patch("torch.cuda.is_available", return_value=False):
                with patch.object(dp.logging, "info") as info:
                    dp.drain_capacity_overflow_outside_graph()
                    info.assert_not_called()


class EagerPercentileTest(unittest.TestCase):
    """p50/p99 appended to the existing eager router-stats line."""

    def setUp(self):
        self.dp = _load_deepep()
        self.mon = self.dp._CapacityOverflowMonitor()

    def _call(self, counts, pctl_every, percentiles=(50, 99)):
        dp = self.dp
        self.mon.record(counts, CAP)
        # NOTE: neither call counter is reset here — both advance across
        # _call()s exactly like they advance across eager forwards, which is
        # what the independent-throttle test observes.
        with patch.object(dp, "_OVF_LOG_EVERY", 1), patch.object(
            dp, "_OVF_ROUTER_PERCENTILES", percentiles
        ), patch.object(
            dp, "_OVF_ROUTER_PCTL_EVERY", pctl_every
        ), patch.object(
            dp, "_CAPACITY_OVERFLOW_INSTANCES", [self.mon]
        ), patch(
            "torch.cuda.is_available", return_value=False
        ):
            with patch.object(dp.logging, "info") as info:
                dp._maybe_log_capacity_overflow(CAP, counts)
        return info

    def test_percentiles_appended_when_enabled(self):
        counts = self.dp.torch.tensor([7, 8, 6, 9, 7, 8, 7, 6])
        info = self._call(counts, pctl_every=1)
        self.assertEqual(info.call_count, 1)
        line = _fmt_info_call(info)
        # Legacy fields stay untouched.
        for needle in (
            "[CAPACITY_OVERFLOW]",
            "n_local_experts=8",
            "routes_sum=58",
            "step_max=9",
            "active_experts=8",
            "step_min=6",
        ):
            self.assertIn(needle, line)
        # New percentile fields: same window (current step) as max/min/sum.
        self.assertIn("p50=7", line)
        self.assertIn("p99=9", line)

    def test_percentiles_throttled_independently(self):
        dp = self.dp
        counts = dp.torch.tensor([7, 8, 6, 9, 7, 8, 7, 6])
        info = self._call(counts, pctl_every=2)
        line = _fmt_info_call(info)
        # First hit is throttled (1 % 2 != 0): legacy line, no percentiles.
        self.assertNotIn("p50=", line)
        self.assertIn("n_local_experts=8", line)
        info2 = self._call(counts, pctl_every=2)
        line2 = _fmt_info_call(info2)
        # Second hit lands the percentile fields.
        self.assertIn("p50=7", line2)
        self.assertIn("p99=9", line2)

    def test_disabled_by_default_leaves_line_byte_identical(self):
        counts = self.dp.torch.tensor([7, 8, 6, 9, 7, 8, 7, 6])
        info = self._call(counts, pctl_every=0)
        line = _fmt_info_call(info)
        self.assertNotIn("p50=", line)
        self.assertNotIn("p99=", line)
        self.assertEqual(
            line,
            "[CAPACITY_OVERFLOW] calls=1 capacity=%d dropped_routes=0 "
            "max_count_seen=9 n_local_experts=8 routes_sum=58 step_max=9 "
            "step_argmax=3 active_experts=8 step_min=6" % CAP,
        )


class WarmupGuardTest(unittest.TestCase):
    """Warmup forwards run on dummy inputs before each capture
    (cuda_graph_runner.cc:1319-1343) and must not pollute the first readout
    window with inflated routing counts."""

    def setUp(self):
        self.dp = _load_deepep()

    def test_warmup_record_allocates_but_never_accumulates(self):
        dp = self.dp
        mon = dp._CapacityOverflowMonitor()
        with patch.object(dp, "_OVF_OUTSIDE_EVERY", 5):
            with patch.object(
                dp, "cuda_graph_warmup_forward_enabled", return_value=True
            ):
                mon.record(dp.torch.tensor([CAP + 7, 3]), CAP)
                mon.record(dp.torch.tensor([CAP + 7, 3]), CAP)
            # Allocation still happened during the eager warmup (outside any
            # capture): the graph will only ever record the adds, never a
            # zeros() fill that would wipe the accumulators each replay.
            self.assertIsNotNone(mon._stats)
            self.assertIsNotNone(mon._per_expert)
            # But nothing accumulated — the first window is unpolluted.
            with patch("torch.cuda.is_available", return_value=False):
                self.assertEqual(mon.read(reset=False), (0, 0))
                self.assertEqual(mon.read_per_expert(reset=False), [0, 0])

    def test_real_record_after_warmup_accumulates_normally(self):
        dp = self.dp
        mon = dp._CapacityOverflowMonitor()
        with patch.object(dp, "_OVF_OUTSIDE_EVERY", 5):
            with patch.object(
                dp, "cuda_graph_warmup_forward_enabled", return_value=True
            ):
                mon.record(dp.torch.tensor([CAP + 7, 3]), CAP)
            with patch.object(
                dp, "cuda_graph_warmup_forward_enabled", return_value=False
            ):
                mon.record(dp.torch.tensor([CAP + 2, 1]), CAP)
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(mon.read(reset=False), (2, CAP + 2))
            self.assertEqual(mon.read_per_expert(reset=False), [CAP + 2, 1])


class ConcurrentDrainTest(unittest.TestCase):
    """``prepare_cuda_graph`` may run on the engine main thread and an
    AsyncRunner worker thread at the same time (cuda_graph_runner.cc:690-696);
    the drain's count-then-maybe-readout must be atomic under that."""

    N_CALLS_PER_THREAD = 25
    THREADS = 2
    EVERY = 5
    ROUNDS = 8

    def test_concurrent_drains_count_atomically(self):
        dp = _load_deepep()
        mon = dp._CapacityOverflowMonitor()
        errors = []

        def worker(barrier):
            try:
                barrier.wait()  # maximise the interleave window
                for _ in range(self.N_CALLS_PER_THREAD):
                    dp.drain_capacity_overflow_outside_graph()
            except Exception as exc:  # pragma: no cover - asserted below
                errors.append(exc)

        for _ in range(self.ROUNDS):
            # A fresh window payload each round: the drain resets on readout,
            # so re-record before every round to keep asserting window data
            # is neither double-read nor lost.
            with patch.object(dp, "_OVF_OUTSIDE_EVERY", self.EVERY):
                mon.record(dp.torch.tensor([CAP + 2, 1]), CAP)  # dropped=2, max=130
            with patch.object(dp, "_OVF_OUTSIDE_EVERY", self.EVERY), patch.object(
                dp, "_OVF_ROUTER_PERCENTILES", ()
            ), patch.object(
                dp, "_CAPACITY_OVERFLOW_INSTANCES", [mon]
            ), patch.object(
                dp, "_OVF_OUTSIDE_CALLS", [0]
            ), patch(
                "torch.cuda.is_available", return_value=False
            ):
                with patch.object(dp.logging, "info") as info:
                    barrier = threading.Barrier(self.THREADS)
                    threads = [
                        threading.Thread(target=worker, args=(barrier,))
                        for _ in range(self.THREADS)
                    ]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()

                    # Assert INSIDE the patch scopes: outside them the patched
                    # call-counter list is restored to the module original.
                    self.assertEqual(
                        dp._OVF_OUTSIDE_CALLS[0],
                        self.N_CALLS_PER_THREAD * self.THREADS,
                        "call counter lost increments to a read-modify-write race",
                    )
                    self.assertEqual(
                        info.call_count,
                        self.N_CALLS_PER_THREAD * self.THREADS // self.EVERY,
                        "readout count drifted: a lost increment or a "
                        "double-triggered window slipped through the drain lock",
                    )
                    lines = [_fmt_info_call_like(c) for c in info.call_args_list]
                    dropped_total = sum(
                        int(l.split("dropped_routes=")[1].split()[0])
                        for l in lines
                    )
                    self.assertEqual(
                        dropped_total, 2, "window data was double-read or lost"
                    )
                    self.assertFalse(
                        errors, f"worker threads raised: {errors}"
                    )


class MultiInstanceAggregationTest(unittest.TestCase):
    """Each monitor instance is one MoE layer; drain aggregates them all."""

    def setUp(self):
        self.dp = _load_deepep()

    def _drain_once(self, monitors):
        with patch.object(
            self.dp, "_OVF_OUTSIDE_EVERY", 1
        ), patch.object(
            self.dp, "_OVF_ROUTER_PERCENTILES", (50, 99)
        ), patch.object(
            self.dp, "_CAPACITY_OVERFLOW_INSTANCES", list(monitors)
        ), patch.object(
            self.dp, "_OVF_OUTSIDE_CALLS", [0]
        ), patch(
            "torch.cuda.is_available", return_value=False
        ):
            with patch.object(self.dp.logging, "info") as info:
                self.dp.drain_capacity_overflow_outside_graph()
        self.assertEqual(info.call_count, 1)
        return _fmt_info_call(info)

    def test_two_instances_sum_dropped_and_max_percentiles(self):
        dp = self.dp
        mon1 = dp._CapacityOverflowMonitor()
        mon2 = dp._CapacityOverflowMonitor()
        with patch.object(dp, "_OVF_OUTSIDE_EVERY", 5):
            mon1.record(dp.torch.tensor([CAP + 3, 1, 0]), CAP)  # dropped=3
            mon2.record(dp.torch.tensor([CAP + 1, 5, 9, 0, 0]), CAP)  # dropped=1
        line = self._drain_once([mon1, mon2])
        # dropped is the SUM across instances.
        self.assertIn("dropped_routes=4", line)
        self.assertIn("max_count_seen=%d" % (CAP + 3), line)
        # Per-expert aggregation is an elementwise max across instances
        # ([CAP+3,1,0] vs [CAP+1,5,9,0,0] -> [131,5,9,0,0]; the longer vector
        # exercises the extend/align branch): sorted [0,0,5,9,131] gives
        # p50=5 and p99=131, active_experts=3.
        self.assertIn("p50=5", line)
        self.assertIn("p99=%d" % (CAP + 3), line)
        self.assertIn("active_experts=3", line)

    def test_alignment_when_first_iterated_instance_is_shorter(self):
        dp = self.dp
        mon1 = dp._CapacityOverflowMonitor()
        mon2 = dp._CapacityOverflowMonitor()
        with patch.object(dp, "_OVF_OUTSIDE_EVERY", 5):
            mon1.record(dp.torch.tensor([CAP + 3, 1, 0]), CAP)
            mon2.record(dp.torch.tensor([CAP + 1, 5, 9, 0, 0]), CAP)
        # Iterate the LONGER instance first: no extend branch, pure
        # elementwise max over the overlapping prefix.
        line = self._drain_once([mon2, mon1])
        self.assertIn("dropped_routes=4", line)
        self.assertIn("p50=5", line)
        self.assertIn("p99=%d" % (CAP + 3), line)


class ReplayHookWiringTest(unittest.TestCase):
    """Source contract: both fmha impls call the drain between replays."""

    @staticmethod
    def _prepare_cuda_graph_last_stmt(path):
        """Last TOP-LEVEL statement of ``prepare_cuda_graph`` in ``path``.

        Being a direct child of the FunctionDef body means the statement is
        not nested inside any ``if``/``try``/``with`` block — a drain call
        parked in a dead branch would not qualify.
        """
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "prepare_cuda_graph"
            ):
                return node.body[-1]
        raise AssertionError(f"prepare_cuda_graph not found in {path}")

    def _assert_tail_is_drain_call(self, path):
        last = self._prepare_cuda_graph_last_stmt(path)
        self.assertIsInstance(
            last,
            ast.Expr,
            f"{path.name}: last top-level statement of prepare_cuda_graph "
            f"must be an expression (got {type(last).__name__})",
        )
        self.assertIsInstance(last.value, ast.Call)
        func = last.value.func
        self.assertIsInstance(func, ast.Name)
        self.assertEqual(
            func.id,
            "drain_capacity_overflow_outside_graph",
            f"{path.name}: drain call is not the final top-level statement "
            f"of prepare_cuda_graph",
        )

    def test_bf16_impl_drains_between_replays(self):
        self._assert_tail_is_drain_call(BF16_IMPL_PATH)

    def test_fp8_impl_drains_between_replays(self):
        self._assert_tail_is_drain_call(FP8_IMPL_PATH)

    def test_tail_check_rejects_dead_branch_wiring(self):
        """Meta-test: the tail assertion really does reject ``if False:``."""
        snippet = (
            "def prepare_cuda_graph(self, attn_inputs):\n"
            "    self.prepare(attn_inputs)\n"
            "    if False:\n"
            "        drain_capacity_overflow_outside_graph()\n"
        )
        tree = ast.parse(snippet)
        last = tree.body[0].body[-1]
        self.assertIsInstance(last, ast.If)  # not an Expr -> would fail above

    def test_deepep_exposes_the_drain_and_per_expert_reader(self):
        tree = ast.parse(DEEPEP_PATH.read_text())
        names = {
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("drain_capacity_overflow_outside_graph", names)
        self.assertIn("read_capacity_overflow_stats", names)  # testbench handle
        monitor = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "_CapacityOverflowMonitor"
        )
        methods = {
            node.name
            for node in monitor.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("read_per_expert", methods)


if __name__ == "__main__":
    unittest.main(verbosity=2)
