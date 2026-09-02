"""Source-only behaviour tests for the P3.1 capture-safe shared-expert overlap.

This host has no torch, so the file loads ``shared_expert.py`` with a minimal
orchestration-recording torch stub plus stub siblings (``_profiler`` /
``warmup_sync``) and the real ``runtime_config.py``, following the repo's
sanctioned monkeypatch-isolation style (see
test_capacity_overflow_outside_graph.py).  The fake CUDA world records every
orchestration op -- stream creation, both ``wait_stream`` dependency edges,
which stream the shared expert ran on -- into a log, so the fork/join contract
is asserted op-by-op instead of trusting comments.

Covers task #14 (ROADMAP P3.1, DSV4_MOE_SHARED_EXPERT_OVERLAP):
* default off: sequential mode + the legacy capture veto stay byte-identical;
* switch on + eager: the legacy overlap orchestration, unchanged;
* switch on + capture: fork/join recorded, stream cache hit, no stream
  creation and no ``record_stream`` inside the capture;
* capture with a cold stream cache: exactly one warning + serial fallback;
* AST wiring: fork edge before the side-stream block, join edge before the
  output is handed back, capture veto gated on the switch.

Run: ``python3 test_shared_expert_overlap_capture.py`` (no bazel).

Stub fidelity boundary: the stub does NOT model cudaEvent dependencies or
the caching allocator -- stream semantics here are pure Python bookkeeping.
The in-graph dependency reality (both wait_stream edges really constraining
kernel execution order inside a captured graph) is verified EXCLUSIVELY by
the CUDA-gated tests in test_shared_expert_executor.py.
"""

import ast
import contextlib
import importlib.util
import logging
import os
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

_DSV4_DIR = Path(__file__).resolve().parents[1]
RC_PATH = _DSV4_DIR / "runtime_config.py"
SE_PATH = _DSV4_DIR / "moe" / "shared_expert.py"

_PKG = "rtp_llm.models_py.modules.dsv4"
_MOE = _PKG + ".moe"

_SWITCH = "DSV4_MOE_SHARED_EXPERT_OVERLAP"
_TOUCHED_ENV = (
    _SWITCH,
    "DSV4_SHARED_EXPERT_MODE",
    "DSV4_MOE_STRICT_FUSED",
    "DSV4_SHARED_EXPERT_BF16_PATH",
    "DSV4_SHARED_EXPERT_STREAM_TOKEN_THRESHOLD",
    "MOEDBG",
    "RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD",
)


# ---------------------------------------------------------------------------
# Fake CUDA world: devices, tensors, streams, current-stream stack.
# ---------------------------------------------------------------------------


class _FakeDevice:
    def __init__(self, index=0):
        self.type = "cuda"
        self.index = index

    def __repr__(self):
        return f"cuda:{self.index}"


class _FakeTensor:
    """Exactly the tensor surface the executor path touches."""

    def __init__(self, tokens=33, device=None, is_cuda=True):
        self.shape = (tokens, 128)
        self.device = device if device is not None else _FakeDevice(0)
        self.is_cuda = is_cuda
        self.recorded_streams = []

    def record_stream(self, stream):
        ORCH.append(("record_stream", stream, None))
        self.recorded_streams.append(stream)

    def float(self):
        return self


class _FakeStream:
    def __init__(self, device=None):
        self.device = device

    def wait_stream(self, other):
        # (who waits, waits-for-whom)
        ORCH.append(("wait", self, other))


ORCH = []
_MAIN = _FakeStream(_FakeDevice(0))
_STREAM_STACK = []


def _current_stream(device=None):
    return _STREAM_STACK[-1] if _STREAM_STACK else _MAIN


def _make_stream(device=None):
    stream = _FakeStream(device)
    ORCH.append(("create", stream, None))
    return stream


@contextmanager
def _stream_cm(stream):
    _STREAM_STACK.append(stream)
    try:
        yield
    finally:
        _STREAM_STACK.pop()


class _StubNNModule:
    def parameters(self, recurse=True):
        return []

    def buffers(self, recurse=True):
        return []

    def modules(self):
        return [self]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


_RUN_OUT = _FakeTensor()


class _Shared(_StubNNModule):
    """Shared expert stand-in: logs the stream it ran on, returns a sentinel."""

    def forward(self, x):
        ORCH.append(("run", _current_stream(), x))
        return _RUN_OUT


class _SharedWithCudaWeight(_Shared):
    def __init__(self):
        self.weight = _FakeTensor(1)


def _install_stub_environment():
    # Never trample a real torch that an outer harness (e.g. a bazel py_test
    # with the full runtime on the path) may already have loaded.
    real = sys.modules.get("torch")
    if real is not None and not getattr(real, "__t14_stub__", False):
        raise unittest.SkipTest("real torch already loaded; stub install refused")
    if getattr(sys.modules.get("torch"), "__t14_stub__", False):
        return
    torch_mod = types.ModuleType("torch")
    torch_mod.__t14_stub__ = True
    torch_mod.Tensor = _FakeTensor
    # Real torch semantics: torch.device("cuda") has index None; only the
    # two-arg form pins the index.  Covers _normalize_cuda_device's None
    # branch, which previously had zero stub coverage.
    torch_mod.device = lambda *a, **k: _FakeDevice(a[1] if len(a) > 1 else None)
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.__t14_stub__ = True
    nn_mod.Module = _StubNNModule
    torch_mod.nn = nn_mod
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.__t14_stub__ = True
    cuda_mod.is_available = lambda: True
    cuda_mod.is_current_stream_capturing = lambda: False
    cuda_mod.current_device = lambda: 0
    cuda_mod.Stream = _make_stream
    cuda_mod.stream = _stream_cm
    cuda_mod.current_stream = _current_stream
    torch_mod.cuda = cuda_mod
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.cuda"] = cuda_mod

    # Package chain + shared_expert's sibling imports, as inert stubs.  The
    # real chain (models_py.modules.__init__ etc.) needs the serving container.
    for name in (
        "rtp_llm",
        "rtp_llm.models_py",
        "rtp_llm.models_py.modules",
        _PKG,
        _MOE,
    ):
        mod = types.ModuleType(name)
        mod.__t14_stub__ = True
        sys.modules[name] = mod

    prof = types.ModuleType(_PKG + "._profiler")
    prof.__t14_stub__ = True

    @contextmanager
    def _record_function_range(name):
        yield

    prof.record_function_range = _record_function_range
    warm = types.ModuleType(_MOE + ".warmup_sync")
    warm.__t14_stub__ = True
    warm.cuda_graph_warmup_forward_enabled = lambda: False
    sys.modules[prof.__name__] = prof
    sys.modules[warm.__name__] = warm
    sys.modules[_PKG]._profiler = prof
    sys.modules[_MOE].warmup_sync = warm


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_se(env=None):
    """Fresh shared_expert (and runtime_config) with the given env snapshot."""
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
        se = _load_module(SE_PATH, _MOE + ".shared_expert")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return se


@contextmanager
def _env(**kv):
    """Env snapshot around CALLS (mode/strict/threshold are read per call)."""
    saved = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _capturing(flag=True):
    return patch("torch.cuda.is_current_stream_capturing", lambda: flag)


def _ops():
    """ORCH normalized to (op, roles...) so different loads compare equal."""

    def role(stream):
        return "main" if stream is _MAIN else "side"

    out = []
    for entry in ORCH:
        op = entry[0]
        if op == "create":
            out.append(("create", role(entry[1])))
        elif op == "record_stream":
            out.append(("record_stream", role(entry[1])))
        elif op == "wait":
            out.append(("wait", role(entry[1]), role(entry[2])))
        elif op == "run":
            out.append(("run", role(entry[1])))
        elif op == "consume":
            out.append(("consume",))
        else:
            raise AssertionError(f"unknown op {entry!r}")
    return out


_EAGER_OVERLAP_OPS = [
    ("create", "side"),
    ("record_stream", "side"),
    ("wait", "side", "main"),  # EDGE 1 (fork): input dependency
    ("run", "side"),
    ("wait", "main", "side"),  # EDGE 2 (join): output dependency
]


class _SeTestCase(unittest.TestCase):
    def setUp(self):
        ORCH.clear()
        _STREAM_STACK.clear()


# ---------------------------------------------------------------------------
# Task 3a: default off == byte-identical legacy behaviour.
# ---------------------------------------------------------------------------


class SwitchDefaultsOffTest(_SeTestCase):
    def test_switch_defaults_to_false_and_audits(self):
        with self.assertLogs(level=logging.INFO) as logs:
            se = _load_se()
        self.assertFalse(se._SHARED_EXPERT_OVERLAP_ENABLED)
        audited = [r for r in logs.output if "[DSV4_CONFIG]" in r]
        self.assertTrue(
            any(_SWITCH in r and "False" in r for r in audited),
            f"missing audit line for {_SWITCH}: {audited}",
        )

    def test_switch_bool_parsing(self):
        for val in ("1", "on", "true", "yes"):
            self.assertTrue(_load_se({_SWITCH: val})._SHARED_EXPERT_OVERLAP_ENABLED, val)
        for val in ("0", "off", "false", "no"):
            self.assertFalse(_load_se({_SWITCH: val})._SHARED_EXPERT_OVERLAP_ENABLED, val)

    def test_invalid_switch_value_fails_loud_at_import(self):
        with self.assertRaises(ValueError) as ctx:
            _load_se({_SWITCH: "sometimes"})
        self.assertIn(_SWITCH, str(ctx.exception))

    def test_default_mode_is_sequential_executor(self):
        se = _load_se()
        self.assertIsInstance(
            se.get_shared_expert_executor(), se.SequentialSharedExpertExecutor
        )

    def test_invalid_legacy_mode_still_fails_loud(self):
        se = _load_se()
        with _env(DSV4_SHARED_EXPERT_MODE="bogus"):
            with self.assertRaisesRegex(ValueError, "invalid DSV4_SHARED_EXPERT_MODE"):
                se.get_shared_expert_executor()

    def test_capture_veto_stays_with_switch_off(self):
        se = _load_se()
        executor = se.OverlapSharedExpertExecutor()
        x = _FakeTensor()
        with _env(DSV4_MOE_STRICT_FUSED="0"), _capturing():
            self.assertFalse(executor._can_overlap(x))
            executor.start(_Shared(), x)
            self.assertIsNone(executor._active_stream)
            got = executor.finish()
        # Byte-identical legacy: zero stream ops, shared expert inline on the
        # main stream, output handed straight through.
        self.assertEqual(_ops(), [("run", "main")])
        self.assertIs(got, _RUN_OUT)

    def test_eager_overlap_mode_still_overlaps_with_switch_off(self):
        se = _load_se({"DSV4_SHARED_EXPERT_MODE": "overlap"})
        executor = se.OverlapSharedExpertExecutor()
        with _env(DSV4_MOE_STRICT_FUSED="0"):
            executor.start(_Shared(), _FakeTensor())
            self.assertIsNotNone(executor._active_stream)
            got = executor.finish()
        self.assertEqual(_ops(), _EAGER_OVERLAP_OPS)
        self.assertIs(got, _RUN_OUT)


# ---------------------------------------------------------------------------
# Task 3b: switch on + eager == the legacy overlap orchestration.
# ---------------------------------------------------------------------------


class SwitchOnEagerTest(_SeTestCase):
    def test_switch_selects_overlap_executor(self):
        se = _load_se({_SWITCH: "1"})
        self.assertIsInstance(
            se.get_shared_expert_executor(), se.OverlapSharedExpertExecutor
        )

    def test_switch_overrides_explicit_sequential_mode(self):
        se = _load_se({_SWITCH: "1", "DSV4_SHARED_EXPERT_MODE": "sequential"})
        self.assertIsInstance(
            se.get_shared_expert_executor(), se.OverlapSharedExpertExecutor
        )

    def test_switch_keeps_auto_overlap_and_legacy_fail_loud(self):
        se = _load_se({_SWITCH: "1"})
        for mode in ("auto", "overlap"):
            with _env(DSV4_SHARED_EXPERT_MODE=mode):
                self.assertIsInstance(
                    se.get_shared_expert_executor(), se.OverlapSharedExpertExecutor
                )
        with _env(DSV4_SHARED_EXPERT_MODE="bogus"):
            with self.assertRaisesRegex(ValueError, "invalid DSV4_SHARED_EXPERT_MODE"):
                se.get_shared_expert_executor()

    def test_switch_on_eager_matches_legacy_overlap_orchestration(self):
        legacy = _load_se({"DSV4_SHARED_EXPERT_MODE": "overlap"})
        modern = _load_se({_SWITCH: "1"})
        with _env(DSV4_MOE_STRICT_FUSED="0"):
            for se in (legacy, modern):
                ORCH.clear()
                executor = se.OverlapSharedExpertExecutor()
                executor.start(_Shared(), _FakeTensor())
                self.assertIsNotNone(executor._active_stream)
                self.assertIs(executor.finish(), _RUN_OUT)
                self.assertEqual(_ops(), _EAGER_OVERLAP_OPS)

    def test_eager_stream_cache_is_reused_across_executors(self):
        se = _load_se({_SWITCH: "1"})
        first = se.OverlapSharedExpertExecutor()
        second = se.OverlapSharedExpertExecutor()
        with _env(DSV4_MOE_STRICT_FUSED="0"):
            first.start(_Shared(), _FakeTensor())
            stream = first._active_stream
            first.finish()
            second.start(_Shared(), _FakeTensor())
            self.assertIs(second._active_stream, stream)
            second.finish()
        self.assertEqual([op for op in _ops() if op[0] == "create"], [("create", "side")])


# ---------------------------------------------------------------------------
# Task 3c: switch on + capture: the fork/join is recorded into the graph.
# ---------------------------------------------------------------------------


class SwitchOnCaptureTest(_SeTestCase):
    def _prepared_executor(self, se):
        executor = se.OverlapSharedExpertExecutor()
        executor.prepare(_SharedWithCudaWeight())
        self.assertIn(0, se._SHARED_EXPERT_STREAM_CACHE)
        ORCH.clear()  # drop the prepare-time creation from the assertions
        return executor, se._SHARED_EXPERT_STREAM_CACHE[0]

    def test_can_overlap_true_during_capture_with_switch_on(self):
        se = _load_se({_SWITCH: "1"})
        executor = se.OverlapSharedExpertExecutor()
        with _capturing():
            self.assertTrue(executor._can_overlap(_FakeTensor()))

    def test_capture_records_fork_join_with_cached_stream(self):
        se = _load_se({_SWITCH: "1"})
        executor, stream = self._prepared_executor(se)
        with _env(DSV4_MOE_STRICT_FUSED="0"), _capturing():
            executor.start(_Shared(), _FakeTensor())
            self.assertIs(executor._active_stream, stream)  # cache HIT
            got = executor.finish()
        self.assertIs(got, _RUN_OUT)
        self.assertEqual(
            _ops(),
            [
                ("wait", "side", "main"),  # EDGE 1 (fork): input dependency
                ("run", "side"),  # shared expert on the side stream
                ("wait", "main", "side"),  # EDGE 2 (join): output dependency
            ],
        )

    def test_capture_never_creates_a_stream_or_records_input(self):
        se = _load_se({_SWITCH: "1"})
        executor, _stream = self._prepared_executor(se)
        with _env(DSV4_MOE_STRICT_FUSED="0"), _capturing():
            executor.start(_Shared(), _FakeTensor())
            executor.finish()
        ops = _ops()
        self.assertFalse(any(op[0] == "create" for op in ops))
        self.assertFalse(any(op[0] == "record_stream" for op in ops))

    def test_join_edge_precedes_first_consumer(self):
        # moe_layer consumes the shared output only AFTER finish() returns
        # (combine_routed_and_shared); the join must therefore already be
        # recorded by the time finish() hands the tensor back.
        se = _load_se({_SWITCH: "1"})
        executor, _stream = self._prepared_executor(se)
        with _env(DSV4_MOE_STRICT_FUSED="0"), _capturing():
            executor.start(_Shared(), _FakeTensor())
            shared = executor.finish()
        ORCH.append(("consume", None, shared))  # combine_routed_and_shared stand-in
        ops = _ops()
        self.assertLess(ops.index(("wait", "main", "side")), ops.index(("consume",)))

    def test_first_in_graph_fork_join_logs_one_info(self):
        # Arm-acceptance evidence: the FIRST capture that records the
        # cross-stream fork/join announces itself exactly once per process.
        se = _load_se({_SWITCH: "1"})
        executor, _stream = self._prepared_executor(se)
        with _env(DSV4_MOE_STRICT_FUSED="0"), _capturing():
            with self.assertLogs(level=logging.INFO) as logs:
                executor.start(_Shared(), _FakeTensor())
                executor.finish()
                executor.start(_Shared(), _FakeTensor())  # second: silent
                executor.finish()
        forked = [
            r
            for r in logs.records
            if "captured in-graph fork/join" in r.getMessage()
        ]
        self.assertEqual(len(forked), 1)
        self.assertIn("cuda:0", forked[0].getMessage())

    def test_warmup_forward_veto_still_applies_with_switch_on(self):
        # The scoped RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD=1 eager iterations that
        # C++ runs right before each capture may rendezvous across ranks; the
        # shared expert stays inline for those (veto untouched by P3.1).
        se = _load_se({_SWITCH: "1"})
        executor = se.OverlapSharedExpertExecutor()
        with _env(DSV4_MOE_STRICT_FUSED="0"), patch.object(
            se, "cuda_graph_warmup_forward_enabled", lambda: True
        ):
            executor.start(_Shared(), _FakeTensor())
            self.assertIsNone(executor._active_stream)
            executor.finish()
        self.assertEqual(_ops(), [("run", "main")])


# ---------------------------------------------------------------------------
# Task 3d: capture with a cold stream cache: warn once + serial fallback.
# ---------------------------------------------------------------------------


class CaptureColdCacheFallbackTest(_SeTestCase):
    def test_capture_with_cold_cache_warns_once_and_serializes(self):
        se = _load_se({_SWITCH: "1"})
        self.assertEqual(se._SHARED_EXPERT_STREAM_CACHE, {})  # cold by construction
        executor = se.OverlapSharedExpertExecutor()
        with _env(DSV4_MOE_STRICT_FUSED="0"), _capturing():
            with self.assertLogs(level=logging.WARNING) as logs:
                executor.start(_Shared(), _FakeTensor())  # first capture: warns
                executor.finish()
                executor.start(_Shared(), _FakeTensor())  # second: silent
                executor.finish()
        warned = [r for r in logs.output if "[DSV4_SHARED_EXPERT_OVERLAP]" in r]
        self.assertEqual(
            len(warned), 1, f"expected exactly one warning, got: {logs.output}"
        )
        self.assertIn("INERT", warned[0])
        # Serial fallback: shared expert inline, no stream created, no edges.
        self.assertEqual(_ops(), [("run", "main"), ("run", "main")])

    def test_cold_cache_miss_logs_progress_info_every_ten_captures(self):
        # Observability fix: a persistent cold cache must not look like one
        # warning followed by silence -- every 10th miss logs an INFO with
        # the running count.
        se = _load_se({_SWITCH: "1"})
        executor = se.OverlapSharedExpertExecutor()
        with _env(DSV4_MOE_STRICT_FUSED="0"), _capturing():
            with self.assertLogs(level=logging.INFO) as logs:
                for _ in range(12):
                    executor.start(_Shared(), _FakeTensor())
                    executor.finish()
        warnings = [
            r
            for r in logs.records
            if r.levelname == "WARNING"
            and "[DSV4_SHARED_EXPERT_OVERLAP]" in r.getMessage()
        ]
        infos = [
            r
            for r in logs.records
            if r.levelname == "INFO" and "miss count" in r.getMessage()
        ]
        self.assertEqual(len(warnings), 1)  # first miss only
        self.assertEqual(len(infos), 1)  # the 10th miss only (11th/12th silent)
        self.assertIn("count=10", infos[0].getMessage())

    def test_cold_cache_fallback_output_is_still_correct(self):
        se = _load_se({_SWITCH: "1"})
        executor = se.OverlapSharedExpertExecutor()
        with _env(DSV4_MOE_STRICT_FUSED="0"), _capturing():
            with self.assertLogs(level=logging.WARNING):
                executor.start(_Shared(), _FakeTensor())
                got = executor.finish()
        self.assertIs(got, _RUN_OUT)


# ---------------------------------------------------------------------------
# prepare(): the capture-safety precondition (stream precreation).
# ---------------------------------------------------------------------------


class PreparePrecreateTest(_SeTestCase):
    def test_prepare_precreates_stream_from_cuda_weight(self):
        se = _load_se({_SWITCH: "1"})
        executor = se.OverlapSharedExpertExecutor()
        executor.prepare(_SharedWithCudaWeight())
        self.assertIn(0, se._SHARED_EXPERT_STREAM_CACHE)
        self.assertEqual([op[0] for op in _ops()], ["create"])

    def test_prepare_falls_back_to_current_device_with_switch_on(self):
        # Weights not on CUDA at construction (unusual load order): the P3.1
        # switch still precreates on the current device so a capture never
        # meets a cold cache -- and says WHICH card it built the stream on,
        # once per process even though prepare runs per MoE layer.
        se = _load_se({_SWITCH: "1"})
        executor = se.OverlapSharedExpertExecutor()
        with self.assertLogs(level=logging.INFO) as logs:
            executor.prepare(_Shared())
            executor.prepare(_Shared())  # throttled: one line per process
        self.assertIn(0, se._SHARED_EXPERT_STREAM_CACHE)
        fallback = [
            r
            for r in logs.records
            if "precreating the side stream" in r.getMessage()
        ]
        self.assertEqual(len(fallback), 1)
        self.assertIn("cuda:0", fallback[0].getMessage())

    def test_normalize_device_without_index_uses_current_device(self):
        # torch.device("cuda") carries index=None (the stub now mirrors real
        # torch): _normalize_cuda_device must resolve it via
        # torch.cuda.current_device().
        se = _load_se()
        no_index = sys.modules["torch"].device("cuda")
        self.assertIsNone(no_index.index)
        normalized = se._normalize_cuda_device(no_index)
        self.assertEqual(normalized.index, 0)  # stub current_device()

    def test_prepare_stays_lazy_with_switch_off(self):
        se = _load_se()
        executor = se.OverlapSharedExpertExecutor()
        executor.prepare(_Shared())
        self.assertEqual(se._SHARED_EXPERT_STREAM_CACHE, {})


# ---------------------------------------------------------------------------
# Review fixes: _mode() override audit + finish() error propagation.
# ---------------------------------------------------------------------------


class ModeOverrideAuditTest(_SeTestCase):
    def test_explicit_sequential_override_logs_exactly_once(self):
        # An EXPLICIT DSV4_SHARED_EXPERT_MODE=sequential must not be silently
        # rewritten by the P3.1 switch -- one audit line per process.
        se = _load_se({_SWITCH: "1"})
        with self.assertLogs(level=logging.INFO) as logs:
            with _env(DSV4_SHARED_EXPERT_MODE="sequential"):
                self.assertEqual(se._mode(), "overlap")
                self.assertEqual(se._mode(), "overlap")  # throttled
        lines = [
            r.getMessage()
            for r in logs.records
            if "[DSV4_CONFIG]" in r.getMessage()
        ]
        self.assertEqual(len(lines), 1)
        self.assertIn("DSV4_SHARED_EXPERT_MODE=sequential", lines[0])
        self.assertIn("DSV4_MOE_SHARED_EXPERT_OVERLAP=1", lines[0])

    def test_unset_sequential_override_is_silent(self):
        # No explicit legacy env var -> no override line (the default flip is
        # already covered by the get_switch [DSV4_CONFIG] audit at import).
        se = _load_se({_SWITCH: "1"})
        self.assertEqual(se._mode(), "overlap")
        self.assertFalse(se._SHARED_EXPERT_MODE_OVERRIDE_LOGGED)

    def test_explicit_non_sequential_mode_is_not_overridden(self):
        se = _load_se({_SWITCH: "1"})
        with _env(DSV4_SHARED_EXPERT_MODE="overlap"):
            self.assertEqual(se._mode(), "overlap")
        self.assertFalse(se._SHARED_EXPERT_MODE_OVERRIDE_LOGGED)


class FinishErrorPropagationTest(_SeTestCase):
    def _boom(self):
        class _BoomShared(_Shared):
            def forward(self, x):
                raise RuntimeError("shared expert kernel exploded")

        return _BoomShared()

    def test_overlap_finish_after_failed_start_raises_runtimeerror(self):
        # Review finding: start() raising on the side stream (fast-path dim
        # mismatch, OOM, ...) must not be masked when moe_layer's except
        # blocks call finish() -- the old bare assert turned the original
        # exception into an AssertionError (or None under python -O).
        se = _load_se({_SWITCH: "1"})
        executor = se.OverlapSharedExpertExecutor()
        with _env(DSV4_MOE_STRICT_FUSED="0"):
            with self.assertRaisesRegex(RuntimeError, "exploded"):
                executor.start(self._boom(), _FakeTensor())
            with self.assertRaisesRegex(RuntimeError, "start"):
                executor.finish()

    def test_sequential_finish_after_failed_start_same_contract(self):
        se = _load_se()
        executor = se.SequentialSharedExpertExecutor()
        with _env(DSV4_MOE_STRICT_FUSED="0"):
            with self.assertRaisesRegex(RuntimeError, "exploded"):
                executor.start(self._boom(), _FakeTensor())
            with self.assertRaisesRegex(RuntimeError, "start"):
                executor.finish()


# ---------------------------------------------------------------------------
# Task 3e: AST wiring: the edges live on the key path, not in dead branches.
# ---------------------------------------------------------------------------


class SourceWiringTest(unittest.TestCase):
    @staticmethod
    def _tree():
        return ast.parse(SE_PATH.read_text())

    @classmethod
    def _class(cls, tree, name):
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == name:
                return node
        raise AssertionError(f"class {name} not found in shared_expert.py")

    @classmethod
    def _method(cls, cls_node, name):
        for node in cls_node.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"method {name} not found")

    @staticmethod
    def _mentions(node, name):
        return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))

    @staticmethod
    def _capture_veto_ifs(method_node):
        """``if`` statements inside the method that test stream capture."""

        def calls_capturing(test):
            return any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "is_current_stream_capturing"
                for n in ast.walk(test)
            )

        return [
            node
            for node in ast.walk(method_node)
            if isinstance(node, ast.If) and calls_capturing(node.test)
        ]

    def test_switch_registered_through_runtime_config(self):
        calls = [
            node
            for node in ast.walk(self._tree())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "get_switch"
        ]
        self.assertEqual(len(calls), 1, "expected exactly one get_switch call")
        args = calls[0].args
        self.assertEqual(args[0].value, _SWITCH)
        self.assertEqual(args[1].value, False)  # default 0 == legacy behaviour
        self.assertEqual(args[2].id, "parse_bool")  # fail-loud validator

    def test_can_overlap_capture_veto_is_switch_gated(self):
        # Every `if is_current_stream_capturing(...)` inside _can_overlap must
        # also reference the P3.1 switch: a bare veto would silently kill the
        # overlap inside every captured graph again.
        cls = self._class(self._tree(), "OverlapSharedExpertExecutor")
        method = self._method(cls, "_can_overlap")
        veto_ifs = self._capture_veto_ifs(method)
        self.assertTrue(veto_ifs, "capture veto disappeared from _can_overlap?")
        for node in veto_ifs:
            self.assertTrue(
                self._mentions(node.test, "_SHARED_EXPERT_OVERLAP_ENABLED"),
                "capture veto in _can_overlap is not gated on the P3.1 switch",
            )

    def test_start_fork_edge_precedes_side_stream_block(self):
        # Structural assertion: pins the statement layout of start() (fork
        # edge before the side-stream with-block, _active_stream assignment
        # last).  Refactoring start()'s statement order requires updating
        # this test in lockstep -- it guards against the edges migrating into
        # a dead/conditional branch, not against logic bugs.
        cls = self._class(self._tree(), "OverlapSharedExpertExecutor")
        body = self._method(cls, "start").body

        def is_wait_stream(stmt):
            return (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "wait_stream"
            )

        def is_stream_with(stmt):
            return (
                isinstance(stmt, ast.With)
                and isinstance(stmt.items[0].context_expr, ast.Call)
                and isinstance(stmt.items[0].context_expr.func, ast.Attribute)
                and stmt.items[0].context_expr.func.attr == "stream"
            )

        fork_idx = next(i for i, s in enumerate(body) if is_wait_stream(s))
        with_idx = next(i for i, s in enumerate(body) if is_stream_with(s))
        self.assertLess(
            fork_idx,
            with_idx,
            "fork edge (main -> side input dependency) must precede the "
            "side-stream with-block",
        )
        # Both are direct children of start's body: neither sits inside an
        # if/try dead branch.
        with_stmt = body[with_idx]
        self.assertTrue(
            any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "_run_shared_expert"
                for n in ast.walk(with_stmt)
            ),
            "shared expert call is not inside the side-stream with-block",
        )
        # The join handle registration is the LAST statement of start.
        last = body[-1]
        self.assertIsInstance(last, ast.Assign)
        self.assertEqual(getattr(last.targets[0], "attr", None), "_active_stream")
        self.assertIsInstance(last.value, ast.Name)
        self.assertEqual(last.value.id, "stream")

    def test_finish_join_edge_precedes_handing_back_output(self):
        # Structural assertion: pins the statement layout of finish() (join
        # edge before out=self._out before return).  Refactoring finish()'s
        # statement order requires updating this test in lockstep.
        cls = self._class(self._tree(), "OverlapSharedExpertExecutor")
        body = self._method(cls, "finish").body
        join_idx = next(
            i
            for i, s in enumerate(body)
            if isinstance(s, ast.If)
            and any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "wait_stream"
                for n in ast.walk(s)
            )
        )
        out_assign_idx = next(
            i
            for i, s in enumerate(body)
            if isinstance(s, ast.Assign)
            and isinstance(s.targets[0], ast.Name)
            and s.targets[0].id == "out"
        )
        ret_idx = next(i for i, s in enumerate(body) if isinstance(s, ast.Return))
        self.assertLess(join_idx, out_assign_idx, "join edge must precede out=self._out")
        self.assertLess(out_assign_idx, ret_idx)

    def test_mode_override_returns_overlap(self):
        mode = next(
            node
            for node in self._tree().body
            if isinstance(node, ast.FunctionDef) and node.name == "_mode"
        )
        found = False
        for node in ast.walk(mode):
            if isinstance(node, ast.If) and self._mentions(
                node.test, "_SHARED_EXPERT_OVERLAP_ENABLED"
            ):
                if any(
                    isinstance(n, ast.Constant) and n.value == "overlap"
                    for n in ast.walk(node)
                ):
                    found = True
        self.assertTrue(
            found, "_mode never flips sequential -> overlap on the P3.1 switch"
        )

    def test_bare_veto_is_rejected_by_the_gate(self):
        # Meta-test: the veto gate really rejects a bare capture veto (the
        # pre-P3.1 shape) rather than passing vacuously.
        snippet = (
            "def _can_overlap(self, x):\n"
            "    if torch.cuda.is_current_stream_capturing():\n"
            "        return False\n"
            "    return True\n"
        )
        method = ast.parse(snippet).body[0]
        veto_ifs = self._capture_veto_ifs(method)
        self.assertTrue(veto_ifs)
        gated = [
            self._mentions(node.test, "_SHARED_EXPERT_OVERLAP_ENABLED")
            for node in veto_ifs
        ]
        self.assertNotIn(True, gated, "meta-test setup error: snippet looks gated")
        # ... which is exactly why the real gate above would fail on it.


if __name__ == "__main__":
    unittest.main(verbosity=2)
