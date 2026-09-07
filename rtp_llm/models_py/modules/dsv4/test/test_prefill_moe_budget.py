"""CPU-only P2 topology, runtime wiring and real chunk-loop structure checks."""

from __future__ import annotations

import ast
import logging
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parents[4]
for name, rel in (
    ("rtp_llm", ""),
    ("rtp_llm.models_py", "models_py"),
    ("rtp_llm.models_py.modules", "models_py/modules"),
    ("rtp_llm.models_py.modules.dsv4", "models_py/modules/dsv4"),
):
    module = types.ModuleType(name)
    module.__path__ = [str(_ROOT / rel)]
    sys.modules.setdefault(name, module)

from rtp_llm.models_py.modules.dsv4 import runtime_config
from rtp_llm.models_py.modules.dsv4.prefill_moe_budget import (
    PREFILL_MOE_MAX_TOKENS_SWITCH,
    resolve_prefill_moe_max_tokens,
)

_DSV4 = _ROOT / "models_py/modules/dsv4"


def _parallel(role="PREFILL", **overrides):
    values = dict(
        role_type=role, tp_size=4, ep_size=1, dp_size=1, world_size=4,
        get_attn_tp_size=lambda: 4, get_attn_tp_rank=lambda: 2,
        prefill_cp_config=SimpleNamespace(prefill_cp_size=1, is_enabled=lambda: False),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _resolve(budget=10240, parallel=None, **kwargs):
    return resolve_prefill_moe_max_tokens(
        budget, parallel if parallel is not None else _parallel(),
        is_decode_role=kwargs.pop("is_decode_role", False),
        chunked_moe=kwargs.pop("chunked_moe", True), **kwargs,
    )


class PrefillMoeBudgetTest(unittest.TestCase):
    def test_default_preserves_legacy_without_topology(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.dict(
            runtime_config._VALUES, {}, clear=True
        ):
            self.assertEqual(resolve_prefill_moe_max_tokens(
                10240, None, is_decode_role=False, chunked_moe=True), 10240)

    def test_zero_preserves_arbitrary_legacy_geometry(self):
        for budget in (320, 1280, 4096, 10240, 32768):
            self.assertEqual(_resolve(budget, configured=0), budget)

    def test_prefill_override_is_independent_of_context_and_legacy_cap(self):
        for budget in (4096, 8192, 10240, 32768):
            self.assertEqual(_resolve(budget, configured=16384), 16384)

    def test_actual_decode_retains_batch_verify_budget(self):
        for budget in (80, 320, 1280):
            self.assertEqual(_resolve(
                budget, _parallel(role="DECODE", ep_size=8),
                configured=16384, is_decode_role=True, chunked_moe=False), budget)

    def test_runtime_decode_wins_over_stale_prefill_config(self):
        self.assertEqual(_resolve(320, configured=16384, is_decode_role=True), 320)

    def test_env_resolves_once_and_logs(self):
        with mock.patch.dict("os.environ", {PREFILL_MOE_MAX_TOKENS_SWITCH: "16384"}), \
             mock.patch.dict(runtime_config._VALUES, {}, clear=True):
            with self.assertLogs(level="INFO") as log:
                self.assertEqual(_resolve(), 16384)
            self.assertTrue(any(PREFILL_MOE_MAX_TOKENS_SWITCH in s for s in log.output))
            with mock.patch.dict("os.environ", {PREFILL_MOE_MAX_TOKENS_SWITCH: "0"}):
                self.assertEqual(_resolve(), 16384)

    def test_bad_values_fail(self):
        for value in ("", "yes", "-1", "8192", "10240", "32768", "16384.0"):
            with self.subTest(value=value), \
                 mock.patch.dict("os.environ", {PREFILL_MOE_MAX_TOKENS_SWITCH: value}), \
                 mock.patch.dict(runtime_config._VALUES, {}, clear=True), \
                 self.assertRaises(ValueError):
                _resolve()

    def test_requires_actual_framework_prefill(self):
        for role in ("DECODE", "PDFUSION", "None"):
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "explicit PREFILL"):
                _resolve(parallel=_parallel(role), configured=16384)
        with mock.patch.dict("os.environ", {"ROLE_TYPE": "DECODE"}):
            self.assertEqual(_resolve(configured=16384), 16384)

    def test_topology_restrictions(self):
        for changes in (
            {"tp_size": 8}, {"ep_size": 8}, {"dp_size": 2}, {"world_size": 8},
            {"get_attn_tp_size": lambda: 1}, {"get_attn_tp_rank": lambda: 4},
            {"get_attn_tp_rank": lambda: -1},
            {"prefill_cp_config": SimpleNamespace(prefill_cp_size=4, is_enabled=lambda: True)},
            {"prefill_cp_config": SimpleNamespace(prefill_cp_size=1, is_enabled=lambda: True)},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "TP4/CP1"):
                _resolve(parallel=_parallel(**changes), configured=16384)

    def test_disabled_chunking_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "enabled MoE chunking"):
            _resolve(configured=16384, chunked_moe=False)


class ProductionWiringTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_tree = ast.parse(
            (_ROOT / "models_py/model_desc/deepseek_v4_model.py").read_text()
        )
        cls.model = next(n for n in cls.model_tree.body
                         if isinstance(n, ast.ClassDef) and n.name == "DeepSeekV4Model")

    def test_actual_initialization_call_preserves_context_and_decode(self):
        init = next(n for n in self.model.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_initialize_impl")
        assignments = [n for n in init.body if isinstance(n, ast.Assign)
                       and isinstance(n.value, ast.Call)
                       and isinstance(n.value.func, ast.Name)
                       and n.value.func.id == "resolve_prefill_moe_max_tokens"]
        self.assertEqual(len(assignments), 1)
        code = compile(ast.Module(body=assignments, type_ignores=[]), "<runtime budget>", "exec")
        for is_decode, legacy, expected in ((False, 10240, 16384), (True, 320, 320)):
            model = SimpleNamespace(
                parallelism_config=_parallel("DECODE" if is_decode else "PREFILL"),
                _is_decode_role=is_decode, _v4_args=SimpleNamespace(max_seq_len=10240),
                _max_context_batch_size=32,
            )
            env = {
                "self": model, "runtime_resolved_max_tokens_per_rank": legacy,
                "chunked_moe_enabled": lambda: True,
                "resolve_prefill_moe_max_tokens": lambda *a, **kw:
                    resolve_prefill_moe_max_tokens(*a, configured=16384, **kw),
            }
            exec(code, env)
            self.assertEqual(env["runtime_resolved_max_tokens_per_rank"], expected)
            self.assertEqual(model._v4_args.max_seq_len, 10240)
            self.assertEqual(model._max_context_batch_size, 32)

    def test_dense_warmup_covers_independent_moe_cap(self):
        assignments = [
            n for n in ast.walk(self.model)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_dense_gemm_prefill_chunk_size"
                    for t in n.targets)
            and isinstance(n.value, ast.Call)
        ]
        self.assertEqual(len(assignments), 1)
        code = compile(ast.Module(body=assignments, type_ignores=[]), "<warmup budget>", "exec")
        for moe_cap, legacy_chunk, expected in (
            (10240, 16384, 16384), (8192, 8192, 8192), (16384, 8192, 16384),
        ):
            env = {
                "self": SimpleNamespace(_v4_args=SimpleNamespace(max_tokens_per_rank=moe_cap)),
                "moe_chunk_tokens_from_env": lambda: legacy_chunk,
            }
            exec(code, env)
            self.assertEqual(env["_dense_gemm_prefill_chunk_size"], expected)

    def test_prefill_shared_q_and_mtp_buffers_still_use_context(self):
        method = next(n for n in self.model.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_resolve_shared_token_capacity")
        env = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<shared capacity>", "exec"), env)
        model = SimpleNamespace(
            _is_decode_role=False, _prefill_cp_size=1, _max_context_batch_size=32,
            _v4_args=SimpleNamespace(max_seq_len=10240, max_tokens_per_rank=16384),
        )
        self.assertEqual(env["_resolve_shared_token_capacity"](model), 10240 * 32)

    def test_existing_moe_loop_covers_chunk_boundaries_without_losing_rows(self):
        tree = ast.parse((_DSV4 / "moe/moe_layer.py").read_text())
        moe = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoE")
        method = next(n for n in moe.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_forward_chunked")
        # Execute the real loop with CPU row IDs, without importing torch/native.
        class Rows:
            def __init__(self, values):
                self.values = list(values)
                self.dtype = "bf16"
                self.device = "cpu"
            def size(self, dim):
                return len(self.values)
            def __getitem__(self, item):
                return Rows(self.values[item])
            def view(self, shape):
                return self
        env = {
            "torch": SimpleNamespace(Tensor=Rows, Size=tuple), "logging": logging,
            "_CHUNKED_MOE_LOGGED": True,
            "_get_or_create_final_out": lambda n, *args: Rows(range(n)),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<real chunk loop>", "exec"), env)
        for n, expected in ((10241, [10241]), (16384, [16384]),
                            (16385, [16384, 1]), (32768, [16384, 16384])):
            seen = []
            model = SimpleNamespace(max_tokens_per_rank=16384, dim=4096)
            def run_chunk(x, ids, out):
                self.assertEqual(x.values, ids.values)
                self.assertEqual(x.values, out.values)
                seen.append(x.values)
            model._run_chunk = run_chunk
            result = env["_forward_chunked"](model, Rows(range(n)), Rows(range(n)), (n, 4096))
            self.assertEqual([len(part) for part in seen], expected)
            self.assertEqual([row for part in seen for row in part], list(range(n)))
            self.assertEqual(result.values, list(range(n)))


if __name__ == "__main__":
    unittest.main()
