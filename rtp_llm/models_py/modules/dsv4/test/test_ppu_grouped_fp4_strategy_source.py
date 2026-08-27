"""Source-only contract tests for the TP4/EP1 PPU grouped-MXFP4 strategy."""

import ast
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
from typing import Sequence, Tuple
import unittest
from unittest.mock import patch


_SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "moe"
    / "strategies"
    / "ppu_grouped_fp4.py"
)
_REGISTRY_PATH = _SOURCE_PATH.parent / "__init__.py"


def _load_helpers():
    tree = ast.parse(_SOURCE_PATH.read_text())
    wanted = {
        "_supports_topology",
        "_runtime_eligible",
        "_derive_inter_local_and_tp",
        "_select_capacity",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {
        "math": math,
        "Sequence": Sequence,
        "Tuple": Tuple,
        "MoeCfg": object,
        "torch": SimpleNamespace(),
        "_GROUPED_M_ALIGNMENT": 128,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(_SOURCE_PATH), "exec"), namespace)
    return tree, namespace


def _cfg(*, tp_size=4, ep_size=1, inter=2048, experts=256, dim=7168):
    return SimpleNamespace(
        tp_size=tp_size,
        ep_size=ep_size,
        dim=dim,
        moe_inter_dim=inter,
        n_local_experts=experts,
        n_routed_experts=experts,
    )


def _shapes(inter_local: int, experts: int = 256, dim: int = 7168):
    w1 = (experts, inter_local, dim // 2)
    w2 = (experts, dim, inter_local // 2)
    s1 = (experts, inter_local, dim // 32)
    s2 = (experts, dim, inter_local // 32)
    return w1, w2, w1, s1, s2, s1


class PpuGroupedFP4SourceContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = _SOURCE_PATH.read_text()
        cls.tree, cls.helpers = _load_helpers()

    def test_strategy_identity_and_exact_topology(self):
        supports = self.helpers["_supports_topology"]
        self.assertTrue(supports(_cfg(tp_size=4, ep_size=1)))
        self.assertFalse(supports(_cfg(tp_size=1, ep_size=1)))
        self.assertFalse(supports(_cfg(tp_size=4, ep_size=8)))
        self.assertFalse(supports(_cfg(tp_size=8, ep_size=1)))

        strategy = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "PpuGroupedFP4Strategy"
        )
        assignments = {
            target.id: ast.literal_eval(stmt.value)
            for stmt in strategy.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name) and isinstance(stmt.value, ast.Constant)
        }
        self.assertEqual(assignments["name"], "ppu_grouped_fp4")
        can_handle = next(
            node
            for node in strategy.body
            if isinstance(node, ast.FunctionDef) and node.name == "can_handle"
        )
        self.assertIn("_supports_topology(cfg)", ast.unparse(can_handle))
        self.assertIn("_runtime_eligible()", ast.unparse(can_handle))

    def test_registered_ahead_of_generic_and_loop_fallbacks(self):
        registry = _REGISTRY_PATH.read_text()
        ppu = registry.index("from .ppu_grouped_fp4 import")
        generic = registry.index("from .grouped_fp4 import")
        deepep = registry.index("from .deepep import")
        local_loop = registry.index("from .local_loop import")
        self.assertLess(ppu, generic)
        self.assertLess(ppu, deepep)
        self.assertLess(ppu, local_loop)

    def test_runtime_eligibility_rejects_generic_gpu_and_missing_symbol(self):
        runtime_eligible = self.helpers["_runtime_eligible"]

        class FakeCuda:
            def __init__(self, *, available, name):
                self.available = available
                self.name = name

            def is_available(self):
                return self.available

            def current_device(self):
                return 0

            def get_device_name(self, device):
                self.test_case.assertEqual(device, 0)
                return self.name

        def set_cuda(*, available, name):
            cuda = FakeCuda(available=available, name=name)
            cuda.test_case = self
            self.helpers["torch"] = SimpleNamespace(cuda=cuda)

        deep_gemm = ModuleType("deep_gemm")
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked = lambda *args: None

        set_cuda(available=False, name="ZW-M890P")
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.assertFalse(runtime_eligible())

        set_cuda(available=True, name="NVIDIA H100")
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.assertFalse(runtime_eligible())

        set_cuda(available=True, name="ZW-M890P")
        with patch.dict(sys.modules, {"deep_gemm": deep_gemm}):
            self.assertTrue(runtime_eligible())

        missing_symbol = ModuleType("deep_gemm")
        with patch.dict(sys.modules, {"deep_gemm": missing_symbol}):
            self.assertFalse(runtime_eligible())

        with patch.dict(sys.modules, {"deep_gemm": None}):
            self.assertFalse(runtime_eligible())

    def test_packed_geometry_detects_pure_tp_preshard(self):
        derive = self.helpers["_derive_inter_local_and_tp"]
        inter_local, routed_tp = derive(_cfg(), *_shapes(512))
        self.assertEqual((inter_local, routed_tp), (512, 4))

        inter_local, routed_tp = derive(_cfg(), *_shapes(2048))
        self.assertEqual((inter_local, routed_tp), (2048, 1))

        with self.assertRaises(ValueError):
            derive(_cfg(), *_shapes(1024))
        bad = list(_shapes(512))
        bad[1] = (256, 7168, 255)
        with self.assertRaises(ValueError):
            derive(_cfg(), *bad)
        with self.assertRaises(RuntimeError):
            derive(_cfg(tp_size=1), *_shapes(2048))

    def test_hidden_dim_requires_ep_scatter_scale_block_alignment(self):
        derive = self.helpers["_derive_inter_local_and_tp"]

        with self.assertRaisesRegex(ValueError, "dim aligned to 128"):
            derive(_cfg(dim=64, inter=256), *_shapes(64, dim=64))

        inter_local, routed_tp = derive(_cfg(dim=7168), *_shapes(512, dim=7168))
        self.assertEqual((inter_local, routed_tp), (512, 4))

    def test_capacity_is_lossless_aligned_and_graph_fails_closed(self):
        select = self.helpers["_select_capacity"]
        self.assertEqual(
            select(128, [1, 137, 4], 256, fixed_shape=False),
            137,
        )
        selected = select(128, [129, 4, 2], 3, fixed_shape=False)
        self.assertEqual(selected, 256)
        self.assertEqual((3 * selected) % 128, 0)

        with self.assertRaises(RuntimeError):
            select(128, (), 256, fixed_shape=True, fixed_required=129)
        with self.assertRaises(RuntimeError):
            select(128, (), 256, fixed_shape=True)
        with self.assertRaises(ValueError):
            select(129, (), 3, fixed_shape=True, fixed_required=128)

    def test_no_count_clamp_or_cuda_grouped_fallback(self):
        calls = [node for node in ast.walk(self.tree) if isinstance(node, ast.Call)]
        called_attributes = {
            node.func.attr for node in calls if isinstance(node.func, ast.Attribute)
        }
        referenced_names = {
            node.id for node in ast.walk(self.tree) if isinstance(node, ast.Name)
        }
        self.assertNotIn("clamp", called_attributes)
        self.assertNotIn("safe_counts", self.source)
        self.assertNotIn("GroupedFP4Strategy", referenced_names)
        self.assertNotIn("LocalLoopStrategy", self.source)
        self.assertNotIn("from .grouped_fp4", self.source)

    def test_quant_device_and_symbol_contracts_fail_closed(self):
        required_fragments = (
            "packed int8/uint8 MXFP4 weights",
            "float8_e8m0fnu checkpoint scales",
            "ZW-M890P",
            "m_grouped_gemm_fp4_fp4_bf16_nt_masked",
            "expert_counts.to(torch.int32).contiguous()",
            "self.routed_tp_size = routed_tp_size",
        )
        for fragment in required_fragments:
            self.assertIn(fragment, self.source)
        self.assertNotIn("except Exception", self.source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
