"""Test the reuse adapter in isolation, without loading the serving framework."""

import ast
from pathlib import Path
import unittest

import torch

SOURCE = Path(__file__).resolve().parents[1] / "moe/strategies/deepep.py"


def load_adapter():
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_full_slot_mxfp4_views")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[fn.name]


class FullSlotViewsTest(unittest.TestCase):
    def setUp(self):
        self.adapt = load_adapter()
        self.x = torch.empty((2, 256, 128), dtype=torch.uint8)
        self.s = torch.empty((2, 4, 256), dtype=torch.uint16).permute(0, 2, 1)

    def test_exact_objects_and_strides_preserved(self):
        x, s = self.adapt(self.x, self.s)
        self.assertIs(x, self.x)
        self.assertIs(s, self.s)
        self.assertEqual(s.stride(), (1024, 1, 256))

    def test_offset_views_preserve_storage_offset(self):
        x = torch.empty((3, 256, 128), dtype=torch.uint8)[1:]
        s = torch.empty((3, 4, 256), dtype=torch.uint16)[1:].permute(0, 2, 1)
        result = self.adapt(x, s)
        self.assertIs(result[0], x)
        self.assertIs(result[1], s)

    def test_wrong_rank_rejected(self):
        with self.assertRaises(ValueError):
            self.adapt(self.x[0], self.s)

    def test_wrong_scale_width_rejected(self):
        with self.assertRaises(ValueError):
            self.adapt(self.x, self.s[:, :, :2])

    def test_row_major_scales_rejected(self):
        with self.assertRaises(ValueError):
            self.adapt(self.x, self.s.contiguous())

    def test_noncontiguous_payload_rejected(self):
        with self.assertRaises(ValueError):
            self.adapt(self.x[:, ::2], self.s[:, ::2])

    def test_wrong_dtype_rejected(self):
        for x, s in ((self.x.to(torch.int8), self.s),
                     (self.x, self.s.to(torch.int32))):
            with self.assertRaises(ValueError):
                self.adapt(x, s)

    def test_mismatched_device_rejected(self):
        with self.assertRaises(ValueError):
            self.adapt(self.x, self.s.to("meta"))

    def test_full_slot_branch_precedes_n3_and_keeps_compact_branch(self):
        tree = ast.parse(SOURCE.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_compute_ppu_grouped_fp4_packed")
        branch = next(n for n in method.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) ==
                      "packed_dispatch and _LL_NO_COMPACT and (not _LL_COPY_INPUT_REFERENCE)")
        self.assertEqual(len(branch.body), 1)
        self.assertEqual(branch.body[0].value.func.id, "_full_slot_mxfp4_views")
        compact = branch.orelse[0]
        self.assertEqual(ast.unparse(compact.test), "packed_dispatch")
        self.assertIn("strided_prefix_to_compact", ast.unparse(compact))
        self.assertIn(".contiguous()", ast.unparse(compact))

    def test_copy_reference_config_is_default_off_and_requires_full_slot_n3(self):
        tree = ast.parse(SOURCE.read_text())
        assign = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and
                              t.id == "_LL_COPY_INPUT_REFERENCE" for t in n.targets))
        guard = tree.body[tree.body.index(assign) + 1]
        self.assertEqual(assign.value.args[0].value,
                         "DSV4_MOE_LL_COPY_INPUT_REFERENCE")
        self.assertIs(assign.value.args[1].value, False)
        for enabled, full, n3 in ((False, False, False), (False, True, False),
                                  (True, True, True), (True, False, True),
                                  (True, True, False), (True, False, False)):
            ns = dict(get_switch=lambda *unused: enabled, parse_bool=None,
                      _LL_NO_COMPACT=full, _dispatch_copy_2d_enabled=lambda: n3)
            code = compile(ast.Module(body=[assign, guard], type_ignores=[]),
                           str(SOURCE), "exec")
            if enabled and not (full and n3):
                with self.assertRaisesRegex(ValueError, "requires"):
                    exec(code, ns)
            else:
                exec(code, ns)
                self.assertEqual(ns["_LL_COPY_INPUT_REFERENCE"], enabled)

    def test_copy_reference_does_not_change_capacity_or_combine(self):
        tree = ast.parse(SOURCE.read_text())
        forward = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                       and n.name == "_forward_ppu_grouped_fp4_low_latency")
        self.assertNotIn("_LL_COPY_INPUT_REFERENCE", ast.unparse(forward))
        self.assertIn("if _LL_NO_COMPACT:", ast.unparse(forward))

    def test_full_slot_capacity_not_128_and_monitor_retained(self):
        tree = ast.parse(SOURCE.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == "_compute_ppu_grouped_fp4_packed")
        select = next(n for n in method.body if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "_LL_NO_COMPACT")
        self.assertEqual(ast.unparse(select.body[0]), "compute_capacity = ll_capacity")
        self.assertIn("self._ovf.record(expert_num_tokens, compute_capacity, local_tokens)",
                      ast.unparse(method))


if __name__ == "__main__":
    unittest.main()
