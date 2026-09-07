"""CPU checks for shared TP layout, raw-byte ownership, and loader dispatch."""

from __future__ import annotations

import ast
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

_ROOT = Path(__file__).resolve().parents[4]
for name, rel in (
    ("rtp_llm", ""),
    ("rtp_llm.utils", "utils"),
    ("rtp_llm.models_py", "models_py"),
    ("rtp_llm.models_py.modules", "models_py/modules"),
    ("rtp_llm.models_py.modules.dsv4", "models_py/modules/dsv4"),
):
    module = types.ModuleType(name)
    module.__path__ = [str(_ROOT / rel)]
    sys.modules.setdefault(name, module)

from rtp_llm.models_py.modules.dsv4 import runtime_config
from rtp_llm.models_py.modules.dsv4.shared_tp import (
    SHARED_TP4_SWITCH,
    SharedTpLayout,
    resolve_prefill_shared_tp4,
    shared_fp8_tp_slice,
)
from rtp_llm.utils.model_weight import W, concat_0


def _parallel(role="PREFILL", **overrides):
    values = dict(
        role_type=role, tp_size=4, ep_size=1, dp_size=1, world_size=4,
        get_attn_tp_size=lambda: 4, get_attn_tp_rank=lambda: 2,
        prefill_cp_config=SimpleNamespace(prefill_cp_size=1, is_enabled=lambda: False),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _raw(shape, scale=False):
    # Byte pattern varies in both axes and between W13 halves. It need not
    # represent finite values: this test verifies ownership, not arithmetic.
    count = shape[0] * shape[1]
    i = torch.arange(count, dtype=torch.int64)
    data = ((i * 31 + i // shape[1] * 17 + i // 131) % 256).to(torch.uint8)
    return data.reshape(shape).view(
        torch.float8_e8m0fnu if scale else torch.float8_e4m3fn
    )


class SharedTpLayoutTest(unittest.TestCase):
    def test_default_is_off_without_topology(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch.dict(
            runtime_config._VALUES, {}, clear=True
        ):
            self.assertEqual(resolve_prefill_shared_tp4(None), SharedTpLayout())

    def test_prefill_rank_comes_from_framework(self):
        with mock.patch.dict("os.environ", {"ROLE_TYPE": "decode"}):
            self.assertEqual(resolve_prefill_shared_tp4(_parallel(), enabled=True), (4, 2))

    def test_decode_with_inherited_switch_stays_replicated(self):
        self.assertEqual(
            resolve_prefill_shared_tp4(_parallel(role="DECODE", ep_size=8), enabled=True),
            (1, 0),
        )

    def test_explicit_role_is_required_when_on(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            resolve_prefill_shared_tp4(_parallel(role="PDFUSION"), enabled=True)

    def test_unsupported_prefill_topologies_fail(self):
        for changes in (
            {"tp_size": 8}, {"ep_size": 4}, {"dp_size": 2}, {"world_size": 8},
            {"get_attn_tp_size": lambda: 1}, {"get_attn_tp_rank": lambda: 4},
            {"prefill_cp_config": SimpleNamespace(prefill_cp_size=4, is_enabled=lambda: True)},
        ):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "TP4/CP1"):
                resolve_prefill_shared_tp4(_parallel(**changes), enabled=True)

    def test_invalid_switch_is_rejected(self):
        with mock.patch.dict("os.environ", {SHARED_TP4_SWITCH: "maybe"}), mock.patch.dict(
            runtime_config._VALUES, {}, clear=True
        ), self.assertRaises(ValueError):
            resolve_prefill_shared_tp4(_parallel())


class SharedTpBytesTest(unittest.TestCase):
    def test_four_shards_rebuild_all_checkpoint_bytes(self):
        for projection, shape in (("w13", (4096, 4096)), ("w2", (4096, 2048))):
            for scale in (False, True):
                raw_shape = tuple(n // 128 for n in shape) if scale else shape
                original = _raw(raw_shape, scale)
                shards = [
                    shared_fp8_tp_slice(
                        original, projection=projection, tp_size=4, tp_rank=r,
                        is_scale=scale, ffn_tp_size=1, ffn_tp_rank=0,
                    )
                    for r in range(4)
                ]
                if projection == "w13":
                    halves = [t.view(torch.uint8).chunk(2, dim=0) for t in shards]
                    rebuilt = torch.cat([torch.cat([h[i] for h in halves]) for i in range(2)])
                else:
                    rebuilt = torch.cat([t.view(torch.uint8) for t in shards], dim=1)
                with self.subTest(projection=projection, scale=scale):
                    self.assertTrue(torch.equal(rebuilt, original.view(torch.uint8)))
                    self.assertTrue(all(t.is_contiguous() for t in shards))

    def test_rank_w13_contains_matching_gate_and_up_neurons(self):
        t = _raw((4096, 4096))
        got = shared_fp8_tp_slice(t, projection="w13", tp_size=4, tp_rank=2)
        expected = concat_0([t[1024:1536], t[3072:3584]])
        self.assertTrue(torch.equal(got.view(torch.uint8), expected.view(torch.uint8)))

    def test_tp1_preserves_object(self):
        for scale in (False, True):
            t = _raw((32, 32) if scale else (4096, 4096), scale)
            self.assertIs(shared_fp8_tp_slice(
                t, projection="w13", tp_size=1, tp_rank=0, is_scale=scale
            ), t)

    def test_wrong_format_and_unaligned_shards_fail(self):
        for tensor, projection, scale in (
            (torch.empty(32, 32), "w13", True),
            (_raw((32, 32), True).view(torch.int32), "w13", True),
            (_raw((768, 128)), "w13", False),
            (_raw((128, 640)), "w2", False),
        ):
            with self.subTest(projection=projection, shape=tensor.shape), self.assertRaises(ValueError):
                shared_fp8_tp_slice(
                    tensor, projection=projection, tp_size=4, tp_rank=0, is_scale=scale
                )


class SharedTpLoaderDispatchTest(unittest.TestCase):
    def test_quant_weight_dispatch_preserves_per_instance_scope(self):
        # Execute the real dispatch method without importing native libraries.
        # Native loader lifecycle/quant-wrapper construction is a later
        # integration check; this verifies its shared split selection itself.
        path = _ROOT / "model_loader/per_block_fp8_quant_weight.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == "W8A8Fp8PerBlockAtomicWeight")
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == "_get_split_func")
        namespace = {"W": W}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
        dispatch = namespace["_get_split_func"]
        identity = lambda t, **_: t
        table = {key: identity for key in (
            W.v4_shared_w13_w, W.v4_shared_w13_s, W.v4_shared_w2_w, W.v4_shared_w2_s,
            W.v4_attn_wq_a_w,
        )}
        for key, projection, scale, shape in (
            (W.v4_shared_w13_w, "w13", False, (1024, 128)),
            (W.v4_shared_w13_s, "w13", True, (8, 1)),
            (W.v4_shared_w2_w, "w2", False, (128, 512)),
            (W.v4_shared_w2_s, "w2", True, (1, 4)),
        ):
            weight = SimpleNamespace(
                name=key, gpt_style_tp_strategy=table,
                config=SimpleNamespace(shared_tp_size=4, shared_tp_rank=3),
            )
            t = _raw(shape, scale)
            actual = dispatch(weight)(t=t, tp=4, tp_rank=0, ffn_tp_size=1, ffn_tp_rank=0)
            expected = shared_fp8_tp_slice(
                t, projection=projection, tp_size=4, tp_rank=3, is_scale=scale
            )
            self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))
            weight.config.shared_tp_size = 1
            self.assertIs(dispatch(weight), identity)
        self.assertTrue(all(fn is identity for fn in table.values()))
        weight.name = W.v4_attn_wq_a_w
        weight.config.shared_tp_size = 4
        self.assertIs(dispatch(weight), identity)


if __name__ == "__main__":
    unittest.main()
