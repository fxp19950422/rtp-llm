"""CPU event-oracle tests for V4 routed scaling, TP reduce, and shared add."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

os.environ["MOEDBG"] = "0"

_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS, "..", "..", "..", "..", ".."))


def _stub_package(name: str, path: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules.setdefault(name, module)


_stub_package("rtp_llm", os.path.join(_REPO, "rtp_llm"))
_stub_package("rtp_llm.models_py", os.path.join(_REPO, "rtp_llm", "models_py"))
_stub_package(
    "rtp_llm.models_py.modules",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4.moe",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4", "moe"),
)
_stub_package(
    "rtp_llm.models_py.modules.dsv4.moe.strategies",
    os.path.join(_REPO, "rtp_llm", "models_py", "modules", "dsv4", "moe", "strategies"),
)

# Orchestration tests replace the shared implementation with a minimal import
# surface, then spy on the combine boundary explicitly.
_shared_stub = types.ModuleType(
    "rtp_llm.models_py.modules.dsv4.moe.shared_expert"
)


class _UnusedSharedExpert(nn.Module):
    pass


def _stub_combine(routed, shared, dtype, out=None):
    result = (routed.float() + shared.float()).to(dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


def _unused_shared_executor(**_kwargs):
    raise AssertionError("constructor path is not used by this contract test")


_shared_stub.W13SharedExpert = _UnusedSharedExpert
_shared_stub.combine_routed_and_shared = _stub_combine
_shared_stub.get_shared_expert_executor = _unused_shared_executor
sys.modules.setdefault(_shared_stub.__name__, _shared_stub)

from rtp_llm.models_py.modules.dsv4.moe import moe_layer
from rtp_llm.models_py.modules.dsv4.moe.moe_layer import MoE
from rtp_llm.models_py.modules.dsv4.moe.strategies.local_loop import (
    LocalLoopStrategy,
)


class _LoggingScale:
    def __init__(self, events: list[str], value: float) -> None:
        self.events = events
        self.value = value

    def __float__(self) -> float:
        self.events.append("route_scale")
        return self.value


class _PostW2Gate(nn.Module):
    def __init__(self, events: list[str], route_scale: float) -> None:
        super().__init__()
        self.events = events
        self.route_scale = _LoggingScale(events, route_scale)

    def forward(self, x, input_ids, *, include_route_scale=True):
        self.events.append("unscaled_weights")
        if include_route_scale:
            raise AssertionError("post-W2 contract must request unscaled weights")
        return (
            torch.ones((x.size(0), 1), dtype=torch.float32, device=x.device),
            torch.zeros((x.size(0), 1), dtype=torch.long, device=x.device),
        )


class _LegacyScaledGate(nn.Module):
    """No new keyword: proves legacy/default strategy API compatibility."""

    def __init__(self, events: list[str], route_scale: float) -> None:
        super().__init__()
        self.events = events
        self.route_scale = route_scale

    def forward(self, x, input_ids):
        self.events.append("legacy_scaled_weights")
        return (
            torch.full(
                (x.size(0), 1),
                self.route_scale,
                dtype=torch.float32,
                device=x.device,
            ),
            torch.zeros((x.size(0), 1), dtype=torch.long, device=x.device),
        )


class _PostW2Strategy(nn.Module):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def forward(self, x, weights, indices):
        del indices
        self.events.append("post_w2_weight")
        if not torch.equal(weights, torch.ones_like(weights)):
            raise AssertionError("strategy must receive normalized unscaled weights")
        return torch.full_like(x, 2.0, dtype=torch.float32)


class _SharedExecutor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.output: torch.Tensor | None = None

    def start(self, shared_experts, x):
        del shared_experts
        self.events.append("shared_start")
        self.output = torch.full_like(x, 5.0, dtype=torch.float32)

    def finish(self):
        self.events.append("shared_finish")
        assert self.output is not None
        output = self.output
        self.output = None
        return output


def _post_w2_moe(
    events: list[str], *, tp_size: int, max_tokens: int = 2
) -> MoE:
    moe = MoE.__new__(MoE)
    nn.Module.__init__(moe)
    moe.layer_id = 0
    moe.dim = 2
    moe.max_tokens_per_rank = max_tokens
    moe._is_decode_role = False
    moe._routed_includes_shared = False
    moe._post_w2_route_weight_contract = True
    moe._routed_tp_size = tp_size
    moe._gate_pack_static = False
    moe.gate = _PostW2Gate(events, route_scale=3.0)
    moe._strategy = _PostW2Strategy(events)
    moe.shared_experts = nn.Identity()
    moe._shared_executor = _SharedExecutor(events)
    moe._dbg_positions = None
    return moe


def _event_combine(events: list[str]):
    def combine(routed, shared, dtype, out=None):
        events.append("shared_add")
        result = (routed.float() + shared.float()).to(dtype)
        if out is not None:
            out.copy_(result)
            return out
        return result

    return combine


def _event_all_reduce(events: list[str]):
    def all_reduce(routed):
        events.append("all_reduce")
        # Strategy emits 2, then route_scale=3 must run before the collective.
        if not torch.equal(routed, torch.full_like(routed, 6.0)):
            raise AssertionError("TP all-reduce observed an unscaled routed output")
        return routed * 2.0

    return all_reduce


class MoEFinishOrderingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        moe_layer._FINAL_OUT_CACHE.clear()
        moe_layer._CHUNKED_MOE_LOGGED = False

    def _patch_finish_boundaries(self, events: list[str]):
        return mock.patch.multiple(
            moe_layer,
            _all_reduce_routed_tp=_event_all_reduce(events),
            combine_routed_and_shared=_event_combine(events),
        )

    def test_nonchunk_sequence_is_scale_reduce_then_single_shared_add(self):
        events: list[str] = []
        moe = _post_w2_moe(events, tp_size=2)
        x = torch.zeros((1, 2), dtype=torch.float32)
        with self._patch_finish_boundaries(events):
            actual = moe(x, torch.tensor([7], dtype=torch.long))

        self.assertEqual(
            events,
            [
                "unscaled_weights",
                "shared_start",
                "post_w2_weight",
                "route_scale",
                "all_reduce",
                "shared_finish",
                "shared_add",
            ],
        )
        self.assertEqual(events.count("shared_add"), 1)
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 17.0)))

    def test_chunk_sequence_repeats_contract_for_each_chunk(self):
        events: list[str] = []
        moe = _post_w2_moe(events, tp_size=2, max_tokens=2)
        x = torch.zeros((3, 2), dtype=torch.float32)
        per_chunk = [
            "unscaled_weights",
            "shared_start",
            "post_w2_weight",
            "route_scale",
            "all_reduce",
            "shared_finish",
            "shared_add",
        ]
        with self._patch_finish_boundaries(events):
            actual = moe._forward_chunked(
                x, torch.tensor([1, 2, 3], dtype=torch.long), x.size()
            )

        self.assertEqual(events, per_chunk + per_chunk)
        self.assertEqual(events.count("shared_add"), 2)
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 17.0)))

    def test_tp1_applies_scale_without_collective(self):
        events: list[str] = []
        moe = _post_w2_moe(events, tp_size=1)
        with mock.patch.object(
            moe_layer,
            "_all_reduce_routed_tp",
            side_effect=AssertionError("TP=1 must not all-reduce"),
        ):
            actual = moe._finish_routed(torch.full((1, 2), 2.0))

        self.assertEqual(events, ["route_scale"])
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 6.0)))

    def test_init_recognizes_local_loop_explicit_contract(self):
        class _InitGate(nn.Module):
            def __init__(
                self,
                _layer_id,
                _dim,
                _n_routed,
                _topk,
                _score_func,
                route_scale,
                *_args,
                **_kwargs,
            ) -> None:
                super().__init__()
                self.route_scale = route_scale

        class _InitSharedExpert(nn.Module):
            def __init__(self, *_args, **_kwargs) -> None:
                super().__init__()

        class _InitSharedExecutor:
            def __init__(self) -> None:
                self.prepared = None

            def prepare(self, shared_experts) -> None:
                self.prepared = shared_experts

        class _FakeW:
            v4_shared_w13_w = "shared.w13.w"
            v4_shared_w13_s = "shared.w13.s"
            v4_shared_w2_w = "shared.w2.w"
            v4_shared_w2_s = "shared.w2.s"

        fake_weight_module = types.ModuleType("rtp_llm.utils.model_weight")
        fake_weight_module.W = _FakeW
        layer_weights = {
            key: torch.empty(0)
            for key in (
                _FakeW.v4_shared_w13_w,
                _FakeW.v4_shared_w13_s,
                _FakeW.v4_shared_w2_w,
                _FakeW.v4_shared_w2_s,
            )
        }
        executor = _InitSharedExecutor()
        select_spy = mock.Mock(return_value=LocalLoopStrategy)
        with mock.patch.dict(
            sys.modules, {"rtp_llm.utils.model_weight": fake_weight_module}
        ), mock.patch.multiple(
            moe_layer,
            Gate=_InitGate,
            W13SharedExpert=_InitSharedExpert,
            get_shared_expert_executor=mock.Mock(return_value=executor),
            _resolve_forced=mock.Mock(return_value=(None, False)),
            select_strategy=select_spy,
        ), mock.patch.object(
            LocalLoopStrategy,
            "setup_weights",
            autospec=True,
            side_effect=lambda strategy, _weights: setattr(
                strategy, "routed_tp_size", strategy.cfg.tp_size
            ),
        ) as setup_spy:
            moe = MoE(
                layer_id=3,
                dim=4,
                moe_inter_dim=8,
                n_routed_experts=8,
                n_activated_experts=2,
                n_shared_experts=1,
                score_func="sqrtsoftplus",
                route_scale=1.5,
                swiglu_limit=10.0,
                n_hash_layers=0,
                vocab_size=0,
                layer_weights=layer_weights,
                tp_size=2,
                max_tokens_per_rank=2,
                strategy="local_loop",
            )

        select_spy.assert_called_once()
        setup_spy.assert_called_once_with(moe._strategy, layer_weights)
        self.assertIsInstance(moe._strategy, LocalLoopStrategy)
        self.assertIs(executor.prepared, moe.shared_experts)
        self.assertTrue(moe._post_w2_route_weight_contract)
        self.assertEqual(moe._strategy.cfg.tp_size, 2)
        self.assertEqual(moe._routed_tp_size, 2)
        self.assertEqual(
            vars(LocalLoopStrategy)["route_weight_contract"],
            "post_w2_normalized_then_scale_v1",
        )

    def test_local_loop_tp_gt1_without_dist_fails_closed(self):
        events: list[str] = []
        moe = _post_w2_moe(events, tp_size=2)
        with mock.patch.object(
            torch.distributed, "is_available", return_value=True
        ), mock.patch.object(
            torch.distributed, "is_initialized", return_value=False
        ), self.assertRaisesRegex(RuntimeError, "require initialized"):
            moe._finish_routed(torch.ones((1, 2), dtype=torch.float32))

        self.assertEqual(events, ["route_scale"])

    def test_other_strategy_keeps_scaled_gate_and_gets_no_second_scale(self):
        events: list[str] = []
        moe = MoE.__new__(MoE)
        nn.Module.__init__(moe)
        moe._post_w2_route_weight_contract = False
        moe._routed_tp_size = 99
        moe.gate = _LegacyScaledGate(events, route_scale=3.0)
        x = torch.zeros((2, 2), dtype=torch.float32)

        weights, _ = moe._route(x, torch.tensor([1, 2], dtype=torch.long))
        routed = torch.full((2, 2), 4.0, dtype=torch.float32)
        with mock.patch.object(
            moe_layer,
            "_all_reduce_routed_tp",
            side_effect=AssertionError("legacy strategy must not be reduced here"),
        ):
            finished = moe._finish_routed(routed)

        self.assertEqual(events, ["legacy_scaled_weights"])
        self.assertTrue(torch.equal(weights, torch.full_like(weights, 3.0)))
        self.assertIs(finished, routed)

    def test_input_id_count_mismatch_is_an_error(self):
        events: list[str] = []
        moe = _post_w2_moe(events, tp_size=1)
        with self.assertRaisesRegex(RuntimeError, "input_ids/token mismatch"):
            moe(torch.zeros((2, 2)), torch.tensor([1], dtype=torch.long))
        self.assertEqual(events, [])


class SharedTpCombinedReductionTest(unittest.TestCase):
    def setUp(self):
        moe_layer._FINAL_OUT_CACHE.clear()

    def _case(self, tokens, cap, debug=False):
        events = []
        moe = _post_w2_moe(events, tp_size=4, max_tokens=cap)
        moe._shared_tp_size = 4
        seen = []

        def reduce(local):
            events.append("combined_reduce")
            self.assertEqual(local.dtype, torch.float32)
            # routed=2, scale=3, shared partial=5: scale must not touch shared.
            self.assertTrue(torch.equal(local, torch.full_like(local, 11)))
            seen.append(local.clone())
            # Return different storage to cover collective fast paths.
            return local * 4

        with mock.patch.object(moe_layer, "_all_reduce_routed_tp", reduce), mock.patch.dict(
            os.environ, {"MOEDBG": "1" if debug else "0"}
        ):
            actual = moe(
                torch.zeros(tokens, 2, dtype=torch.bfloat16),
                torch.arange(tokens, dtype=torch.long),
            ).clone()
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, torch.full_like(actual, 44)))
        self.assertEqual(len(seen), (tokens + cap - 1) // cap)
        self.assertEqual(events.count("route_scale"), len(seen))
        for i, event in enumerate(events):
            if event == "combined_reduce":
                self.assertEqual(events[i - 1], "shared_finish")
        return moe

    def test_nonchunk_single_combined_fp32_reduce(self):
        self._case(2, 4)

    def test_chunk_and_tail_each_reduce_once(self):
        self._case(5, 2)

    def test_debug_path_keeps_combined_reduction(self):
        self._case(2, 4, debug=True)

    def test_failed_routed_path_still_finishes_shared(self):
        events = []
        moe = _post_w2_moe(events, tp_size=4)
        moe._shared_tp_size = 4
        with mock.patch.object(
            moe._strategy, "forward", side_effect=RuntimeError("routed")
        ), self.assertRaisesRegex(RuntimeError, "routed"):
            moe(torch.zeros(1, 2), torch.tensor([0]))
        self.assertEqual(events[-1], "shared_finish")

    def test_bf16_shared_partial_is_added_before_final_rounding(self):
        moe = _post_w2_moe([], tp_size=4)
        moe._shared_tp_size = 4
        # Cancellation distinguishes FP32 add from an accidental BF16 local
        # sum. This models the production BF16 shared-expert output path.
        local_routed = torch.tensor([[2.0 ** -10]], dtype=torch.float32)
        shared_bf16 = torch.ones(1, 1, dtype=torch.bfloat16)

        def reduce(local):
            self.assertEqual(local.dtype, torch.float32)
            self.assertEqual(local.item(), 1.0 + 2.0 ** -10)
            return local - 1.0

        with mock.patch.object(moe_layer, "_all_reduce_routed_tp", reduce):
            result = moe._combine_shared(local_routed, shared_bf16, torch.bfloat16)
        self.assertEqual(result.dtype, torch.bfloat16)
        self.assertEqual(result.item(), 2.0 ** -10)

    def test_unsupported_shared_topologies_fail_before_weight_use(self):
        kwargs = dict(
            layer_id=0, dim=4096, moe_inter_dim=2048, n_routed_experts=256,
            n_activated_experts=6, n_shared_experts=1, score_func="sqrtsoftplus",
            route_scale=1.5, swiglu_limit=10.0, n_hash_layers=3, vocab_size=129280,
            layer_weights={}, shared_tp_size=4, tp_size=4, ep_size=1,
        )
        for changes in (
            {"is_decode_role": True}, {"tp_size": 1}, {"ep_size": 8},
            {"moe_inter_dim": 4096}, {"dim": 8192}, {"shared_tp_size": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                MoE(**dict(kwargs, **changes))

    def test_partial_dtype_mismatch_is_rejected(self):
        moe = _post_w2_moe([], tp_size=4)
        moe._shared_tp_size = 4
        with self.assertRaisesRegex(RuntimeError, "FP32"):
            moe._combine_shared(torch.ones(1, 2, dtype=torch.bfloat16),
                                torch.ones(1, 2), torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
