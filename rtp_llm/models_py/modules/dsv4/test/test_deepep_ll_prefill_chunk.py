"""Unit test for ``DeepEPLowLatencyStrategy.forward`` prefill chunking.

The LL dispatch buffer holds ``ll_num_max_token_per_rank`` (``cap``) tokens per
rank. Decode fits in one shot; prefill can exceed ``cap`` and previously raised
a hard ``RuntimeError`` (observed live: "received 292 tokens but LL buffer holds
256"). ``forward`` now streams ``N > cap`` prefill through the same working LL
path in ``<=cap`` chunks and stitches the result (MoE routing is per-token, so
this is exact). This test pins that behavior without needing DeepEP / EP / CUDA:
``_forward_chunk_ll`` and the ``deepep_wrapper`` module are stubbed.
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

import torch


def _install_wrapper_stub(cap: int, num_topk: int) -> None:
    mod = types.ModuleType("rtp_llm.models_py.distributed.deepep_wrapper")

    class DeepEPMode:
        LOW_LATENCY = "low_latency"
        NORMAL = "normal"

    _cap = cap
    _topk = num_topk

    class _Wrap:
        mode = DeepEPMode.LOW_LATENCY
        ll_num_max_token_per_rank = _cap
        num_topk = _topk

    class DeepEPWrapper:
        _instance = _Wrap()

    mod.DeepEPMode = DeepEPMode
    mod.DeepEPWrapper = DeepEPWrapper
    sys.modules["rtp_llm.models_py.distributed.deepep_wrapper"] = mod


class PrefillChunkTest(unittest.TestCase):
    def _make(self, cap: int, dim: int = 2048, num_topk: int = 6):
        _install_wrapper_stub(cap, num_topk)
        from rtp_llm.models_py.modules.dsv4.moe.strategies.deepep_low_latency import (
            DeepEPLowLatencyStrategy,
        )

        st = DeepEPLowLatencyStrategy.__new__(DeepEPLowLatencyStrategy)

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.dim = dim
        st.cfg = cfg
        return st

    def test_prefill_chunks_when_N_gt_cap(self):
        # Reproduces the live crash scenario (N=292 > cap=256): must chunk, not raise.
        st = self._make(cap=256)
        calls = []

        def fake_ll(wrapper, x, w, idx):
            calls.append(x.size(0))
            return x.clone()  # identity: out must reconstruct the input exactly

        st._forward_chunk_ll = fake_ll
        N, D, topk = 292, 2048, 6
        x = torch.randn(N, D, dtype=torch.bfloat16)
        w = torch.randn(N, topk, dtype=torch.float32)
        idx = torch.randint(0, 64, (N, topk), dtype=torch.int64)

        out = st.forward(x, w, idx)

        self.assertEqual(tuple(out.shape), (N, D))
        self.assertEqual(calls, [256, 36])  # ceil(292/256) chunks
        self.assertTrue(torch.equal(out, x))

    def test_decode_single_call_when_N_le_cap(self):
        st = self._make(cap=256)
        calls = []

        def fake_ll(wrapper, x, w, idx):
            calls.append(x.size(0))
            return x.clone()

        st._forward_chunk_ll = fake_ll
        x = torch.randn(5, 2048, dtype=torch.bfloat16)
        out = st.forward(
            x, torch.randn(5, 6), torch.randint(0, 64, (5, 6), dtype=torch.int64)
        )
        self.assertEqual(calls, [5])
        self.assertTrue(torch.equal(out, x))

    def test_chunk_output_copied_not_aliased(self):
        # _forward_chunk_ll returns a rotating shared buffer, like the LL combine
        # RDMA result (only ~2 live at once). forward must copy each chunk
        # immediately; otherwise later chunks corrupt earlier ones.
        st = self._make(cap=2)
        shared = torch.zeros(2, 2048, dtype=torch.bfloat16)

        def rotating(wrapper, x, w, idx):
            n = x.size(0)
            shared[:n].copy_(x)
            return shared[:n]

        st._forward_chunk_ll = rotating
        N = 5
        x = torch.randn(N, 2048, dtype=torch.bfloat16)
        out = st.forward(
            x, torch.randn(N, 6), torch.randint(0, 64, (N, 6), dtype=torch.int64)
        )
        self.assertTrue(
            torch.equal(out, x),
            "aliasing regression: chunks corrupted (missing immediate copy)",
        )

    def test_topk_mismatch_raises(self):
        st = self._make(cap=256, num_topk=8)
        st._forward_chunk_ll = lambda *a: None
        with self.assertRaises(RuntimeError):
            st.forward(
                torch.randn(3, 2048, dtype=torch.bfloat16),
                torch.randn(3, 6),
                torch.randint(0, 64, (3, 6), dtype=torch.int64),
            )

    def test_invalid_cap_raises(self):
        st = self._make(cap=0)
        st._forward_chunk_ll = lambda *a: None
        with self.assertRaises(RuntimeError):
            st.forward(
                torch.randn(3, 2048, dtype=torch.bfloat16),
                torch.randn(3, 6),
                torch.randint(0, 64, (3, 6), dtype=torch.int64),
            )



def _install_wrapper_stub_with_experts(cap: int, num_topk: int, num_experts: int) -> None:
    _install_wrapper_stub(cap, num_topk)
    wrapper = sys.modules[
        "rtp_llm.models_py.distributed.deepep_wrapper"
    ].DeepEPWrapper._instance
    type(wrapper).num_experts = num_experts


class _FakeStrategyLayer:
    """Minimal stand-in for ``MoE`` as seen by ``ll_chunk_align.begin_layer``."""

    def __init__(self, strategy, max_tokens_per_rank: int):
        self._strategy = strategy
        self.max_tokens_per_rank = max_tokens_per_rank


class LocalLlCallsFormulaTest(unittest.TestCase):
    """The mirror of the two prefill chunk loops must be exact.

    ``ll_chunk_align`` votes on this number *before* any dispatch runs, so a
    formula that disagrees with the loops would align the group on the wrong
    count.
    """

    def setUp(self):
        from rtp_llm.models_py.modules.dsv4.moe import ll_chunk_align

        self.mod = ll_chunk_align

    def test_single_chunk_when_tokens_below_cap(self):
        self.assertEqual(self.mod.local_ll_calls(1, 640, 16384, True), 1)
        self.assertEqual(self.mod.local_ll_calls(640, 640, 16384, True), 1)

    def test_live_hang_case_needs_two_calls(self):
        # The 08-19 hang: seq_len=1005 with GEN=3 (cap=640) issues 2 rounds
        # while the fake-prefill ranks issue 1.
        self.assertEqual(self.mod.local_ll_calls(1005, 640, 16384, True), 2)

    def test_moe_chunking_restarts_the_inner_loop(self):
        # Not ceil(tokens/cap): every MoE chunk pays a full round for its tail,
        # so a chunk size that is not a multiple of cap costs extra rounds.
        self.assertEqual(self.mod.local_ll_calls(2000, 640, 700, True), 5)
        self.assertEqual(self.mod.local_ll_calls(2000, 640, 700, False), 4)

    def test_degenerate_inputs(self):
        self.assertEqual(self.mod.local_ll_calls(0, 640, 16384, True), 0)
        self.assertEqual(self.mod.local_ll_calls(100, 0, 16384, True), 0)


class ChunkAlignmentTest(unittest.TestCase):
    """Ranks with different token counts must end up with equal dispatch counts."""

    def setUp(self):
        _install_wrapper_stub_with_experts(cap=640, num_topk=6, num_experts=64)
        from rtp_llm.models_py.modules.dsv4.moe import ll_chunk_align

        self.mod = ll_chunk_align
        self.mod.begin_forward(is_prefill=False)
        # Voting is only safe on an MTP serve (fake-prefill injection keeps
        # every rank in the prefill forward); emulate one for these tests.
        self.mod.set_mtp_enabled(True)

    def tearDown(self):
        self.mod.set_mtp_enabled(False)

    def _make_strategy(self, counter):
        from rtp_llm.models_py.modules.dsv4.moe.strategies.deepep_low_latency import (
            DeepEPLowLatencyStrategy,
        )

        st = DeepEPLowLatencyStrategy.__new__(DeepEPLowLatencyStrategy)

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.dim = 2048
        st.cfg = cfg
        st._W1_w = torch.zeros(1, dtype=torch.int8)

        def fake_ll(wrapper, x, w, idx):
            counter.append(x.size(0))
            self.mod.note_ll_call()
            return x.clone()

        st._forward_chunk_ll = fake_ll
        return st

    def _run_rank(self, tokens: int, global_calls: int, calls: list):
        """One rank's layer: vote (stubbed), real chunks, then alignment padding."""
        st = self._make_strategy(calls)
        layer = _FakeStrategyLayer(st, max_tokens_per_rank=16384)
        with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=8), \
             mock.patch.object(self.mod, "_vote_max", return_value=global_calls):
            self.mod.begin_forward(is_prefill=True)
            plan = self.mod.begin_layer(layer, tokens, False, torch.device("cpu"))
            x = torch.randn(tokens, 2048, dtype=torch.bfloat16)
            st.forward(
                x,
                torch.randn(tokens, 6),
                torch.randint(0, 64, (tokens, 6), dtype=torch.int64),
            )
            self.mod.finish_layer(plan)
        return calls

    def test_asymmetric_token_counts_are_padded_to_the_group_max(self):
        # cap=640: 1 token -> 1 round, 1005 -> 2, 4000 -> 7. Group max is 7, so
        # every rank must issue exactly 7 dispatch rounds.
        for tokens in (1, 1005, 4000):
            calls: list = []
            self._run_rank(tokens, global_calls=7, calls=calls)
            self.assertEqual(
                len(calls),
                7,
                f"tokens={tokens} issued {len(calls)} rounds, group expects 7",
            )

    def test_real_chunks_are_untouched_by_padding(self):
        calls: list = []
        self._run_rank(1005, global_calls=7, calls=calls)
        # Real work first (640 + 365), then single-token padding rounds.
        self.assertEqual(calls[:2], [640, 365])
        self.assertEqual(calls[2:], [1, 1, 1, 1, 1])

    def test_output_is_bit_exact_with_padding_enabled(self):
        st = self._make_strategy([])
        layer = _FakeStrategyLayer(st, max_tokens_per_rank=16384)
        x = torch.randn(1005, 2048, dtype=torch.bfloat16)
        w = torch.randn(1005, 6)
        idx = torch.randint(0, 64, (1005, 6), dtype=torch.int64)
        with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=8), \
             mock.patch.object(self.mod, "_vote_max", return_value=7):
            self.mod.begin_forward(is_prefill=True)
            plan = self.mod.begin_layer(layer, 1005, False, torch.device("cpu"))
            out = st.forward(x, w, idx)
            self.mod.finish_layer(plan)
        self.assertTrue(torch.equal(out, x))

    def test_decode_forward_never_votes(self):
        # Decode/verify may replay a captured graph on one rank and run eagerly
        # on another, so a collective there would not be rank-uniform.
        st = self._make_strategy([])
        layer = _FakeStrategyLayer(st, max_tokens_per_rank=16384)
        with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=8), \
             mock.patch.object(self.mod, "_vote_max", side_effect=AssertionError("voted")):
            self.mod.begin_forward(is_prefill=False)
            self.assertIsNone(
                self.mod.begin_layer(layer, 4000, False, torch.device("cpu"))
            )

    def test_single_rank_never_votes(self):
        st = self._make_strategy([])
        layer = _FakeStrategyLayer(st, max_tokens_per_rank=16384)
        with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=1), \
             mock.patch.object(self.mod, "_vote_max", side_effect=AssertionError("voted")):
            self.mod.begin_forward(is_prefill=True)
            self.assertIsNone(
                self.mod.begin_layer(layer, 4000, False, torch.device("cpu"))
            )

    def test_non_mtp_serve_never_votes(self):
        # Hardware-verified regression guard: a non-MTP serve injects no fake
        # prefill (mayAddFakeStream only adds a fake *decode* stream there), so
        # ranks without a context stream never reach the vote — ALIGN=1 on a
        # non-MTP serve deadlocked even a 1-request paris probe. The vote must
        # stay off until a DeepSeekV4MtpModel marks the process as MTP.
        self.mod.set_mtp_enabled(False)
        st = self._make_strategy([])
        layer = _FakeStrategyLayer(st, max_tokens_per_rank=16384)
        with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=8), \
             mock.patch.object(self.mod, "_vote_max", side_effect=AssertionError("voted")):
            self.mod.begin_forward(is_prefill=True)
            self.assertIsNone(
                self.mod.begin_layer(layer, 4000, False, torch.device("cpu"))
            )

    def test_disabled_by_env(self):
        st = self._make_strategy([])
        layer = _FakeStrategyLayer(st, max_tokens_per_rank=16384)
        with mock.patch.dict(os.environ, {"DSV4_LL_CHUNK_ALIGN": "0"}), \
             mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=8), \
             mock.patch.object(self.mod, "_vote_max", side_effect=AssertionError("voted")):
            self.mod.begin_forward(is_prefill=True)
            self.assertIsNone(
                self.mod.begin_layer(layer, 4000, False, torch.device("cpu"))
            )

    def test_non_ll_strategy_is_ignored(self):
        class _Other:
            name = "mega"

        layer = _FakeStrategyLayer(_Other(), max_tokens_per_rank=16384)
        with mock.patch.object(torch.distributed, "is_initialized", return_value=True), \
             mock.patch.object(torch.distributed, "get_world_size", return_value=8), \
             mock.patch.object(self.mod, "_vote_max", side_effect=AssertionError("voted")):
            self.mod.begin_forward(is_prefill=True)
            self.assertIsNone(
                self.mod.begin_layer(layer, 4000, False, torch.device("cpu"))
            )


class PadLlCallsTest(unittest.TestCase):
    def setUp(self):
        _install_wrapper_stub_with_experts(cap=640, num_topk=6, num_experts=64)

    def test_padding_uses_single_token_legal_routing(self):
        from rtp_llm.models_py.modules.dsv4.moe.strategies.deepep_low_latency import (
            DeepEPLowLatencyStrategy,
        )

        st = DeepEPLowLatencyStrategy.__new__(DeepEPLowLatencyStrategy)

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.dim = 2048
        st.cfg = cfg
        st._W1_w = torch.zeros(1, dtype=torch.int8)
        seen = []

        def fake_ll(wrapper, x, w, idx):
            seen.append((x.size(0), x.dtype, w.dtype, idx.dtype, int(idx.max())))
            return x.clone()

        st._forward_chunk_ll = fake_ll
        st.pad_ll_calls(3)

        self.assertEqual(len(seen), 3)
        for n, x_dtype, w_dtype, idx_dtype, max_expert in seen:
            self.assertEqual(n, 1)
            self.assertEqual(x_dtype, torch.bfloat16)
            self.assertEqual(w_dtype, torch.float32)
            self.assertEqual(idx_dtype, torch.int64)
            self.assertLess(max_expert, 64)

    def test_non_positive_count_is_noop(self):
        from rtp_llm.models_py.modules.dsv4.moe.strategies.deepep_low_latency import (
            DeepEPLowLatencyStrategy,
        )

        st = DeepEPLowLatencyStrategy.__new__(DeepEPLowLatencyStrategy)
        st._forward_chunk_ll = lambda *a: self.fail("padding ran for count<=0")
        st.pad_ll_calls(0)
        st.pad_ll_calls(-2)


if __name__ == "__main__":
    unittest.main()
