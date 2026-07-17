import types
from unittest import mock

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy import (
    int8_per_channel,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.strategy.int8_per_channel import (
    Int8PerChannelEpNormalExecutor,
)


def _make_executor():
    executor = Int8PerChannelEpNormalExecutor.__new__(Int8PerChannelEpNormalExecutor)
    executor._use_deep_gemm = True
    executor.has_int8 = True
    executor.start_expert_id = 0
    executor.end_expert_id = 1
    executor.num_experts_per_partition = 2
    executor.E = 2
    return executor


def _make_payload(rows: int) -> ExpertForwardPayload:
    return ExpertForwardPayload(
        expert_x=torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3),
        expert_topk_ids=torch.zeros((rows, 1), dtype=torch.int64),
        expert_topk_weights=torch.ones((rows, 1), dtype=torch.float32),
    )


def test_ep_normal_prefill_chunks_before_route_selection():
    executor = _make_executor()
    payload = _make_payload(5)
    chunk_rows = []

    def fake_grouped_gemm(
        self,
        hidden_states,
        token_ids,
        local_expert_ids,
        route_weights,
        grouped_gemm_fn,
    ):
        chunk_rows.append(hidden_states.shape[0])
        assert token_ids.max().item() < hidden_states.shape[0]
        return CombineForwardPayload(fused_expert_output=hidden_states + 7)

    executor._execute_local_routes_deep_gemm = types.MethodType(
        fake_grouped_gemm, executor
    )

    with (
        mock.patch.object(int8_per_channel, "_PREFILL_CHUNK_SIZE", 2),
        mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ),
    ):
        result = executor.execute(payload, "silu", None, None, False, None)

    assert chunk_rows == [2, 2, 1]
    torch.testing.assert_close(result.fused_expert_output, payload.expert_x + 7)


def test_ep_normal_decode_and_graph_shapes_are_not_chunked():
    for rows, capturing in ((1, False), (5, True)):
        executor = _make_executor()
        payload = _make_payload(rows)
        calls = []

        def fake_grouped_gemm(
            self,
            hidden_states,
            token_ids,
            local_expert_ids,
            route_weights,
            grouped_gemm_fn,
        ):
            calls.append(hidden_states.shape[0])
            return CombineForwardPayload(fused_expert_output=hidden_states)

        executor._execute_local_routes_deep_gemm = types.MethodType(
            fake_grouped_gemm, executor
        )
        with (
            mock.patch.object(int8_per_channel, "_PREFILL_CHUNK_SIZE", 2),
            mock.patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                return_value=capturing,
            ),
        ):
            executor.execute(payload, "silu", None, None, False, None)
        assert calls == [rows]


def test_ep_normal_grouped_gemm_oom_does_not_fallback():
    executor = _make_executor()
    payload = _make_payload(1)

    def raise_oom(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    def fail_if_fallback(*args, **kwargs):
        raise AssertionError("OOM must not enter the fallback path")

    executor._execute_local_routes_deep_gemm = raise_oom
    executor._execute_fallback = fail_if_fallback

    with (
        mock.patch.object(int8_per_channel, "_PREFILL_CHUNK_SIZE", 0),
        mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ),
        mock.patch.object(int8_per_channel.logger, "warning") as warning,
    ):
        try:
            executor.execute(payload, "silu", None, None, False, None)
        except RuntimeError as error:
            assert "out of memory" in str(error).lower()
        else:
            raise AssertionError("expected grouped-GEMM OOM to be propagated")

    warning.assert_not_called()
