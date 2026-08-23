import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPMode, DeepEPWrapper
from rtp_llm.models_py.modules.dsv4.moe.strategies.base import MoeCfg
from rtp_llm.models_py.modules.dsv4.moe.strategies.deepep import DeepEPStrategy


class _FakeBuffer:
    def __init__(self) -> None:
        self.num_worst_tokens = None

    def get_dispatch_layout(self, indices, _num_experts):
        rows = indices.size(0)
        return (
            torch.zeros(8, dtype=torch.int32),
            None,
            torch.zeros(256, dtype=torch.int32),
            torch.zeros(rows, 8, dtype=torch.bool),
            None,
        )

    def dispatch(self, x, *_args, num_worst_tokens=0, **_kwargs):
        self.num_worst_tokens = num_worst_tokens
        rows = num_worst_tokens or x.size(0)
        recv_x = torch.zeros(rows, x.size(1), dtype=x.dtype)
        recv_idx = torch.full((rows, 8), -1, dtype=torch.int64)
        recv_weights = torch.zeros(rows, 8, dtype=torch.float32)
        return recv_x, recv_idx, recv_weights, [], object(), None

    def combine(self, y_local, _handle):
        return y_local[:1].to(torch.bfloat16), None, None


class _FakeLocal:
    def _forward_into_buf(self, x, _weights, _indices, **_kwargs):
        return torch.zeros(x.size(0), x.size(1), dtype=torch.float32)


def _strategy():
    strategy = DeepEPStrategy.__new__(DeepEPStrategy)
    torch.nn.Module.__init__(strategy)
    strategy.cfg = MoeCfg(
        layer_id=0,
        dim=4,
        moe_inter_dim=8,
        n_routed_experts=256,
        n_activated_experts=6,
        swiglu_limit=1.0,
        ep_size=8,
        ep_rank=0,
        n_local_experts=32,
        local_expert_start=0,
        local_expert_end=32,
        max_tokens_per_rank=1,
    )
    strategy._local = _FakeLocal()
    return strategy


class DeepEPGraphDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = _FakeBuffer()
        self.saved_instance = DeepEPWrapper._instance
        DeepEPWrapper._instance = SimpleNamespace(
            mode=DeepEPMode.NORMAL, buffer=self.buffer
        )
        self.x = torch.zeros(1, 4, dtype=torch.bfloat16)
        self.weights = torch.ones(1, 6, dtype=torch.float32)
        self.indices = torch.arange(6, dtype=torch.int64).view(1, 6)

    def tearDown(self) -> None:
        DeepEPWrapper._instance = self.saved_instance
        os.environ.pop("RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD", None)

    def test_capture_uses_fixed_worst_case_receive_capacity(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), patch(
            "torch.cuda.is_current_stream_capturing", return_value=True
        ):
            out = _strategy()(self.x, self.weights, self.indices)

        self.assertEqual(self.buffer.num_worst_tokens, 8)
        self.assertEqual(tuple(out.shape), (1, 4))

    def test_graph_warmup_syncs_and_uses_capture_shape(self) -> None:
        os.environ["RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD"] = "1"
        with patch("torch.cuda.is_available", return_value=False), patch(
            "rtp_llm.models_py.modules.dsv4.moe.strategies.deepep.sync_cuda_graph_warmup_ranks"
        ) as sync:
            _strategy()(self.x, self.weights, self.indices)

        self.assertEqual(self.buffer.num_worst_tokens, 8)
        self.assertEqual(
            [call.args[0] for call in sync.call_args_list],
            ["deepep_before_dispatch", "deepep_after_combine"],
        )

    def test_eager_keeps_dynamic_receive_path(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            _strategy()(self.x, self.weights, self.indices)

        self.assertEqual(self.buffer.num_worst_tokens, 0)


if __name__ == "__main__":
    unittest.main()
