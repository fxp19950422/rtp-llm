from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from rtp_llm.models_py.modules.hybrid import dense_mlp
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.ops import ActivationType


class _RecordingProjection(nn.Module):
    def __init__(self, calls, transform):
        super().__init__()
        self.calls = calls
        self.transform = transform

    def forward(self, x):
        self.calls.append(x.shape[0])
        return self.transform(x)


def _make_mlp(ffn_tp_size=1):
    up_calls = []
    down_calls = []
    mlp = DenseMLP.__new__(DenseMLP)
    nn.Module.__init__(mlp)
    mlp.activation_type = ActivationType.Swiglu
    mlp.is_gated = True
    mlp.up_proj = _RecordingProjection(up_calls, lambda x: x + 3)
    mlp.act_fn = nn.Identity()
    mlp.down_proj = _RecordingProjection(down_calls, lambda x: x * 2)
    mlp.parallelism_config = SimpleNamespace(get_ffn_tp_size=lambda: ffn_tp_size)
    return mlp, up_calls, down_calls


def test_dense_mlp_prefill_chunks_local_compute_but_allreduces_once():
    mlp, up_calls, down_calls = _make_mlp(ffn_tp_size=2)
    x = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    allreduce_rows = []

    def fake_all_reduce(output, group):
        allreduce_rows.append(output.shape[0])
        return output

    with (
        mock.patch.object(dense_mlp, "_PREFILL_CHUNK_SIZE", 2),
        mock.patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False
        ),
        mock.patch.object(dense_mlp, "all_reduce", side_effect=fake_all_reduce),
    ):
        output = mlp(x)

    assert up_calls == [2, 2, 1]
    assert down_calls == [2, 2, 1]
    assert allreduce_rows == [5]
    torch.testing.assert_close(output, (x + 3) * 2)


def test_dense_mlp_decode_and_graph_shapes_are_not_chunked():
    for rows, capturing in ((1, False), (5, True)):
        mlp, up_calls, down_calls = _make_mlp()
        x = torch.zeros((rows, 3))
        with (
            mock.patch.object(dense_mlp, "_PREFILL_CHUNK_SIZE", 2),
            mock.patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                return_value=capturing,
            ),
        ):
            mlp(x)
        assert up_calls == [rows]
        assert down_calls == [rows]
