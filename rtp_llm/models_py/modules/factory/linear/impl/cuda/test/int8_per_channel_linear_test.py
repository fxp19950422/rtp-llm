from unittest import mock

import torch
from torch import nn

from rtp_llm.models_py.modules.factory.linear.impl.cuda import int8_per_channel_linear
from rtp_llm.models_py.modules.factory.linear.impl.cuda.int8_per_channel_linear import (
    Int8PerChannelLinear,
)


def test_deep_gemm_bias_add_reuses_fresh_output_storage():
    layer = Int8PerChannelLinear.__new__(Int8PerChannelLinear)
    nn.Module.__init__(layer)
    layer.weight = torch.zeros((4, 3), dtype=torch.int8)
    layer._w_scale_fp32 = torch.ones(4, dtype=torch.float32)
    layer.bias = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.bfloat16)
    gemm_outputs = []
    raw_gemm = torch.tensor(
        [[10.0, 20.0, 30.0, 40.0], [5.0, 6.0, 7.0, 8.0]],
        dtype=torch.bfloat16,
    )

    def fake_gemm(a, b, out):
        out.copy_(raw_gemm)
        gemm_outputs.append(out)

    layer._int8_gemm_nt = fake_gemm
    x = torch.tensor([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]], dtype=torch.bfloat16)

    with (
        mock.patch.object(
            int8_per_channel_linear, "_should_use_triton_quant", return_value=False
        ),
        mock.patch.object(int8_per_channel_linear, "_HAS_FUSED_QUANT", False),
        mock.patch.object(
            int8_per_channel_linear,
            "_should_use_triton_scaled_mm",
            return_value=False,
        ),
    ):
        output = layer._forward_deep_gemm(x)

    expected = raw_gemm + layer.bias
    torch.testing.assert_close(output, expected)
    assert output.data_ptr() == gemm_outputs[0].data_ptr()
