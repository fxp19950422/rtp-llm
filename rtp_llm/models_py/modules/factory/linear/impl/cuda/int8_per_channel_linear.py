"""CUDA per-channel INT8 (W8A8) quantized Linear implementation.

Handles compressed-tensors ``int-quantized`` weights: int8 ``[out, in]`` weight
+ FP32 per-output-channel scale ``[out, 1]``. The forward dequantizes the weight
to the input dtype and runs a plain matmul (weights are exact; activations stay
in the input dtype). The fused deep_gemm ``gemm_int8_int8_bf16_nt`` path
(per-token int8 activation quant) is a tracked perf follow-on.

No ``quantize_input`` / ``forward_quantized`` methods are exposed, so the
DeepSeek-V4 attention's FP8 shared-input-quant fast path (guarded by
``hasattr(linear, "quantize_input")``) automatically falls back to the generic
per-linear path when these int8 linears are used.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig

logger = logging.getLogger(__name__)


class CudaInt8PerChannelLinear(LinearBase):
    """CUDA per-channel INT8 (W8A8) Linear via on-the-fly weight dequant."""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional['HWKernelConfig'] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if weight_scales is None or quant_config is None:
            return False
        if weight.dtype != torch.int8:
            return False
        return quant_config.get_method() == "INT8_PER_CHANNEL_COMPRESSED"

    def __init__(
        self,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
        quant_config: object = None,
        weight_scale_2: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            weight, weight_scales, input_scales, bias, quant_config, weight_scale_2
        )
        # weight stays [out, in] int8; scale is per-output-channel [out, 1] fp32.
        self.weight = weight
        self.weight_scale = weight_scales.reshape(weight.shape[0], 1)
        self.bias = bias

    def forward(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Dequant [out, in] int8 -> input dtype via per-row scale, then matmul.
        w = (self.weight.to(torch.float32) * self.weight_scale).to(input.dtype)
        y = F.linear(input, w, self.bias)
        if out is not None:
            out.copy_(y)
            return out
        return y
