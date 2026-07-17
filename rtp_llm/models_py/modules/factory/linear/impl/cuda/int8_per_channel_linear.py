"""INT8 per-channel Linear implementation with deep_gemm INT8 GEMM acceleration.

On PPU (or any device with deep_gemm INT8 support), this uses hardware-accelerated
INT8 GEMM via deep_gemm's gemm_int8_int8_bf16_nt kernel:
  - Activation is dynamically quantized to INT8 per-token
  - Weight stays INT8 with per-channel scales
  - GEMM output is BF16

Fallback: on devices without deep_gemm, dequantizes INT8 weights to input dtype
on-the-fly and performs standard matmul (memory savings only).
"""

import logging
import os
from typing import Optional

import torch
from torch.nn import functional as F

from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig

logger = logging.getLogger(__name__)

try:
    from rtp_llm.ops.compute_ops import per_token_group_quant_int8

    _HAS_FUSED_QUANT = True
except (ImportError, AttributeError):
    _HAS_FUSED_QUANT = False
    logger.info(
        "per_token_group_quant_int8 not available, using PyTorch fallback for INT8 quant"
    )

try:
    from rtp_llm.models_py.kernels.cuda.sglang_triton_moe import int8_scaled_mm_triton

    _HAS_TRITON_SCALED_MM = True
except (ImportError, Exception):
    _HAS_TRITON_SCALED_MM = False

try:
    from rtp_llm.models_py.kernels.cuda.ppu_int8_quant import (
        per_token_quant_int8_triton,
    )

    _HAS_TRITON_QUANT = True
except (ImportError, Exception):
    _HAS_TRITON_QUANT = False

_USE_PPU_TRITON_INT8_QUANT = (
    _HAS_TRITON_QUANT
    and is_ppu()
    and os.environ.get("RTP_PPU_USE_TRITON_INT8_QUANT", "1").lower()
    in ("1", "true", "yes", "on")
)
_PPU_TRITON_INT8_QUANT_MIN_K = int(
    os.environ.get("RTP_PPU_TRITON_INT8_QUANT_MIN_K", "4096")
)
_USE_TRITON_SCALED_MM = _HAS_TRITON_SCALED_MM and os.environ.get(
    "RTP_PPU_USE_TRITON_INT8_SCALED_MM", "0"
).lower() in ("1", "true", "yes", "on")


def _should_use_triton_quant(input_tensor: torch.Tensor) -> bool:
    return (
        _USE_PPU_TRITON_INT8_QUANT
        and input_tensor.is_contiguous()
        and input_tensor.shape[-1] >= _PPU_TRITON_INT8_QUANT_MIN_K
    )


def _should_use_triton_scaled_mm(rows: int, bias: Optional[torch.Tensor]) -> bool:
    return _USE_TRITON_SCALED_MM and bias is not None and rows <= 128


class Int8PerChannelLinear(LinearBase):
    """INT8 per-channel Linear with optional deep_gemm INT8 GEMM acceleration."""

    @classmethod
    def can_handle(
        cls,
        quant_config: object,
        weight: torch.Tensor,
        weight_scales: Optional[torch.Tensor],
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        weight_scale_2: Optional[torch.Tensor] = None,
        input_scale: Optional[torch.Tensor] = None,
    ) -> bool:
        if quant_config is None:
            return False
        if weight.dtype != torch.int8:
            return False
        if weight_scales is None:
            return False
        quant_method = getattr(quant_config, "get_method", None)
        if quant_method is None:
            return False
        return quant_method() == "INT8_PER_CHANNEL_COMPRESSED"

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
        self.weight = weight
        self.weight_scales = weight_scales
        self.bias = bias

        self._use_deep_gemm = False
        self._int8_gemm_nt = None
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                has_deep_gemm,
                int8_gemm_nt,
            )

            if has_deep_gemm():
                self._use_deep_gemm = True
                self._int8_gemm_nt = int8_gemm_nt
                logger.info("Int8PerChannelLinear: using deep_gemm INT8 GEMM")
        except Exception as e:
            logger.debug(
                f"Int8PerChannelLinear: deep_gemm not available, using fallback: {e}"
            )

        if self.weight_scales is not None:
            self._w_scale_fp32 = self.weight_scales.reshape(-1).to(torch.float32)
        else:
            self._w_scale_fp32 = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._use_deep_gemm:
            return self._forward_deep_gemm(input)
        scale = self.weight_scales.reshape(-1, 1).to(input.dtype)
        weight_dequant = self.weight.to(input.dtype) * scale
        return F.linear(input, weight_dequant, self.bias)

    def _forward_deep_gemm(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        x = input.reshape(-1, orig_shape[-1])
        M, K = x.shape
        N = self.weight.shape[0]

        if _should_use_triton_quant(x):
            x_int8, x_scale = per_token_quant_int8_triton(x)
            x_scale = x_scale.squeeze(-1)
        elif _HAS_FUSED_QUANT and x.is_contiguous():
            x_int8 = torch.empty_like(x, dtype=torch.int8)
            x_scale = torch.empty(M, 1, device=x.device, dtype=torch.float32)
            per_token_group_quant_int8(
                x,
                x_int8,
                x_scale,
                group_size=K,
                eps=1e-12,
                int8_min=-128.0,
                int8_max=127.0,
                scale_ue8m0=False,
            )
            x_scale = x_scale.squeeze(-1)
        else:
            x_max = x.abs().amax(dim=-1, keepdim=False)
            x_scale = (x_max / 127.0).clamp(min=1e-12).to(torch.float32)
            x_int8 = (x / x_scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)

        w_scale = self._w_scale_fp32

        if _should_use_triton_scaled_mm(M, self.bias):
            out = int8_scaled_mm_triton(
                x_int8,
                self.weight,
                x_scale,
                w_scale,
                bias=(
                    self.bias.to(torch.bfloat16)
                    if self.bias.dtype != torch.bfloat16
                    else self.bias
                ),
            )
            return out.reshape(*orig_shape[:-1], N)

        out = torch.empty(M, N, dtype=torch.bfloat16, device=input.device)
        self._int8_gemm_nt((x_int8, x_scale), (self.weight, w_scale), out)

        if self.bias is not None:
            # ``out`` is a fresh BF16 GEMM destination.  Reuse it for bias so
            # long-prefill QKV projection does not transiently hold two full
            # [tokens, qkv_width] buffers.
            out.add_(self.bias)

        return out.reshape(*orig_shape[:-1], N)
