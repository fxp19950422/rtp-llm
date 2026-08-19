"""CUDA per-channel INT8 (W8A8) quantized Linear implementation.

Handles compressed-tensors ``int-quantized`` weights: int8 ``[out, in]`` weight
+ FP32 per-output-channel scale ``[out, 1]``.

Two forward paths, selected once by platform capability:

* **True W8A8 (PPU):** when the DeepGEMM dense INT8 kernel is available
  (:func:`~rtp_llm.models_py.modules.dsv4.int8_gemm.has_int8_dense_gemm`),
  the activation is quantized per-token to INT8 and multiplied against the
  int8 weight via ``gemm_int8_int8_bf16_nt`` -- the weight is read once. This
  exposes ``quantize_input`` / ``forward_quantized`` so DeepSeek-V4
  attention's shared-input-quant fast path
  (``_can_reuse_qkv_input_quant``) reuses a single per-token quantization
  across ``wq_a`` and ``wkv``.
* **Dequant fallback (CUDA/ROCm):** dequantizes the weight to the input
  dtype and runs a plain matmul (weights exact, activations stay in the
  input dtype). Used when the INT8 kernel is absent, or if the INT8 GEMM
  raises on first use (a one-time self-check that then latches to dequant).

Note: the fast path exposes ``quantize_input`` for reuse but NOT the FP8
attention norm-fusion (``can_fuse_prefill_attn_norm_input_quant``), which
additionally requires ``scale_ue8m0`` -- an FP8-only attribute this INT8
linear does not carry.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.factory.linear import LinearBase
from rtp_llm.ops import HWKernelConfig

logger = logging.getLogger(__name__)


class CudaInt8PerChannelLinear(LinearBase):
    """CUDA per-channel INT8 (W8A8) Linear.

    Prefers a true per-token INT8 GEMM (PPU) and falls back to on-the-fly
    weight dequant when the INT8 kernel is unavailable.
    """

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
        self.N, self.K = weight.shape
        # Normalise scale to fp32 contiguous [N, 1] -- the layout DeepGEMM's
        # dense int8 GEMM consumes directly as the ``rhs`` scale.
        self.weight_scale = weight_scales.reshape(self.N, 1).to(torch.float32).contiguous()
        self.bias = bias

        # Resolve the fast path once. ``_int8_gemm_ok`` latches to False if the
        # kernel raises on first real use (self-check + cache).
        from rtp_llm.models_py.modules.dsv4.int8_gemm import has_int8_dense_gemm

        self._int8_gemm_ok = has_int8_dense_gemm()

    def _dequant_forward(
        self, input: torch.Tensor, out: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Dequant [out, in] int8 -> input dtype via per-row scale, then matmul.
        w = (self.weight.to(torch.float32) * self.weight_scale).to(input.dtype)
        y = F.linear(input, w, self.bias)
        if out is not None:
            out.copy_(y)
            return out
        return y

    def quantize_input(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a 2D BF16 input once to per-token INT8 for reuse across
        INT8 GEMMs that share the same activation (``wq_a`` / ``wkv``)."""
        from rtp_llm.models_py.modules.dsv4.int8_gemm import quantize_per_token_int8

        x = input if input.dtype == torch.bfloat16 else input.to(torch.bfloat16)
        return quantize_per_token_int8(x.contiguous())

    def forward_quantized(
        self,
        input_i8: torch.Tensor,
        input_scales: torch.Tensor,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the INT8 GEMM against a caller-provided per-token quantized
        input. ``input_i8`` is 2D ``[M, K]`` int8, ``input_scales`` ``[M, 1]``
        fp32."""
        from rtp_llm.models_py.modules.dsv4.int8_gemm import int8_dense_gemm

        y = int8_dense_gemm(
            input_i8, input_scales, self.weight, self.weight_scale, out=out
        )
        if self.bias is not None:
            y = y.add_(self.bias.to(y.dtype)) if out is not None else y + self.bias.to(y.dtype)
        return y

    def forward(
        self, input: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self._int8_gemm_ok:
            return self._dequant_forward(input, out)
        # Fast W8A8 path: per-token quantize then INT8 GEMM. Collapse N-D
        # inputs to 2D [M, K] (the GEMM is 2D) and restore the leading shape.
        orig_shape = input.shape
        x_2d = input.reshape(-1, self.K)
        try:
            x_i8, x_scale = self.quantize_input(x_2d)
            y = self.forward_quantized(x_i8, x_scale)
        except Exception as exc:  # pragma: no cover - defensive self-check
            # One-time self-check: a shape/scale layout the kernel rejects
            # latches this linear back to the exact dequant path.
            logger.warning(
                "INT8 dense GEMM failed (%s); falling back to dequant matmul "
                "for this linear (N=%d, K=%d).",
                exc,
                self.N,
                self.K,
            )
            self._int8_gemm_ok = False
            return self._dequant_forward(input, out)
        y = y.reshape(*orig_shape[:-1], self.N)
        if out is not None:
            out.copy_(y)
            return out
        return y
