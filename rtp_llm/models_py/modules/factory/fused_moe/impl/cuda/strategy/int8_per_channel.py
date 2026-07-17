"""INT8 per-channel MoE strategy for PPU/non-CUDA devices

Supports two execution paths:
1. **deep_gemm INT8 grouped GEMM** (if available): uses
   `m_grouped_int8_gemm_nt_masked` for hardware-accelerated INT8 GEMM.
2. **Fallback**: per-expert dequantization (INT8 → BF16) with torch.mm.

Weights are kept as INT8 with per-channel FP32 scales in memory, saving
~50% storage vs full BF16.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch

from rtp_llm.device.device_type import is_ppu
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import (
    ExecutorType,
    RouterType,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
    PureTpRouterBase,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)

logger = logging.getLogger(__name__)

_PREFILL_CHUNK_SIZE = int(os.environ.get("MOE_INT8_PREFILL_CHUNK_SIZE", "0") or "0")
_PPU_INT8_MOE_BACKEND = os.environ.get(
    "RTP_PPU_INT8_MOE_BACKEND", "deepgemm_nopad"
).lower()

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

# Try to import fused CUDA quantization kernels (available on CUDA and PPU)
try:
    from rtp_llm.ops.compute_ops import per_token_group_quant_int8

    _HAS_FUSED_QUANT = True
except (ImportError, AttributeError):
    _HAS_FUSED_QUANT = False
    logger.info(
        "per_token_group_quant_int8 not available, using PyTorch fallback for INT8 MoE quant"
    )

# Try to import v2 fused kernel (SiLU+mul+INT8 quant in 1 kernel)
try:
    from rtp_llm.ops.compute_ops import per_token_group_quant_int8_v2

    _HAS_FUSED_QUANT_V2 = True
except (ImportError, AttributeError):
    _HAS_FUSED_QUANT_V2 = False
    logger.info(
        "per_token_group_quant_int8_v2 not available, using separate SiLU+mul+quant"
    )

try:
    from rtp_llm.ops.compute_ops import silu_mul_quant_int8_glm

    _HAS_GLM_FUSED_SILU_MUL_QUANT = True
except (ImportError, AttributeError):
    _HAS_GLM_FUSED_SILU_MUL_QUANT = False
    logger.info("silu_mul_quant_int8_glm not available, using separate SiLU+mul+quant")


def _should_use_triton_quant(input_tensor: torch.Tensor) -> bool:
    return (
        _USE_PPU_TRITON_INT8_QUANT
        and input_tensor.is_contiguous()
        and input_tensor.shape[-1] >= _PPU_TRITON_INT8_QUANT_MIN_K
    )


def _per_token_quant_int8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activation to INT8 with per-token (per-row) scales.

    Uses fused CUDA kernel `per_token_group_quant_int8` (1 kernel launch)
    instead of 5 separate PyTorch ops when available.

    Args:
        x: BF16/FP16 activation, shape [M, K]
    Returns:
        (x_int8, x_scale) where:
          x_int8: INT8 tensor [M, K]
          x_scale: FP32 tensor [M, 1] (per-token max-abs scale)
    """
    if _should_use_triton_quant(x):
        return per_token_quant_int8_triton(x)
    if _HAS_FUSED_QUANT and x.is_cuda and x.is_contiguous():
        M, K = x.shape
        x_int8 = torch.empty_like(x, dtype=torch.int8)
        x_scale = torch.empty(M, 1, device=x.device, dtype=torch.float32)
        per_token_group_quant_int8(
            x,
            x_int8,
            x_scale,
            group_size=K,  # per-token = whole row is one group
            eps=1e-12,
            int8_min=-128.0,
            int8_max=127.0,
            scale_ue8m0=False,
        )
        return x_int8, x_scale
    else:
        # Fallback: PyTorch ops (5 kernel launches)
        x_max = x.abs().amax(dim=-1, keepdim=True)  # [M, 1]
        x_scale = (x_max / 127.0).clamp(min=1e-12).to(torch.float32)
        x_int8 = (x / x_scale).round().clamp(-128, 127).to(torch.int8)
        return x_int8, x_scale


def _fused_silu_mul_quant_int8(
    gemm1_output: torch.Tensor,
    inter_size: int,
    masked_m: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused SiLU+mul+INT8 quantization using v2 kernel.

    Replaces 3 separate ops (silu + mul + per_token_quant) with 1 fused kernel.

    GLM4 convention: GEMM1 output is [up_proj, gate_proj] (up first, gate second).
    v2 kernel convention: silu(first_half) * second_half = silu(gate) * up.
    So we swap halves before calling v2 kernel to match its convention.

    Args:
        gemm1_output: [E, M, 2*inter] or [E*M, 2*inter] BF16 tensor
        inter_size: inter dimension (half of last dim)
        masked_m: optional [E] int32 tensor for masked layout (real tokens per expert)
    Returns:
        (b_int8, b_scale) where:
          b_int8: INT8 tensor [..., inter]
          b_scale: FP32 tensor [..., 1]
    """
    # Swap halves: [up, gate] -> [gate, up] to match v2 kernel's silu(gate)*up convention
    swapped = torch.cat(
        [gemm1_output[..., inter_size:], gemm1_output[..., :inter_size]], dim=-1
    )

    # Flatten to 2D for v2 kernel: [E*M, 2*inter]
    orig_shape = swapped.shape
    swapped_flat = swapped.reshape(-1, 2 * inter_size)
    M = swapped_flat.shape[0]

    b_int8 = torch.empty(M, inter_size, device=swapped_flat.device, dtype=torch.int8)
    b_scale = torch.empty(M, 1, device=swapped_flat.device, dtype=torch.float32)

    # The v2 kernel's masked_layout path requires 3D output tensors
    # (output_s.dim() == 3 when masked_m is not None). Since we flatten
    # to 2D [E*M, inter], we must NOT pass masked_m — otherwise the
    # TORCH_CHECK(output_s.dim() == (masked_layout ? 3 : 2)) assertion fails.
    # With scale_ue8m0=False, the kernel uses NaiveScheduler regardless of
    # masked_m, so skipping it has zero performance impact. Padding tokens
    # (from torch.zeros in _execute_deep_gemm_impl) quantize to zero harmlessly.
    per_token_group_quant_int8_v2(
        swapped_flat,
        b_int8,
        b_scale,
        group_size=inter_size,
        eps=1e-10,
        int8_min=-128.0,
        int8_max=127.0,
        scale_ue8m0=False,
        fuse_silu_and_mul=True,
        masked_m=None,
    )

    # Reshape back to original batch dims
    b_int8 = b_int8.reshape(*orig_shape[:-1], inter_size)
    b_scale = b_scale.reshape(*orig_shape[:-1], 1)
    return b_int8, b_scale


def _glm_fused_silu_mul_quant_int8(
    gemm1_output: torch.Tensor,
    inter_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse GLM [up, gate] activation and per-row INT8 quantization."""
    flat = gemm1_output.reshape(-1, 2 * inter_size)
    output_q = torch.empty(
        flat.shape[0], inter_size, device=flat.device, dtype=torch.int8
    )
    output_s = torch.empty(flat.shape[0], 1, device=flat.device, dtype=torch.float32)
    silu_mul_quant_int8_glm(flat, output_q, output_s, inter_size, 1e-12, -128.0, 127.0)
    return (
        output_q.reshape(*gemm1_output.shape[:-1], inter_size),
        output_s.reshape(*gemm1_output.shape[:-1], 1),
    )


class PureTpRouterInt8PerChannel(PureTpRouterBase):
    """Pure TP router for INT8 per-channel quantization (no input quantization)."""

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config, do_recompute_topk=False)

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if this router can handle the configuration"""
        super().check_conditions(checker, config)
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "INT8_PER_CHANNEL_COMPRESSED")

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """No input quantization for weight-only INT8 per-channel"""
        return a1, None


class Int8PerChannelDequantExecutor(FusedMoeExpertExecutor):
    """MoE executor for INT8 per-channel weights.

    Stores INT8 weights + FP32 scales in memory (~50% savings vs BF16).
    Execution path (ordered by M size):
      1. Triton fused MoE (decode, M <= _TRITON_FUSED_M_THRESHOLD)
      2. deep_gemm INT8 grouped GEMM (prefill, large M)
      3. Fallback: per-expert on-the-fly dequantization + torch.mm
    """

    # Triton fused MoE is optimized for decode (small M, memory-bound).
    # For prefill (large M), deep_gemm's INT8 Tensor Cores are faster.
    # Triton fused MoE is now correct (bug fixed: GEMM2 activation indexing).
    # Benchmark on PPU: Triton fused 5.7 tok/s vs deep_gemm 6.1 tok/s for decode.
    # deep_gemm is ~7% faster on PPU, so keep threshold=0 to always use deep_gemm.
    # Triton fused path remains available as fallback if deep_gemm is unavailable.
    _TRITON_FUSED_M_THRESHOLD = (
        0  # deep_gemm is faster on PPU (A/B Test: 18.81 vs 11.1 tok/s with CUDA Graph)
    )

    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "INT8_PER_CHANNEL_COMPRESSED")

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        from rtp_llm.utils.model_weight import W

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
        self.top_k = config.moe_k

        # Store weights and scales
        self.w1 = weights[W.moe_w1]  # [E, 2*inter, hidden] INT8 or BF16
        self.w2 = weights[W.moe_w2]  # [E, hidden, inter] INT8 or BF16

        self.w1_scale = weights.get(W.moe_s1)  # [E, 2*inter] FP32 or None
        self.w2_scale = weights.get(W.moe_s2)  # [E, hidden] FP32 or None

        self.E = self.w1.size(0)
        self.N = self.w1.size(1)  # 2 * inter_size
        self.K = self.w1.size(2)  # hidden_size
        self.inter_size = self.N // 2

        self.has_int8 = self.w1.dtype == torch.int8 and self.w1_scale is not None

        # Try to use deep_gemm INT8 grouped GEMM
        self._use_deep_gemm = False
        self._use_deep_gemm_nopad = False
        if self.has_int8:
            try:
                from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                    grouped_gemm_nt_i8i8bf16_nopad,
                    has_deep_gemm,
                    m_grouped_int8_gemm_nt_masked,
                )

                if has_deep_gemm():
                    self._use_deep_gemm = True
                    self._use_deep_gemm_nopad = (
                        _PPU_INT8_MOE_BACKEND == "deepgemm_nopad"
                        and grouped_gemm_nt_i8i8bf16_nopad is not None
                    )
                    logger.info(
                        "INT8 MoE executor: using deep_gemm INT8 grouped GEMM "
                        f"(backend={_PPU_INT8_MOE_BACKEND}, "
                        f"nopad={self._use_deep_gemm_nopad})"
                    )
            except Exception as e:
                logger.warning(f"deep_gemm INT8 not available, using fallback: {e}")

        # Try to use Triton fused MoE kernel (port of SGLang fused_moe)
        # This is the highest-performance path: 5-6 kernel launches vs 15+
        self._use_triton_fused = False
        if self.has_int8:
            try:
                import triton  # noqa: F401

                from rtp_llm.models_py.kernels.cuda.sglang_triton_moe import (  # noqa: F401
                    fused_moe_int8_forward,
                )

                self._use_triton_fused = True
                logger.info("INT8 MoE executor: Triton fused MoE kernel available")
            except (ImportError, Exception) as e:
                logger.info(f"Triton fused MoE not available: {e}")

        # Pre-convert weight scales to FP32 and reshape to [E, N, 1]
        # (avoids per-forward .to(float32) and .unsqueeze(-1) calls)
        if self.w1_scale is not None:
            self._w1_scale_fp32 = self.w1_scale.to(torch.float32)
            if self._w1_scale_fp32.dim() == 2:
                self._w1_scale_fp32 = self._w1_scale_fp32.unsqueeze(-1)
        else:
            self._w1_scale_fp32 = None
        if self.w2_scale is not None:
            self._w2_scale_fp32 = self.w2_scale.to(torch.float32)
            if self._w2_scale_fp32.dim() == 2:
                self._w2_scale_fp32 = self._w2_scale_fp32.unsqueeze(-1)
        else:
            self._w2_scale_fp32 = None

        logger.info(
            f"INT8 per-channel MoE executor: "
            f"E={self.E}, N={self.N}, K={self.K}, inter_size={self.inter_size}, "
            f"w1_dtype={self.w1.dtype}, w2_dtype={self.w2.dtype}, "
            f"has_w1_scale={self.w1_scale is not None}, "
            f"has_w2_scale={self.w2_scale is not None}, "
            f"has_int8={self.has_int8}, use_deep_gemm={self._use_deep_gemm}, "
            f"use_deep_gemm_nopad={self._use_deep_gemm_nopad}, "
            f"use_triton_fused={self._use_triton_fused}"
        )

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        """MoE forward with INT8 weights.

        If deep_gemm INT8 grouped GEMM is available, uses it for hardware-
        accelerated INT8 computation. Otherwise falls back to per-expert
        dequantization + torch.mm.
        """
        hidden_states = payload.expert_x  # [M, K]
        topk_ids = payload.expert_topk_ids  # [M, top_k]
        topk_weights = payload.expert_topk_weights  # [M, top_k]

        assert topk_ids is not None
        assert topk_weights is not None

        if self.ep_size > 1 and not payload.expert_ids_are_local:
            local_ids = topk_ids - self.start_expert_id
            local_mask = (local_ids >= 0) & (local_ids < self.num_experts_per_partition)
            if self._use_deep_gemm_nopad:
                topk_ids = torch.where(
                    local_mask, local_ids, torch.full_like(local_ids, -1)
                )
            else:
                topk_ids = local_ids.clamp(
                    min=0, max=self.num_experts_per_partition - 1
                )
            topk_weights = topk_weights * local_mask

        _is_capturing = torch.cuda.is_current_stream_capturing()

        if self._use_triton_fused and self.has_int8:
            M = hidden_states.shape[0]
            # Triton fused path is optimized for decode (small M, memory-bound).
            # For prefill (large M), deep_gemm's INT8 Tensor Cores are faster.
            #
            # During CUDA Graph capture, we MUST use the same path as runtime
            # (deep_gemm). Trying Triton fused first and falling back to deep_gemm
            # on failure would pollute the capture buffer with partial Triton ops,
            # corrupting the captured graph. Deep_gemm is now graph-safe after
            # fixing: bincount→scatter_add_, .item() bypass, masked_m=None.
            #
            # _TRITON_FUSED_M_THRESHOLD=0 means Triton fused is disabled on PPU
            # (deep_gemm is 70% faster: 18.81 vs 11.1 tok/s with CUDA Graph).
            if not _is_capturing and M <= self._TRITON_FUSED_M_THRESHOLD:
                try:
                    return self._execute_triton_fused(
                        hidden_states, topk_ids, topk_weights
                    )
                except Exception as e:
                    logger.warning(
                        f"Triton fused MoE failed, falling back to deep_gemm: {e}"
                    )
            # else: fall through to deep_gemm for prefill (large M) or capture

        if self._use_deep_gemm_nopad and self.has_int8:
            try:
                if (
                    _PREFILL_CHUNK_SIZE > 0
                    and not torch.cuda.is_current_stream_capturing()
                    and hidden_states.shape[0] > _PREFILL_CHUNK_SIZE
                ):
                    return self._execute_prefill_chunks(
                        self._execute_deep_gemm_nopad,
                        hidden_states,
                        topk_ids,
                        topk_weights,
                    )
                return self._execute_deep_gemm_nopad(
                    hidden_states, topk_ids, topk_weights
                )
            except Exception as e:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "deep_gemm INT8 nopad MoE failed during CUDA Graph capture "
                        f"and fallback is not graph-safe: {e} | "
                        f"M={hidden_states.shape[0]}, top_k={topk_ids.shape[1]}, "
                        f"E={self.E}"
                    ) from e
                logger.warning(
                    "deep_gemm INT8 nopad MoE failed, using masked fallback: "
                    f"{e} | M={hidden_states.shape[0]}, "
                    f"top_k={topk_ids.shape[1]}, E={self.E}"
                )

        if self._use_deep_gemm and self.has_int8:
            return self._execute_deep_gemm(hidden_states, topk_ids, topk_weights)
        else:
            if _is_capturing:
                raise RuntimeError(
                    "INT8 MoE fallback path is not graph-safe (contains nonzero + "
                    f"per-expert Python loop) and cannot be used during CUDA Graph "
                    f"capture (deep_gemm unavailable) | M={hidden_states.shape[0]}, E={self.E}"
                )
            return self._execute_fallback(hidden_states, topk_ids, topk_weights)

    def _execute_prefill_chunks(
        self,
        execute_fn,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> CombineForwardPayload:
        """Bound non-Graph prefill workspaces before entering a MoE backend."""
        output = torch.empty_like(hidden_states)
        for start in range(0, hidden_states.shape[0], _PREFILL_CHUNK_SIZE):
            end = min(start + _PREFILL_CHUNK_SIZE, hidden_states.shape[0])
            chunk_payload = execute_fn(
                hidden_states[start:end],
                topk_ids[start:end],
                topk_weights[start:end],
            )
            output[start:end].copy_(chunk_payload.fused_expert_output)
        return CombineForwardPayload(fused_expert_output=output)

    def _execute_deep_gemm_nopad(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> CombineForwardPayload:
        """SGLang-style contiguous nopad INT8 MoE path for PPU."""
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            grouped_gemm_nt_i8i8bf16_nopad,
        )
        from rtp_llm.models_py.kernels.cuda.ppu_deepgemm_moe import (
            deepgemm_moe_permute,
            deepgemm_unpermute_and_reduce,
        )

        M, K = hidden_states.shape
        a_int8, a_scale = _per_token_quant_int8(hidden_states)
        a, a_scale, expert_ids, inv_perm, expert_num_tokens = deepgemm_moe_permute(
            a_int8,
            a_scale,
            topk_ids.to(torch.int32),
            self.E,
            block_align=1,
        )

        output1 = torch.empty(
            (a.shape[0], self.N), device=hidden_states.device, dtype=torch.bfloat16
        )
        grouped_gemm_nt_i8i8bf16_nopad(
            (a, a_scale),
            (self.w1, self._w1_scale_fp32),
            output1,
            expert_ids,
            expert_num_tokens,
        )

        if _HAS_GLM_FUSED_SILU_MUL_QUANT:
            b_int8, b_scale = _glm_fused_silu_mul_quant_int8(output1, self.inter_size)
        else:
            # Current GLM weight layout is [up_proj, gate_proj].
            up = output1[:, : self.inter_size]
            gate = output1[:, self.inter_size :]
            intermediate = torch.nn.functional.silu(gate) * up
            b_int8, b_scale = _per_token_quant_int8(intermediate)

        output2 = torch.empty(
            (a.shape[0], K), device=hidden_states.device, dtype=torch.bfloat16
        )
        grouped_gemm_nt_i8i8bf16_nopad(
            (b_int8, b_scale),
            (self.w2, self._w2_scale_fp32),
            output2,
            expert_ids,
            expert_num_tokens,
        )

        output = torch.empty((M, K), device=hidden_states.device, dtype=torch.bfloat16)
        deepgemm_unpermute_and_reduce(
            output2,
            topk_ids.to(torch.int32),
            topk_weights,
            inv_perm,
            output,
        )
        return CombineForwardPayload(fused_expert_output=output.to(hidden_states.dtype))

    def _execute_triton_fused(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> CombineForwardPayload:
        """Use Triton fused MoE kernel (port of SGLang fused_moe).

        This is the highest-performance path:
        - moe_align_block_size (token sorting, no CPU-GPU sync)
        - per_token_group_quant_int8 (1 CUDA kernel)
        - invoke_fused_moe_int8_kernel (GEMM1, fused scale in epilogue)
        - SiLU+mul+quant (1 v2 fused kernel or 3 separate ops)
        - invoke_fused_moe_int8_kernel (GEMM2, fused routing weight)
        - moe_sum_reduce (combine top-k)

        Falls back to deep_gemm on any error (caught by caller).
        """
        from rtp_llm.models_py.kernels.cuda.sglang_triton_moe import (
            fused_moe_int8_forward,
        )

        output = fused_moe_int8_forward(
            hidden_states=hidden_states,
            w1=self.w1,
            w2=self.w2,
            w1_scale=self._w1_scale_fp32,
            w2_scale=self._w2_scale_fp32,
            topk_weights=topk_weights,
            topk_ids=topk_ids.to(torch.int32),
            inter_size=self.inter_size,
            top_k=self.top_k,
        )
        return CombineForwardPayload(fused_expert_output=output.to(hidden_states.dtype))

    def _execute_deep_gemm(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> CombineForwardPayload:
        """Use deep_gemm INT8 grouped GEMM for hardware INT8 acceleration.

        Converts flat 2D activations [M, K] into masked grouped layout
        [E, expected_m, K], runs two grouped GEMMs (gate+up then down),
        and scatters results back to the original token order.

        Falls back to per-expert loop if deep_gemm call fails.
        """
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                m_grouped_int8_gemm_nt_masked,
            )
        except Exception as e:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "deep_gemm import failed during CUDA Graph capture "
                    f"and fallback is not graph-safe: {e}"
                ) from e
            return self._execute_fallback(hidden_states, topk_ids, topk_weights)

        # The masked layout expands every token by top-k for every local expert.
        # Bound that workspace during prefill; decode Graph shapes stay static.
        if (
            _PREFILL_CHUNK_SIZE > 0
            and not torch.cuda.is_current_stream_capturing()
            and hidden_states.shape[0] > _PREFILL_CHUNK_SIZE
        ):
            output = torch.empty_like(hidden_states)
            for start in range(0, hidden_states.shape[0], _PREFILL_CHUNK_SIZE):
                end = min(start + _PREFILL_CHUNK_SIZE, hidden_states.shape[0])
                chunk_payload = self._execute_deep_gemm_impl(
                    hidden_states[start:end],
                    topk_ids[start:end],
                    topk_weights[start:end],
                    m_grouped_int8_gemm_nt_masked,
                )
                output[start:end].copy_(chunk_payload.fused_expert_output)
            return CombineForwardPayload(fused_expert_output=output)

        try:
            return self._execute_deep_gemm_impl(
                hidden_states,
                topk_ids,
                topk_weights,
                m_grouped_int8_gemm_nt_masked,
            )
        except Exception as e:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"deep_gemm INT8 grouped GEMM failed during CUDA Graph capture "
                    f"and fallback is not graph-safe: {e} | "
                    f"M={hidden_states.shape[0]}, top_k={topk_ids.shape[1]}, E={self.E}"
                ) from e
            logger.warning(
                f"deep_gemm INT8 grouped GEMM failed, using fallback: {e} | "
                f"M={hidden_states.shape[0]}, top_k={topk_ids.shape[1]}, E={self.E}"
            )
            return self._execute_fallback(hidden_states, topk_ids, topk_weights)

    def _execute_deep_gemm_impl(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        grouped_gemm_fn,
    ) -> CombineForwardPayload:
        """Core deep_gemm INT8 grouped GEMM implementation.

        Token arrangement for masked layout:
        - topk_ids [M, top_k] is flattened → [M*top_k]
        - Sorted by expert id → tokens grouped by expert
        - Each expert's slot padded to expected_m
        - GEMM runs on [E, expected_m, K] grouped layout
        - Results scattered back to [M, K] via inverse permutation
        """
        _is_capturing = torch.cuda.is_current_stream_capturing()

        M, K = hidden_states.shape
        top_k = topk_ids.size(1)
        device = hidden_states.device
        E = self.E
        N = self.N  # 2 * inter_size
        inter_size = self.inter_size

        # Flatten top-k dimension: each token appears top_k times
        flat_topk_ids = topk_ids.reshape(-1)  # [M*top_k]
        flat_topk_weights = topk_weights.reshape(-1)  # [M*top_k]
        total_expanded = M * top_k

        # Sort by expert id to group tokens by expert
        sorted_expert_ids, perm = flat_topk_ids.sort(stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(total_expanded, device=device, dtype=perm.dtype)

        # Number of tokens per expert (from expanded top-k view)
        # torch.bincount() is NOT graph-safe (it does internal sync to determine
        # output size on some backends). Use scatter_add_ instead, which is a
        # pure CUDA operation compatible with CUDA Graph capture.
        clamped_ids = flat_topk_ids.clamp(min=0, max=E - 1).long()
        expert_counts = torch.zeros(E, dtype=torch.long, device=device)
        expert_counts.scatter_add_(0, clamped_ids, torch.ones_like(clamped_ids))
        # Use actual max expert count to guarantee no overflow (requires 1 GPU→CPU sync).
        # This is correct for all token distributions, including heavily skewed ones
        # where one expert gets >>4x the average load.
        # During CUDA Graph capture, .item() (GPU→CPU sync) is not allowed.
        # Use the safe upper bound M*top_k (max tokens any single expert can receive)
        # which is small for decode-sized batches (e.g. M=1, top_k=6 → 6 slots/expert).
        if _is_capturing:
            expected_m = M * top_k
        else:
            expected_m = min(M * top_k, max(int(expert_counts.max().item()), top_k))

        # Quantize activation to INT8 per-token: [M, K] → int8, [M, 1] FP32
        a_int8, a_scale = _per_token_quant_int8(hidden_states)

        # Expand activations to [M*top_k, K] by gathering (each token → top_k copies)
        expanded_int8 = a_int8[perm // top_k]  # [M*top_k, K]
        expanded_scale = a_scale[perm // top_k]  # [M*top_k, 1]

        # --- Arrange into masked grouped layout [E, expected_m, ...] ---
        # Vectorized token arrangement:
        # - Each expert e occupies flat positions [e*expected_m, e*expected_m + count_e)
        # - Padding positions (count_e .. expected_m) stay zero
        # - This matches inv_perm which maps to e*expected_m + j
        cum_offsets = torch.zeros(E, dtype=torch.long, device=device)
        cum_offsets[1:] = expert_counts[:-1].cumsum(0)

        sorted_indices = torch.arange(total_expanded, device=device)
        slot_in_sorted = sorted_indices - cum_offsets[sorted_expert_ids.long()]
        # Write position: expert_start + slot_within_expert
        write_pos = sorted_expert_ids.long() * expected_m + slot_in_sorted

        # Place data at correct strided positions
        grouped_a_flat = torch.zeros(E * expected_m, K, device=device, dtype=torch.int8)
        grouped_a_s_flat = torch.zeros(
            E * expected_m, 1, device=device, dtype=torch.float32
        )
        grouped_a_flat[write_pos] = expanded_int8
        grouped_a_s_flat[write_pos] = expanded_scale

        grouped_a = grouped_a_flat.reshape(E, expected_m, K)
        grouped_a_s = grouped_a_s_flat.reshape(E, expected_m, 1)

        # Use pre-converted weight scales (done in __init__)
        w1_scale_fp32 = self._w1_scale_fp32
        w2_scale_fp32 = self._w2_scale_fp32

        # === GEMM1: grouped_a @ w1.T → [E, expected_m, N=2*inter] ===
        # a: (int8[E, em, K], scale[E, em, 1])
        # b: (int8[E, N, K], scale[E, N, 1])
        output1 = torch.zeros(E, expected_m, N, device=device, dtype=torch.bfloat16)
        grouped_gemm_fn(
            (grouped_a, grouped_a_s),
            (self.w1, w1_scale_fp32),
            output1,
            expert_counts.to(torch.int32),
            expected_m,
        )

        # SiLU-and-mul + INT8 re-quantization: fused into 1 kernel (v2) or 3 separate ops
        # The v2 kernel only supports group_size in {16, 32, 64, 128}.
        # GLM-4.7 has inter_size=12288 which is not supported, so use separate path.
        _V2_SUPPORTED_GROUP_SIZES = (16, 32, 64, 128)
        if _HAS_FUSED_QUANT_V2 and inter_size in _V2_SUPPORTED_GROUP_SIZES:
            # Fused: swap halves + v2 kernel (silu+mul+quant in 1 launch)
            b_int8, b_scale = _fused_silu_mul_quant_int8(
                output1, inter_size, masked_m=expert_counts.to(torch.int32)
            )
            # b_int8: [E, expected_m, inter], b_scale: [E, expected_m, 1]
            del output1
        else:
            # Separate: silu + mul + quant (3 kernel launches)
            # Used when inter_size is not a supported v2 group_size (e.g., GLM-4.7 inter_size=12288)
            # GLM4 convention: w1 stores [up_proj, gate_proj] (up first, gate second).
            # SiLU+mul = silu(gate) * up = silu(second_half) * first_half
            up = output1[:, :, :inter_size]  # first half = up_proj
            gate = output1[:, :, inter_size:]  # second half = gate_proj
            intermediate = torch.nn.functional.silu(gate) * up
            del output1, gate, up
            b_int8, b_scale = _per_token_quant_int8(
                intermediate.reshape(E * expected_m, inter_size)
            )
            b_int8 = b_int8.reshape(E, expected_m, inter_size)
            b_scale = b_scale.reshape(E, expected_m, 1)
        output2 = torch.zeros(E, expected_m, K, device=device, dtype=torch.bfloat16)
        grouped_gemm_fn(
            (b_int8, b_scale),
            (self.w2, w2_scale_fp32),
            output2,
            expert_counts.to(torch.int32),
            expected_m,
        )

        # --- Scatter back to original [M*top_k, K] order ---
        # inv_perm maps original_pos → sorted_pos, but we need flat grouped positions
        # (e * expected_m + slot) to index into output2.
        # read_pos[orig_pos] = write_pos[sorted_pos] = write_pos[inv_perm[orig_pos]]
        read_pos = write_pos[inv_perm]  # [M*top_k] → flat grouped positions
        output2_flat = output2.reshape(-1, K)  # [E * expected_m, K]
        output2_expanded = output2_flat[read_pos]  # [M*top_k, K]

        # Combine top-k: reshape to [M, top_k, K], weight and sum
        output2_per_token = output2_expanded.reshape(M, top_k, K)
        weights_expanded = (
            flat_topk_weights.reshape(M, top_k)
            .unsqueeze(-1)
            .to(output2_per_token.dtype)
        )  # [M, top_k, 1]
        output = (output2_per_token * weights_expanded).sum(dim=1)  # [M, K]

        return CombineForwardPayload(fused_expert_output=output.to(hidden_states.dtype))

    def _execute_fallback(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> CombineForwardPayload:
        """Fallback: per-expert dequantization + torch.mm.

        Works with both INT8 + scale (dequantize on-the-fly) and BF16 weights.
        """
        M, K = hidden_states.shape
        top_k = topk_ids.size(1)
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Prepare scales if INT8
        has_scale = self.has_int8
        if has_scale:
            w1_scale = self.w1_scale.to(dtype)
            if w1_scale.dim() < self.w1.dim():
                w1_scale = w1_scale.unsqueeze(-1)
            w2_scale = self.w2_scale.to(dtype)
            if w2_scale.dim() < self.w2.dim():
                w2_scale = w2_scale.unsqueeze(-1)

        # Pure PyTorch grouped GEMM: per-expert loop
        output = torch.zeros(M, K, device=device, dtype=dtype)

        for expert_idx in range(self.E):
            mask = topk_ids == expert_idx  # [M, top_k]
            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]

            if token_indices.numel() == 0:
                continue

            # Per-expert dequantization
            if has_scale:
                w1_e = self.w1[expert_idx].to(dtype) * w1_scale[expert_idx]
                w2_e = self.w2[expert_idx].to(dtype) * w2_scale[expert_idx]
            else:
                w1_e = self.w1[expert_idx].to(dtype)
                w2_e = self.w2[expert_idx].to(dtype)

            expert_input = hidden_states[token_indices]  # [n_tokens, K]

            # GEMM1: [n_tokens, K] @ [2*inter, K].T → [n_tokens, 2*inter]
            intermediate = torch.mm(expert_input, w1_e.t())

            # SiLU-and-mul: first half = up, second half = gate
            up = intermediate[:, : self.inter_size]
            gate = intermediate[:, self.inter_size :]
            intermediate_act = torch.nn.functional.silu(gate) * up
            del intermediate, gate, up

            # GEMM2: [n_tokens, inter] @ [hidden, inter].T → [n_tokens, K]
            expert_output = torch.mm(intermediate_act, w2_e.t())
            del intermediate_act, w1_e, w2_e

            # Apply routing weights and accumulate
            expert_weights = mask[token_indices].to(dtype)
            token_topk_weights = topk_weights[token_indices]
            combined_weight = (expert_weights * token_topk_weights).sum(
                dim=1, keepdim=True
            )
            output[token_indices] += expert_output * combined_weight

        return CombineForwardPayload(fused_expert_output=output)


class Int8PerChannelEpNormalExecutor(Int8PerChannelDequantExecutor):
    """INT8 executor for the rank-deduplicated DeepEP normal payload."""

    @staticmethod
    def _select_local_routes(
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        start_expert_id: int,
        num_local_experts: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        local_ids = topk_ids - start_expert_id
        local_mask = (local_ids >= 0) & (local_ids < num_local_experts)
        token_ids = (
            torch.arange(topk_ids.size(0), device=topk_ids.device)
            .unsqueeze(1)
            .expand_as(topk_ids)
        )
        return (
            token_ids[local_mask],
            local_ids[local_mask],
            topk_weights[local_mask],
        )

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        hidden_states = payload.expert_x
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        assert topk_ids is not None
        assert topk_weights is not None

        if (
            _PREFILL_CHUNK_SIZE > 0
            and not torch.cuda.is_current_stream_capturing()
            and hidden_states.shape[0] > _PREFILL_CHUNK_SIZE
        ):
            output = torch.empty_like(hidden_states)
            for start in range(0, hidden_states.shape[0], _PREFILL_CHUNK_SIZE):
                end = min(start + _PREFILL_CHUNK_SIZE, hidden_states.shape[0])
                chunk_output = self._execute_rows(
                    hidden_states[start:end],
                    topk_ids[start:end],
                    topk_weights[start:end],
                )
                output[start:end].copy_(chunk_output.fused_expert_output)
            return CombineForwardPayload(fused_expert_output=output)

        return self._execute_rows(hidden_states, topk_ids, topk_weights)

    def _execute_rows(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> CombineForwardPayload:
        """Execute a bounded set of received rows for DeepEP normal prefill."""

        token_ids, local_expert_ids, route_weights = self._select_local_routes(
            topk_ids,
            topk_weights,
            self.start_expert_id,
            self.num_experts_per_partition,
        )
        if token_ids.numel() == 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros_like(hidden_states)
            )

        if self._use_deep_gemm and self.has_int8:
            try:
                from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                    m_grouped_int8_gemm_nt_masked,
                )

                return self._execute_local_routes_deep_gemm(
                    hidden_states,
                    token_ids,
                    local_expert_ids,
                    route_weights,
                    m_grouped_int8_gemm_nt_masked,
                )
            except Exception as e:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "DeepEP normal INT8 grouped GEMM failed during CUDA Graph "
                        f"capture and fallback is not graph-safe: {e} | "
                        f"M={hidden_states.shape[0]}, "
                        f"local_routes={token_ids.numel()}, E={self.E}"
                    ) from e
                if (
                    isinstance(e, torch.OutOfMemoryError)
                    or "out of memory" in str(e).lower()
                ):
                    raise RuntimeError(
                        "DeepEP normal INT8 grouped GEMM out of memory; "
                        "refusing the higher-memory fallback path: "
                        f"{e} | M={hidden_states.shape[0]}, "
                        f"local_routes={token_ids.numel()}, E={self.E}"
                    ) from e
                logger.warning(
                    "DeepEP normal INT8 grouped GEMM failed, using fallback: %s | "
                    "M=%d, local_routes=%d, E=%d",
                    e,
                    hidden_states.shape[0],
                    token_ids.numel(),
                    self.E,
                )
        else:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DeepEP normal INT8 MoE fallback path is not graph-safe (contains "
                    f"nonzero + per-expert Python loop) and cannot be used during CUDA "
                    f"Graph capture (deep_gemm unavailable) | "
                    f"M={hidden_states.shape[0]}, E={self.E}"
                )

        local_ids = (topk_ids - self.start_expert_id).clamp(
            min=0, max=self.num_experts_per_partition - 1
        )
        local_mask = (topk_ids >= self.start_expert_id) & (
            topk_ids <= self.end_expert_id
        )
        return self._execute_fallback(
            hidden_states,
            local_ids,
            topk_weights * local_mask,
        )

    def _execute_local_routes_deep_gemm(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        local_expert_ids: torch.Tensor,
        route_weights: torch.Tensor,
        grouped_gemm_fn: Any,
    ) -> CombineForwardPayload:
        """Compute only valid rank-local routes, then reduce them by token."""
        _is_capturing = torch.cuda.is_current_stream_capturing()

        M, K = hidden_states.shape
        device = hidden_states.device
        E = self.E
        N = self.N
        inter_size = self.inter_size

        sorted_expert_ids, perm = local_expert_ids.sort(stable=True)
        sorted_token_ids = token_ids[perm]
        sorted_route_weights = route_weights[perm]
        total_routes = sorted_expert_ids.numel()

        expert_counts = torch.zeros(E, dtype=torch.long, device=device)
        expert_counts.scatter_add_(
            0,
            sorted_expert_ids.long(),
            torch.ones_like(sorted_expert_ids, dtype=torch.long),
        )
        # During CUDA Graph capture, .item() (GPU→CPU sync) is not allowed.
        # Use the safe upper bound total_routes (max routes any single expert
        # can receive) which avoids CPU sync.
        if _is_capturing:
            expected_m = total_routes
        else:
            expected_m = max(int(expert_counts.max().item()), 1)

        a_int8, a_scale = _per_token_quant_int8(hidden_states)
        expanded_int8 = a_int8[sorted_token_ids]
        expanded_scale = a_scale[sorted_token_ids]

        cum_offsets = torch.zeros(E, dtype=torch.long, device=device)
        cum_offsets[1:] = expert_counts[:-1].cumsum(0)
        sorted_indices = torch.arange(total_routes, device=device)
        slot_in_expert = sorted_indices - cum_offsets[sorted_expert_ids.long()]
        write_pos = sorted_expert_ids.long() * expected_m + slot_in_expert

        grouped_a_flat = torch.zeros(E * expected_m, K, device=device, dtype=torch.int8)
        grouped_a_scale_flat = torch.zeros(
            E * expected_m, 1, device=device, dtype=torch.float32
        )
        grouped_a_flat[write_pos] = expanded_int8
        grouped_a_scale_flat[write_pos] = expanded_scale
        grouped_a = grouped_a_flat.reshape(E, expected_m, K)
        grouped_a_scale = grouped_a_scale_flat.reshape(E, expected_m, 1)
        masked_m = expert_counts.to(torch.int32)

        output1 = torch.zeros(E, expected_m, N, device=device, dtype=torch.bfloat16)
        grouped_gemm_fn(
            (grouped_a, grouped_a_scale),
            (self.w1, self._w1_scale_fp32),
            output1,
            masked_m,
            expected_m,
        )

        up = output1[:, :, :inter_size]
        gate = output1[:, :, inter_size:]
        intermediate = torch.nn.functional.silu(gate) * up
        b_int8, b_scale = _per_token_quant_int8(
            intermediate.reshape(E * expected_m, inter_size)
        )
        b_int8 = b_int8.reshape(E, expected_m, inter_size)
        b_scale = b_scale.reshape(E, expected_m, 1)

        output2 = torch.zeros(E, expected_m, K, device=device, dtype=torch.bfloat16)
        grouped_gemm_fn(
            (b_int8, b_scale),
            (self.w2, self._w2_scale_fp32),
            output2,
            masked_m,
            expected_m,
        )

        route_output = output2.reshape(E * expected_m, K)[write_pos]
        route_output = route_output * sorted_route_weights.to(
            route_output.dtype
        ).unsqueeze(-1)
        output = torch.zeros(M, K, device=device, dtype=route_output.dtype)
        output.index_add_(0, sorted_token_ids, route_output)
        return CombineForwardPayload(fused_expert_output=output.to(hidden_states.dtype))


class Int8PerChannelMaskedExecutor(FusedMoeExpertExecutor):
    """INT8 per-channel executor for masked/batched layout (Low Latency).

    Input: [E, M, K] BF16 activations (pre-arranged by expert from DeepEP LL dispatch)
    Output: [E, M, K] BF16 expert outputs (no routing weight application)

    Flow:
    1. Quantize BF16 activation to INT8 per-token
    2. INT8 grouped GEMM (gate+up) via m_grouped_int8_gemm_nt_masked
    3. SiLU-and-mul activation
    4. Re-quantize to INT8 per-token
    5. INT8 grouped GEMM (down) via m_grouped_int8_gemm_nt_masked

    Routing weights are NOT applied here; the Low Latency Router's
    low_latency_combine handles that in finalize().
    """

    @classmethod
    def executor_type(cls):
        return ExecutorType.FUSED_MOE

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "INT8_PER_CHANNEL_COMPRESSED")

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        from rtp_llm.utils.model_weight import W

        self.ep_size = config.ep_size
        self.num_experts = config.expert_num
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size

        # Store weights and scales
        self.w1 = weights[W.moe_w1]  # [E_local, 2*inter, hidden] INT8 or BF16
        self.w2 = weights[W.moe_w2]  # [E_local, hidden, inter] INT8 or BF16
        self.w1_scale = weights.get(W.moe_s1)  # [E_local, 2*inter] FP32 or None
        self.w2_scale = weights.get(W.moe_s2)  # [E_local, hidden] FP32 or None

        self.E = self.w1.size(0)
        self.N = self.w1.size(1)  # 2 * inter_size
        self.K = self.w1.size(2)  # hidden_size
        self.inter_size = self.N // 2

        self.has_int8 = self.w1.dtype == torch.int8 and self.w1_scale is not None

        # Try to use deep_gemm INT8 grouped GEMM
        self._use_deep_gemm = False
        if self.has_int8:
            try:
                from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                    has_deep_gemm,
                    m_grouped_int8_gemm_nt_masked,
                )

                if has_deep_gemm():
                    self._use_deep_gemm = True
                    logger.info(
                        "INT8 Masked executor: using deep_gemm INT8 grouped GEMM"
                    )
            except Exception as e:
                logger.warning(
                    f"deep_gemm INT8 masked not available, using fallback: {e}"
                )

        if config.enable_cuda_graph and not self.has_int8:
            raise RuntimeError(
                "INT8 EP low-latency with CUDA Graph requires INT8 MoE weights "
                "and per-channel scales, but the loader produced "
                f"w1.dtype={self.w1.dtype}, w1_scale={self.w1_scale is not None}. "
                "Ensure CompressedInt8PerChannelWeight conversion is enabled."
            )
        if config.enable_cuda_graph and not self._use_deep_gemm:
            raise RuntimeError(
                "INT8 EP low-latency with CUDA Graph requires the DeepGEMM INT8 "
                "masked kernel; the Python fallback is not graph-safe."
            )

        # Pre-convert weight scales to FP32 and reshape to [E, N, 1]
        if self.w1_scale is not None:
            self._w1_scale_fp32 = self.w1_scale.to(torch.float32)
            if self._w1_scale_fp32.dim() == 2:
                self._w1_scale_fp32 = self._w1_scale_fp32.unsqueeze(-1)
        else:
            self._w1_scale_fp32 = None
        if self.w2_scale is not None:
            self._w2_scale_fp32 = self.w2_scale.to(torch.float32)
            if self._w2_scale_fp32.dim() == 2:
                self._w2_scale_fp32 = self._w2_scale_fp32.unsqueeze(-1)
        else:
            self._w2_scale_fp32 = None

        logger.info(
            f"INT8 per-channel Masked executor: "
            f"E={self.E}, N={self.N}, K={self.K}, inter_size={self.inter_size}, "
            f"has_int8={self.has_int8}, use_deep_gemm={self._use_deep_gemm}"
        )

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        """MoE forward with INT8 weights on masked/batched 3D layout.

        Input expert_x shape: [E, M, K] (pre-arranged by DeepEP Low Latency)
        Output: [E, M, K] BF16 (no routing weight applied)
        """
        assert (
            payload.expert_tokens_meta is not None
            and payload.expert_tokens_meta.expert_num_tokens is not None
        )
        expert_x = payload.expert_x  # [E, M, K]
        E, M, K = expert_x.size()
        assert E == self.E and K == self.K
        masked_m = payload.expert_tokens_meta.expert_num_tokens
        expected_m = (
            min(M, payload.expert_tokens_meta.expected_m)
            if payload.expert_tokens_meta.expected_m is not None
            else M
        )

        _is_capturing = torch.cuda.is_current_stream_capturing()

        if self._use_deep_gemm and self.has_int8:
            try:
                return self._execute_deep_gemm_masked(
                    expert_x, masked_m, expected_m, payload.expert_x_scale
                )
            except Exception as e:
                if _is_capturing:
                    raise RuntimeError(
                        "deep_gemm INT8 masked MoE failed during CUDA Graph capture "
                        f"and fallback is not graph-safe: {e} | "
                        f"E={self.E}, M={M}, K={K}"
                    ) from e
                logger.warning(
                    f"deep_gemm INT8 masked MoE failed, using fallback: {e} | "
                    f"E={self.E}, M={M}, K={K}"
                )
                return self._execute_fallback_masked(expert_x, masked_m, expected_m)
        else:
            if _is_capturing:
                raise RuntimeError(
                    "INT8 masked MoE fallback path is not graph-safe (contains "
                    f".item() + per-expert Python loop) and cannot be used during "
                    f"CUDA Graph capture (deep_gemm unavailable) | E={self.E}, M={M}, K={K}"
                )
            return self._execute_fallback_masked(expert_x, masked_m, expected_m)

    def _execute_deep_gemm_masked(
        self,
        expert_x: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
        pre_quant_scale: Optional[torch.Tensor] = None,
    ) -> CombineForwardPayload:
        """Use deep_gemm INT8 grouped GEMM on 3D masked layout.

        Args:
            expert_x: [E, M, K] BF16 activations
            masked_m: [E] number of real tokens per expert
            expected_m: padded M dimension
        """
        try:
            from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
                m_grouped_int8_gemm_nt_masked,
            )
        except Exception as e:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "deep_gemm import failed during CUDA Graph capture "
                    f"and fallback is not graph-safe: {e}"
                ) from e
            return self._execute_fallback_masked(expert_x, masked_m, expected_m)

        E, M, K = expert_x.shape
        device = expert_x.device

        # Use pre-converted weight scales (done in __init__)
        w1_scale_fp32 = self._w1_scale_fp32
        w2_scale_fp32 = self._w2_scale_fp32

        # M2: if DeepEP LL already dispatched int8 + per-token scale, consume it
        # directly and skip the redundant re-quant (comm already halved upstream).
        if expert_x.dtype == torch.int8 and pre_quant_scale is not None:
            grouped_a = expert_x  # [E, M, K] int8
            grouped_a_s = pre_quant_scale
            if grouped_a_s.dim() == 2:
                grouped_a_s = grouped_a_s.unsqueeze(-1)
            grouped_a_s = grouped_a_s.to(torch.float32).reshape(E, M, 1)
        else:
            # Quantize BF16 activation to INT8 per-token
            # Reshape [E, M, K] -> [E*M, K], quantize, reshape back
            flat_x = expert_x.reshape(E * M, K)
            a_int8, a_scale = _per_token_quant_int8(flat_x)
            grouped_a = a_int8.reshape(E, M, K)
            grouped_a_s = a_scale.reshape(E, M, 1)

        # === GEMM1: grouped_a @ w1.T -> [E, M, N=2*inter] ===
        output1 = torch.zeros(E, M, self.N, device=device, dtype=torch.bfloat16)
        m_grouped_int8_gemm_nt_masked(
            (grouped_a, grouped_a_s),
            (self.w1, w1_scale_fp32),
            output1,
            masked_m.to(torch.int32),
            expected_m,
        )

        # SiLU-and-mul + INT8 re-quantization: fused into 1 kernel (v2) or 3 separate ops
        # The v2 kernel only supports group_size in {16, 32, 64, 128}.
        # GLM-4.7 has inter_size=12288 which is not supported, so use separate path.
        _V2_SUPPORTED_GROUP_SIZES_MASKED = (16, 32, 64, 128)
        if _HAS_FUSED_QUANT_V2 and self.inter_size in _V2_SUPPORTED_GROUP_SIZES_MASKED:
            # Fused: swap halves + v2 kernel (silu+mul+quant in 1 launch)
            b_int8, b_scale = _fused_silu_mul_quant_int8(
                output1, self.inter_size, masked_m=masked_m.to(torch.int32)
            )
            # b_int8: [E, M, inter], b_scale: [E, M, 1]
            del output1
        else:
            # Separate: silu + mul + quant (3 kernel launches)
            # GLM4 convention: first half = up_proj, second half = gate_proj
            up = output1[:, :, : self.inter_size]
            gate = output1[:, :, self.inter_size :]
            intermediate = torch.nn.functional.silu(gate) * up
            del output1, gate, up
            flat_inter = intermediate.reshape(E * M, self.inter_size)
            b_int8, b_scale = _per_token_quant_int8(flat_inter)
            b_int8 = b_int8.reshape(E, M, self.inter_size)
            b_scale = b_scale.reshape(E, M, 1)
            del intermediate, flat_inter

        output2 = torch.zeros(E, M, K, device=device, dtype=torch.bfloat16)
        m_grouped_int8_gemm_nt_masked(
            (b_int8, b_scale),
            (self.w2, w2_scale_fp32),
            output2,
            masked_m.to(torch.int32),
            expected_m,
        )
        del b_int8, b_scale

        return CombineForwardPayload(fused_expert_output=output2)

    def _execute_fallback_masked(
        self,
        expert_x: torch.Tensor,
        masked_m: torch.Tensor,
        expected_m: int,
    ) -> CombineForwardPayload:
        """Fallback: per-expert dequantization + torch.mm on 3D masked layout.

        Args:
            expert_x: [E, M, K] BF16 activations
            masked_m: [E] number of real tokens per expert
            expected_m: padded M dimension
        Returns:
            CombineForwardPayload with [E, M, K] output
        """
        E, M, K = expert_x.shape
        device = expert_x.device
        dtype = expert_x.dtype

        has_scale = self.has_int8
        if has_scale:
            w1_scale = self.w1_scale.to(dtype)
            if w1_scale.dim() < self.w1.dim():
                w1_scale = w1_scale.unsqueeze(-1)
            w2_scale = self.w2_scale.to(dtype)
            if w2_scale.dim() < self.w2.dim():
                w2_scale = w2_scale.unsqueeze(-1)

        output = torch.zeros(E, M, K, device=device, dtype=dtype)

        for expert_idx in range(E):
            n_tokens = (
                int(masked_m[expert_idx].item())
                if torch.is_tensor(masked_m[expert_idx])
                else int(masked_m[expert_idx])
            )
            if n_tokens == 0:
                continue

            expert_input = expert_x[expert_idx, :n_tokens, :]  # [n_tokens, K]

            # Per-expert dequantization
            if has_scale:
                w1_e = self.w1[expert_idx].to(dtype) * w1_scale[expert_idx]
                w2_e = self.w2[expert_idx].to(dtype) * w2_scale[expert_idx]
            else:
                w1_e = self.w1[expert_idx].to(dtype)
                w2_e = self.w2[expert_idx].to(dtype)

            # GEMM1: [n_tokens, K] @ [2*inter, K].T -> [n_tokens, 2*inter]
            intermediate = torch.mm(expert_input, w1_e.t())

            # SiLU-and-mul
            up = intermediate[:, : self.inter_size]
            gate = intermediate[:, self.inter_size :]
            intermediate_act = torch.nn.functional.silu(gate) * up
            del intermediate, gate, up

            # GEMM2: [n_tokens, inter] @ [hidden, inter].T -> [n_tokens, K]
            expert_output = torch.mm(intermediate_act, w2_e.t())
            del intermediate_act, w1_e, w2_e

            output[expert_idx, :n_tokens, :] = expert_output

        return CombineForwardPayload(fused_expert_output=output)


class CudaInt8PerChannelCppStrategy(MoeStrategy):
    """CUDA INT8 per-channel dequantization MoE strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "INT8_PER_CHANNEL_COMPRESSED")
        checker.check(
            config.moe_strategy == "int8_per_channel_cpp"
            or config.moe_strategy == "auto"
        )
        # Warn about known PureTP + CUDA Graph instability: all_reduce in
        # pure_tp_router does not support graph replay, causing crashes on
        # long sequences (e.g., 200k). This is a warning only, not a hard
        # check, to allow users who know the risks to proceed.
        if (
            quant_method == "INT8_PER_CHANNEL_COMPRESSED"
            and config.tp_size > 1
            and config.ep_size == 1
            and getattr(config, "enable_cuda_graph", False)
        ):
            logger.warning(
                "PureTP INT8 MoE with CUDA Graph may crash on long sequences "
                "due to all_reduce not supporting graph replay. "
                "Consider using CP mode or disabling CUDA Graph."
            )

    def get_attributes(self) -> StrategyAttributes:
        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=PureTpRouterInt8PerChannel,
            executor_class=Int8PerChannelDequantExecutor,
            quant_config=quant_config,
        )


class CudaInt8PerChannelPureCPStrategy(MoeStrategy):
    """Compressed INT8 W8A8 strategy for pure CP+EP topology."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(
            resolver.get_quant_method(config) == "INT8_PER_CHANNEL_COMPRESSED"
        )
        checker.check(config.moe_strategy == "int8_per_channel_pure_cp")
        checker.check(config.dp_size == 1)
        checker.check(resolver.is_cp_equal_ep(config))
        checker.check(config.ep_size > 1)
        checker.check(config.parallelism_config.prefill_cp_config.is_enabled())

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router import (
            PureCpRouterInt8PerChannel,
        )

        return StrategyAttributes(
            router_class=PureCpRouterInt8PerChannel,
            executor_class=Int8PerChannelDequantExecutor,
            quant_config=FusedMoEQuantConfig(quant_dtype=None),
        )


class CudaInt8PerChannelEpNormalStrategy(MoeStrategy):
    """INT8 per-channel EP Normal mode strategy.

    Uses DeepEP Normal Router (All-to-All dispatch/combine) with an executor
    that expands only rank-local expert routes from the deduplicated payload.

    Priority: DEEPEP_NORMAL(2)*10 + FUSED_MOE(2) = 22
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "INT8_PER_CHANNEL_COMPRESSED")
        checker.check(
            config.moe_strategy == "int8_per_channel_ep_normal"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterInt8PerChannel,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=DeepepNormalRouterInt8PerChannel,
            executor_class=Int8PerChannelEpNormalExecutor,
            quant_config=quant_config,
        )


class CudaInt8PerChannelEpLowLatencyStrategy(MoeStrategy):
    """INT8 per-channel EP Low Latency mode strategy.

    Uses DeepEP Low Latency Router with the new Int8PerChannelMaskedExecutor.
    The masked executor handles [E, M, K] 3D input layout from LL dispatch.

    Priority: DEEPEP_LOW_LATENCY(4)*10 + FUSED_MOE(2) = 42
    """

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "INT8_PER_CHANNEL_COMPRESSED")
        checker.check(
            config.moe_strategy == "int8_per_channel_ep_low_latency"
            or config.moe_strategy == "auto"
        )
        if quant_method == "INT8_PER_CHANNEL_COMPRESSED" and getattr(
            config, "enable_cuda_graph", False
        ):
            logger.info(
                "INT8 EP Low Latency MoE with CUDA Graph enabled. "
                "DeepEP LL router uses static buffers and is graph-safe. "
                "TP=1 avoids all_reduce/all_gather collective ops."
            )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouterInt8PerChannel,
        )

        quant_config = FusedMoEQuantConfig(quant_dtype=None)
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouterInt8PerChannel,
            executor_class=Int8PerChannelMaskedExecutor,
            quant_config=quant_config,
        )
