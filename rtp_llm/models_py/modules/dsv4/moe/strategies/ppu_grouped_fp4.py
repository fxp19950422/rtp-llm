"""M890P grouped-MXFP4 routed experts for the DSV4 TP4/EP1 topology.

This strategy is deliberately separate from the CUDA ``grouped_fp4`` backend.
It consumes the checkpoint's packed MXFP4 tensors, executes two PPU masked
grouped GEMMs, and returns the normalized routed partial expected by ``MoE``'s
post-W2 route-scale/TP-reduce contract.

There is no fallback in this module.  A wrong topology, storage geometry,
device, external symbol, or fixed graph capacity fails closed.
"""

from __future__ import annotations

import math
import os
from typing import Dict, Sequence, Tuple

import torch

from .base import MoeCfg, RoutedExpertsStrategy, register_strategy


_ROUTE_WEIGHT_CONTRACT = "post_w2_normalized_then_scale_v1"
_CAPACITY_ENV = "DSV4_PPU_GROUPED_FP4_CAPACITY"
_GROUPED_M_ALIGNMENT = 128


def _supports_topology(cfg: MoeCfg) -> bool:
    """The provider seam is intentionally limited to the TP4/EP1 pivot."""

    return int(cfg.tp_size) == 4 and int(cfg.ep_size) == 1


def _runtime_eligible() -> bool:
    """Probe the PPU-only leaf without making generic GPU TP4 select it."""

    if not torch.cuda.is_available():
        return False
    try:
        current_device = torch.cuda.current_device()
        if torch.cuda.get_device_name(current_device) != "ZW-M890P":
            return False
        import deep_gemm
    except (ImportError, RuntimeError):
        return False
    return callable(
        getattr(deep_gemm, "m_grouped_gemm_fp4_fp4_bf16_nt_masked", None)
    )


def _derive_inter_local_and_tp(
    cfg: MoeCfg,
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    w3_shape: Tuple[int, ...],
    s1_shape: Tuple[int, ...],
    s2_shape: Tuple[int, ...],
    s3_shape: Tuple[int, ...],
) -> Tuple[int, int]:
    """Validate packed MXFP4 geometry and derive the routed TP contract."""

    if not _supports_topology(cfg):
        raise RuntimeError(
            "ppu_grouped_fp4 requires exactly tp_size=4 and ep_size=1"
        )

    experts = int(cfg.n_local_experts)
    dim = int(cfg.dim)
    if experts != int(cfg.n_routed_experts):
        raise ValueError(
            "EP1 grouped MXFP4 must bind every routed expert on each TP rank"
        )
    if len(w1_shape) != 3:
        raise ValueError(f"packed MXFP4 w1 must be rank 3, got {w1_shape}")
    inter_local = int(w1_shape[1])
    if dim <= 0 or inter_local <= 0 or dim % 64 or inter_local % 64:
        raise ValueError(
            "PPU grouped MXFP4 requires positive dim/inter_local aligned to 64, "
            f"got dim={dim}, inter_local={inter_local}"
        )

    expected_w1 = (experts, inter_local, dim // 2)
    expected_w2 = (experts, dim, inter_local // 2)
    expected_s1 = (experts, inter_local, dim // 32)
    expected_s2 = (experts, dim, inter_local // 32)
    if w1_shape != expected_w1 or w3_shape != expected_w1:
        raise ValueError(
            "packed MXFP4 w1/w3 geometry mismatch: "
            f"got {w1_shape}/{w3_shape}, expected {expected_w1}"
        )
    if w2_shape != expected_w2:
        raise ValueError(
            f"packed MXFP4 w2 geometry mismatch: got {w2_shape}, expected {expected_w2}"
        )
    if s1_shape != expected_s1 or s3_shape != expected_s1:
        raise ValueError(
            "MXFP4 w1/w3 scale geometry mismatch: "
            f"got {s1_shape}/{s3_shape}, expected {expected_s1}"
        )
    if s2_shape != expected_s2:
        raise ValueError(
            f"MXFP4 w2 scale geometry mismatch: got {s2_shape}, expected {expected_s2}"
        )

    full_inter = int(cfg.moe_inter_dim)
    if inter_local == full_inter:
        routed_tp_size = 1
    elif inter_local * int(cfg.tp_size) == full_inter:
        routed_tp_size = int(cfg.tp_size)
    else:
        raise ValueError(
            "routed intermediate is neither full nor a pure TP preshard: "
            f"inter_local={inter_local}, moe_inter_dim={full_inter}, "
            f"tp_size={cfg.tp_size}"
        )
    return inter_local, routed_tp_size


def _select_capacity(
    configured_capacity: int,
    observed_counts: Sequence[int],
    n_experts: int,
    *,
    fixed_shape: bool,
    fixed_required: int = 0,
) -> int:
    """Select an aligned capacity without ever truncating an expert count."""

    configured_capacity = int(configured_capacity)
    n_experts = int(n_experts)
    if configured_capacity <= 0 or n_experts <= 0:
        raise ValueError("grouped MXFP4 capacity and expert count must be positive")
    alignment = _GROUPED_M_ALIGNMENT // math.gcd(_GROUPED_M_ALIGNMENT, n_experts)

    if fixed_shape:
        if configured_capacity % alignment:
            raise ValueError(
                "fixed grouped MXFP4 capacity must make E*capacity divisible by 128"
            )
        if fixed_required <= 0:
            raise RuntimeError("fixed grouped MXFP4 capacity requires a proven upper bound")
        if configured_capacity < int(fixed_required):
            raise RuntimeError(
                "fixed grouped MXFP4 capacity cannot cover the proven expert-count "
                f"upper bound: capacity={configured_capacity}, required={fixed_required}"
            )
        return configured_capacity

    if not observed_counts:
        raise ValueError("eager grouped MXFP4 requires observed per-expert counts")
    required = max(int(count) for count in observed_counts)
    if required < 0:
        raise ValueError("grouped MXFP4 expert counts must be non-negative")
    selected = max(configured_capacity, required)
    return ((selected + alignment - 1) // alignment) * alignment


@register_strategy
class PpuGroupedFP4Strategy(RoutedExpertsStrategy):
    """Strict M890P packed-MXFP4 strategy for DSV4 TP4/EP1."""

    name = "ppu_grouped_fp4"
    route_weight_contract = _ROUTE_WEIGHT_CONTRACT
    routed_tp_size = 1

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        self._configured_capacity = int(os.environ.get(_CAPACITY_ENV, "128"))
        if self._configured_capacity <= 0:
            raise ValueError(f"{_CAPACITY_ENV} must be positive")
        self.inter_local = 0

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        return _supports_topology(cfg) and _runtime_eligible()

    def setup_weights(self, layer_weights: Dict) -> None:
        """Bind packed routed tensors and prepare their PPU scale layout once."""

        if not _supports_topology(self.cfg):
            raise RuntimeError(
                "ppu_grouped_fp4 setup rejected a non-TP4/EP1 configuration"
            )

        from rtp_llm.utils.model_weight import W

        keys = (
            W.v4_routed_w1_w,
            W.v4_routed_w1_s,
            W.v4_routed_w2_w,
            W.v4_routed_w2_s,
            W.v4_routed_w3_w,
            W.v4_routed_w3_s,
        )
        try:
            w1, s1, w2, s2, w3, s3 = (layer_weights[key] for key in keys)
        except KeyError as error:
            raise KeyError(
                f"ppu_grouped_fp4 requires all six packed routed tensors; missing {error}"
            ) from error

        inter_local, routed_tp_size = _derive_inter_local_and_tp(
            self.cfg,
            tuple(w1.shape),
            tuple(w2.shape),
            tuple(w3.shape),
            tuple(s1.shape),
            tuple(s2.shape),
            tuple(s3.shape),
        )
        if any(weight.dtype not in (torch.int8, torch.uint8) for weight in (w1, w2, w3)):
            raise TypeError("ppu_grouped_fp4 requires packed int8/uint8 MXFP4 weights")
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None or any(scale.dtype != e8m0_dtype for scale in (s1, s2, s3)):
            raise TypeError(
                "ppu_grouped_fp4 requires float8_e8m0fnu checkpoint scales"
            )
        tensors = (w1, s1, w2, s2, w3, s3)
        if any(not tensor.is_cuda for tensor in tensors):
            raise ValueError("ppu_grouped_fp4 requires CUDA-compatible PPU tensors")
        if any(tensor.device != w1.device for tensor in tensors):
            raise ValueError("ppu_grouped_fp4 weights/scales must share one device")
        if any(not tensor.is_contiguous() for tensor in tensors):
            raise ValueError("ppu_grouped_fp4 weights/scales must be contiguous")
        if torch.cuda.get_device_name(w1.device) != "ZW-M890P":
            raise RuntimeError("ppu_grouped_fp4 requires ZW-M890P")

        from internal_source.rtp_llm.models_py.kernels.ppu_mxfp4 import (
            prepare_fp4_weight_scale_mxfp4,
        )

        w13 = torch.cat((w1, w3), dim=1).view(torch.uint8).contiguous()
        s13 = prepare_fp4_weight_scale_mxfp4(
            torch.cat((s1, s3), dim=1).contiguous()
        )
        w2_packed = w2.view(torch.uint8).contiguous()
        s2_prepared = prepare_fp4_weight_scale_mxfp4(s2)

        self.register_buffer("_ppu_w13", w13, persistent=False)
        self.register_buffer("_ppu_s13", s13, persistent=False)
        self.register_buffer("_ppu_w2", w2_packed, persistent=False)
        self.register_buffer("_ppu_s2", s2_prepared, persistent=False)
        self.inter_local = inter_local
        self.routed_tp_size = routed_tp_size
        for key in keys:
            layer_weights.pop(key)

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Route EP1 tokens, run two masked PPU grouped GEMMs, and gather."""

        if self.inter_local <= 0:
            raise RuntimeError("ppu_grouped_fp4 weights were not bound")
        if x.ndim != 2 or x.dtype != torch.bfloat16:
            raise TypeError(f"x must be BF16 [N,D], got dtype={x.dtype}, shape={x.shape}")
        if weights.ndim != 2 or weights.dtype != torch.float32:
            raise TypeError(
                "route weights must be FP32 [N,K], "
                f"got dtype={weights.dtype}, shape={weights.shape}"
            )
        if indices.ndim != 2 or indices.dtype != torch.int64:
            raise TypeError(
                "expert indices must be int64 [N,K], "
                f"got dtype={indices.dtype}, shape={indices.shape}"
            )
        if x.size(0) != weights.size(0) or weights.shape != indices.shape:
            raise ValueError(
                f"incompatible routed shapes x={x.shape}, weights={weights.shape}, "
                f"indices={indices.shape}"
            )
        if x.size(1) != int(self.cfg.dim):
            raise ValueError(f"x hidden dim must be {self.cfg.dim}, got {x.size(1)}")
        if any(tensor.device != x.device for tensor in (weights, indices)):
            raise ValueError("x, route weights, and expert indices must share one device")

        token_count, dim = x.shape
        if token_count == 0:
            return torch.zeros((0, dim), dtype=torch.float32, device=x.device)

        import deep_gemm

        grouped_gemm = getattr(
            deep_gemm, "m_grouped_gemm_fp4_fp4_bf16_nt_masked", None
        )
        if not callable(grouped_gemm):
            raise RuntimeError(
                "PPU deep_gemm lacks m_grouped_gemm_fp4_fp4_bf16_nt_masked"
            )
        from internal_source.rtp_llm.models_py.kernels.ppu_mxfp4 import (
            downcast_to_mxfp4,
        )
        from rtp_llm.models_py.modules.dsv4.moe.expert import require_silu_mul_split
        from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
            ep_gather,
            ep_scatter_v2,
            recompute_topk_ids_sum_expert_count,
        )

        experts = int(self.cfg.n_local_experts)
        adjusted_ids, expert_counts = recompute_topk_ids_sum_expert_count(
            indices.contiguous(),
            current_expert_start_id=0,
            num_local_experts=experts,
        )
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing:
            capacity = _select_capacity(
                self._configured_capacity,
                (),
                experts,
                fixed_shape=True,
                fixed_required=token_count * indices.size(1),
            )
        else:
            observed_counts = expert_counts.cpu().tolist()
            capacity = _select_capacity(
                self._configured_capacity,
                observed_counts,
                experts,
                fixed_shape=False,
            )
        expert_counts = expert_counts.to(torch.int32).contiguous()

        total = experts * capacity
        expert_start = torch.empty(experts, dtype=torch.int32, device=x.device)
        output_index = torch.full_like(adjusted_ids, -1, dtype=torch.int64)
        scatter_x = torch.zeros((total, dim), dtype=x.dtype, device=x.device)
        dummy_in_scale = torch.zeros(
            (token_count, dim // 128), dtype=torch.float32, device=x.device
        )
        dummy_out_scale = torch.zeros(
            (experts, capacity, dim // 128), dtype=torch.float32, device=x.device
        )
        ep_scatter_v2(
            x.contiguous(),
            dummy_in_scale,
            adjusted_ids,
            capacity,
            expert_start,
            scatter_x,
            dummy_out_scale,
            output_index,
            scale_ue8m0=False,
        )

        scatter_fp4, scatter_scale = downcast_to_mxfp4(scatter_x.contiguous())
        scatter_fp4 = scatter_fp4.view(experts, capacity, -1)
        scatter_scale = scatter_scale.as_strided(
            (experts, capacity, scatter_scale.size(1)),
            (capacity, 1, total),
        )
        scatter_scale = (
            scatter_scale.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        )
        gate_up = torch.empty(
            (experts, capacity, 2 * self.inter_local),
            dtype=torch.bfloat16,
            device=x.device,
        )
        expected_m = max(
            1,
            (
                int(self.cfg.max_tokens_per_rank)
                * int(self.cfg.n_activated_experts)
                + int(self.cfg.n_routed_experts)
                - 1
            )
            // int(self.cfg.n_routed_experts),
        )
        grouped_gemm(
            (scatter_fp4, scatter_scale),
            (self._ppu_w13, self._ppu_s13),
            None,
            gate_up,
            expert_counts,
            expected_m,
        )

        gate_up_flat = gate_up.view(total, 2 * self.inter_local)
        hidden = require_silu_mul_split()(
            gate_up_flat[:, : self.inter_local].float().contiguous(),
            gate_up_flat[:, self.inter_local :].float().contiguous(),
            clamp_limit=self.cfg.swiglu_limit,
        ).to(torch.bfloat16).contiguous()
        hidden_fp4, hidden_scale = downcast_to_mxfp4(hidden)
        hidden_fp4 = hidden_fp4.view(experts, capacity, -1)
        hidden_scale = hidden_scale.as_strided(
            (experts, capacity, hidden_scale.size(1)),
            (capacity, 1, total),
        )
        hidden_scale = hidden_scale.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        down = torch.empty(
            (experts, capacity, dim), dtype=torch.bfloat16, device=x.device
        )
        grouped_gemm(
            (hidden_fp4, hidden_scale),
            (self._ppu_w2, self._ppu_s2),
            None,
            down,
            expert_counts,
            expected_m,
        )

        gathered = torch.empty((token_count, dim), dtype=torch.bfloat16, device=x.device)
        ep_gather(
            down.view(total, dim),
            adjusted_ids,
            weights.contiguous(),
            output_index,
            gathered,
        )
        return gathered.float()


__all__ = ["PpuGroupedFP4Strategy"]
