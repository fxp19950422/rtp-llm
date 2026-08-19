"""DeepEPLowLatencyStrategy v2: INT8 走线 + dual-path (LL decode / NORMAL prefill).

Decode path (N <= ll_num_max_token_per_rank):
    low_latency_dispatch(use_int8=True, quant_size=D)
      → (int8 [E,M,D], fp32 [E,M,1]) — already quantised on the wire
    3x m_grouped_int8_gemm_nt_masked  (gate, up, down)
    1x silu_mul_clamp_quant_int8_masked_dsv4 (fused SiLU×up×clamp×quant)
    low_latency_combine(y, topk_idx, weights, handle)
      → router weights applied in combine

Prefill fallback path (N > cap):
    get_dispatch_layout + dispatch (NORMAL mode, same buffer)
    per-expert LocalLoopStrategy._forward_into_buf
    combine

Both paths are served by a single DeepEP buffer that has NVL+RDMA workspace.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from .. import ll_chunk_align
from ..expert import require_silu_mul_split
from ..warmup_sync import sync_cuda_graph_warmup_ranks
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .deepep import DeepEPStrategy, _DEEPEP_SUPPORTED_TOPK
from .local_loop import LocalLoopStrategy

# Supported hidden sizes for DeepEP low-latency dispatch.
_LL_SUPPORTED_HIDDEN_SIZES = (1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192)

# Scratch buffer cache keyed by (slot, device, shape, dtype).
# The slot tag is load-bearing: gate and up are same shape/dtype but must not alias.
_BUF_CACHE: Dict[Tuple[str, torch.device, Tuple[int, ...], torch.dtype], torch.Tensor] = {}


def _scratch(
    slot: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    zero: bool = False,
) -> torch.Tensor:
    key = (slot, device, tuple(shape), dtype)
    buf = _BUF_CACHE.get(key)
    if buf is None:
        buf = torch.empty(shape, dtype=dtype, device=device)
        _BUF_CACHE[key] = buf
    if zero:
        buf.zero_()
    return buf


@register_strategy
class DeepEPLowLatencyStrategy(RoutedExpertsStrategy):
    name = "deepep_low_latency"

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        # Stacked weights for the LL masked-GEMM path.
        self._W1_w: Optional[torch.Tensor] = None
        self._W1_s: Optional[torch.Tensor] = None
        self._W2_w: Optional[torch.Tensor] = None
        self._W2_s: Optional[torch.Tensor] = None
        self._W3_w: Optional[torch.Tensor] = None
        self._W3_s: Optional[torch.Tensor] = None
        # NORMAL fallback path (prefill): per-expert loop via LocalLoopStrategy.
        self._local = LocalLoopStrategy(cfg)

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        if cfg.ep_size <= 1:
            return False
        try:
            from internal_source.rtp_llm.models_py.kernels.ppu.deepgemm_wrapper import (
                has_deep_gemm_int8_grouped_masked,
            )
            if not has_deep_gemm_int8_grouped_masked():
                return False
        except Exception:
            return False
        try:
            from rtp_llm.models_py.distributed.deepep_wrapper import (
                DeepEPMode,
                DeepEPWrapper,
            )
            wrapper = DeepEPWrapper._instance
            if wrapper is None:
                return False
            return wrapper.mode == DeepEPMode.LOW_LATENCY
        except Exception:
            return False

    def setup_weights(self, layer_weights: Dict) -> None:
        from rtp_llm.utils.model_weight import W

        # Stash references for LL path BEFORE local pops them.
        self._W1_w = layer_weights[W.v4_routed_w1_w]
        self._W1_s = layer_weights[W.v4_routed_w1_s]
        self._W2_w = layer_weights[W.v4_routed_w2_w]
        self._W2_s = layer_weights[W.v4_routed_w2_s]
        self._W3_w = layer_weights[W.v4_routed_w3_w]
        self._W3_s = layer_weights[W.v4_routed_w3_s]

        # Normalise scales to fp32 (loader may hand fp32 already, but be safe).
        self._W1_s = self._W1_s.float().contiguous()
        self._W2_s = self._W2_s.float().contiguous()
        self._W3_s = self._W3_s.float().contiguous()

        # NORMAL fallback path: builds per-expert Expert instances.
        # _setup_weights_int8 pops from the dict — our stashed refs still point
        # to the same underlying tensors (pop removes the dict entry, not the data).
        self._local.setup_weights(layer_weights)

    def forward(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        wrapper = DeepEPWrapper._instance
        if wrapper is None:
            raise RuntimeError(
                "DeepEPWrapper not initialised; ep_size>1 requires "
                "init_deepep_wrapper() at engine startup."
            )
        if wrapper.mode != DeepEPMode.LOW_LATENCY:
            raise RuntimeError(
                f"{type(self).__name__} requires LOW_LATENCY mode, got {wrapper.mode}."
            )

        cfg = self.cfg
        D = cfg.dim
        if D not in _LL_SUPPORTED_HIDDEN_SIZES:
            raise RuntimeError(
                f"DeepEP LL does not support hidden_size={D}; "
                f"supported: {_LL_SUPPORTED_HIDDEN_SIZES}"
            )

        x_bf = x if x.dtype == torch.bfloat16 else x.to(torch.bfloat16)
        topk_idx = indices if indices.dtype == torch.int64 else indices.to(torch.int64)

        cap = wrapper.ll_num_max_token_per_rank
        N = x_bf.size(0)

        topk = topk_idx.size(1)
        if topk != wrapper.num_topk:
            raise RuntimeError(
                f"topk width {topk} != wrapper.num_topk {wrapper.num_topk}. "
                "LL buffer is sized from moe_k; padding is not allowed."
            )
        if cap <= 0:
            raise RuntimeError(
                f"{type(self).__name__}: invalid ll_num_max_token_per_rank={cap}"
            )

        if N <= cap:
            # LL path (decode / small prefill): INT8 走线 + masked GEMM.
            return self._forward_chunk_ll(wrapper, x_bf, weights, topk_idx)

        # Prefill with N > cap. The LL dispatch buffer holds `cap` tokens/rank.
        # The NORMAL dual-path dispatch on the LL-mode buffer is unreliable
        # (buf.dispatch() -> CPU recv timeout; see task history), so the prior
        # options were only: raise CONCURRENCY_LIMIT (which sets
        # ll_num_max_token_per_rank; linear buffer-memory cost, and STILL
        # crashes for any prefill past the configured cap). Instead, stream the
        # prefill tokens through the SAME working LL path in <=cap chunks and
        # stitch: MoE routing is per-token, so chunked dispatch/combine is
        # numerically exact. Copy each chunk's combine output into `out`
        # immediately -- the LL combine result aliases a rotating RDMA buffer
        # (at most 2 live), so holding chunk views for a later torch.cat would
        # corrupt earlier chunks.
        #
        # The chunk COUNT is per-rank data, while dispatch/combine are EP
        # collectives: ``moe/ll_chunk_align.py`` votes on the global count and
        # tops this rank up via ``pad_ll_calls`` so the group stays in lockstep.
        D = cfg.dim
        out = torch.empty((N, D), dtype=torch.bfloat16, device=x_bf.device)
        for s in range(0, N, cap):
            e = min(s + cap, N)
            out[s:e] = self._forward_chunk_ll(
                wrapper,
                x_bf[s:e].contiguous(),
                weights[s:e].contiguous(),
                topk_idx[s:e].contiguous(),
            )
        return out

    # =========================================================================
    # LL decode path
    # =========================================================================

    def pad_ll_calls(self, count: int) -> None:
        """Run ``count`` single-token dispatch/combine rounds and drop the result.

        Alignment padding for ranks whose prefill needed fewer chunks than the
        group maximum (see ``moe/ll_chunk_align.py``). The dummy routes one
        token to the first ``topk`` experts -- a legal routing, so the kernels
        take the same path as a real chunk -- and reuses the cached scratch
        buffers, so a pad round costs one dispatch/combine round trip and no
        extra memory.
        """
        if count <= 0:
            return
        from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper

        wrapper = DeepEPWrapper._instance
        if wrapper is None:
            return
        device = self._W1_w.device
        topk = wrapper.num_topk
        x_bf = _scratch("pad_x", (1, self.cfg.dim), torch.bfloat16, device, zero=True)
        weights = _scratch("pad_w", (1, topk), torch.float32, device, zero=True)
        topk_idx = _scratch("pad_idx", (1, topk), torch.int64, device)
        topk_idx.copy_(
            (
                torch.arange(topk, dtype=torch.int64, device=device)
                % wrapper.num_experts
            ).unsqueeze(0)
        )
        for _ in range(int(count)):
            self._forward_chunk_ll(wrapper, x_bf, weights, topk_idx)

    def _forward_chunk_ll(
        self,
        wrapper,
        x_bf: torch.Tensor,
        weights: torch.Tensor,
        topk_idx: torch.Tensor,
    ) -> torch.Tensor:
        from internal_source.rtp_llm.models_py.kernels.ppu.deepgemm_wrapper import (
            configure_deep_gemm_num_sms,
            deep_gemm_default_num_sms,
            m_grouped_int8_gemm_nt_masked,
        )
        from internal_source.rtp_llm.models_py.kernels.ppu.fused_silu_mul_int8_quant_dsv4 import (
            silu_mul_clamp_quant_int8_masked_dsv4,
        )

        cfg = self.cfg
        D = cfg.dim
        inter = cfg.moe_inter_dim
        n_local = cfg.n_local_experts
        device = x_bf.device

        # Bookkeeping for cross-rank chunk alignment: every dispatch/combine
        # round -- real or padding -- has to be counted (see ll_chunk_align).
        ll_chunk_align.note_ll_call()

        # 1. Dispatch with INT8 quantisation on the wire.
        # CUDA-graph capture prerequisite: low_latency_dispatch/combine are
        # EP collectives whose device side spin-waits for peer data. The C++
        # CudaGraphRunner runs two REAL warmup forwards and then captures
        # (recorded, not executed) -- per rank, with no cross-rank barrier. If
        # the ranks drift, a rank still executing its warmup dispatch waits for
        # a peer that has already moved into capture, and DeepEP times out
        # ("DeepEP timeout during graph capture", the reason ENABLE_CUDA_GRAPH
        # was pinned to 0 on M890P). The Mega* strategies already guard their
        # collective this way; this strategy was missing it. The helper is a
        # no-op outside CUDA-graph warmup and self-disables during capture, so
        # it costs nothing on the normal decode path.
        sync_cuda_graph_warmup_ranks(
            f"dsv4.deepep_ll.layer{cfg.layer_id}.before_dispatch",
            x_bf.device,
        )
        expert_x, recv_count, handle, _, _ = wrapper.buffer.low_latency_dispatch(
            x_bf,
            topk_idx,
            wrapper.ll_num_max_token_per_rank,
            wrapper.num_experts,
            use_fp8=False,
            use_int8=True,
            quant_size=D,
            async_finish=False,
            return_recv_hook=False,
        )
        # use_int8=True returns (tensor_i8, scale_fp32)
        if isinstance(expert_x, tuple):
            expert_x_i8, expert_x_s = expert_x
        else:
            raise TypeError(
                f"use_int8 dispatch must return (int8, scale) tuple, got {type(expert_x)}"
            )

        masked_m = (
            recv_count if recv_count.dtype == torch.int32
            else recv_count.to(torch.int32)
        )
        M_max = expert_x_i8.size(1)
        expected_m = max(
            1, int(x_bf.size(0) * wrapper.ep_size * topk_idx.size(1) // wrapper.num_experts)
        )
        num_sms = deep_gemm_default_num_sms()

        # 2. Gate/up masked GEMMs (share the same quantised lhs).
        gate_out = _scratch("gate", (n_local, M_max, inter), torch.bfloat16, device)
        up_out = _scratch("up", (n_local, M_max, inter), torch.bfloat16, device)

        with configure_deep_gemm_num_sms(num_sms):
            m_grouped_int8_gemm_nt_masked(
                (expert_x_i8, expert_x_s), (self._W1_w, self._W1_s),
                gate_out, masked_m, expected_m,
            )
            m_grouped_int8_gemm_nt_masked(
                (expert_x_i8, expert_x_s), (self._W3_w, self._W3_s),
                up_out, masked_m, expected_m,
            )

            # 3. Fused SiLU × mul × asymmetric clamp × per-token INT8 quant.
            down_q = _scratch("down_q", (n_local, M_max, inter), torch.int8, device)
            down_s = _scratch("down_s", (n_local, M_max, 1), torch.float32, device)
            silu_mul_clamp_quant_int8_masked_dsv4(
                gate_out, up_out, down_q, down_s, masked_m,
                clamp_limit=self.cfg.swiglu_limit,
            )

            # 4. Down projection. No zero-init: low_latency_combine reduces
            # only the valid slots encoded in the dispatch handle
            # (src_info/layout_range = per-expert [0, recv_count[e])), which is
            # exactly the region this masked GEMM writes (same masked_m). The
            # [recv_count[e], M_max) tail is never read by combine, so zeroing
            # the ~49MB slab every layer was pure waste. (DeepEP's own zero-init
            # requirement -- clean_low_latency_buffer -- is for the internal
            # RDMA workspace, not this output tensor.)
            y = _scratch("down_out", (n_local, M_max, D), torch.bfloat16, device)
            m_grouped_int8_gemm_nt_masked(
                (down_q, down_s), (self._W2_w, self._W2_s),
                y, masked_m, expected_m,
            )

        # 5. Combine: router weights applied here.
        combined, _, _ = wrapper.buffer.low_latency_combine(
            y, topk_idx, weights, handle,
            zero_copy=False, async_finish=False, return_recv_hook=False,
        )
        # Return combine's native (bf16) output instead of upcasting to fp32:
        # every consumer (``moe_layer._run_chunk``'s ``out.copy_`` and
        # ``combine_routed_and_shared``) re-casts to fp32 internally, so the
        # eager ``.float()`` here was a lossless-but-wasted upcast + downcast
        # round-trip on the combine hot path.
        return combined

    # =========================================================================
    # NORMAL prefill fallback path
    # =========================================================================

    def _forward_chunk_normal(
        self,
        wrapper,
        x_bf: torch.Tensor,
        weights: torch.Tensor,
        topk_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Standard DeepEP normal dispatch for tokens exceeding LL capacity."""
        cfg = self.cfg
        buf = wrapper.buffer

        # Pad topk to nearest supported value (V4's 6 → 8).
        indices_p, weights_p = DeepEPStrategy._pad_topk_for_deepep(topk_idx, weights)

        # 1. Dispatch layout.
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buf.get_dispatch_layout(indices_p, cfg.n_routed_experts)

        # 2. Dispatch.
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            _,
        ) = buf.dispatch(
            x_bf,
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            indices_p,
            weights_p,
            expert_alignment=1,
        )

        # 3. Local per-expert compute.
        M = recv_x.size(0)
        if M > 0:
            global_topk_idx = recv_topk_idx.to(torch.int64).contiguous()
            global_topk_idx = torch.where(
                global_topk_idx == -1,
                global_topk_idx,
                global_topk_idx + cfg.local_expert_start,
            )
            y_local = self._local._forward_into_buf(
                recv_x.contiguous(),
                recv_topk_weights.contiguous(),
                global_topk_idx,
                local_start=cfg.local_expert_start,
                local_end=cfg.local_expert_end,
            )
        else:
            y_local = torch.zeros(M, cfg.dim, dtype=torch.float32, device=recv_x.device)

        # 4. Combine.
        y_combined, _, _ = buf.combine(y_local.to(x_bf.dtype), handle)
        return y_combined.float()
