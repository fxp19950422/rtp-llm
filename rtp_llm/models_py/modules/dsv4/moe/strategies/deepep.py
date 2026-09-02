"""DeepEPStrategy: ACCL-EP normal-mode dispatch + per-expert local compute + combine.

EP > 1 DeepEP implementation. DSV4 automatic strategy selection no longer
falls back here when Mega is unavailable; EP>1 requires Mega and fails fast.
This class is kept as an explicit implementation for targeted tests or
experiments. Composes ``LocalLoopStrategy`` for the local per-expert compute
on the dispatched recv tokens.

Direct port of the pre-refactor ``_routed_experts_deepep`` +
``_pad_topk_for_deepep`` + the ``_DEEPEP_SUPPORTED_TOPK`` constant.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ...runtime_config import (
    get_switch,
    parse_nonneg_int,
    parse_percentile_list,
)
from ..warmup_sync import (
    cuda_graph_warmup_forward_enabled,
    sync_cuda_graph_warmup_ranks,
)
from .base import MoeCfg, RoutedExpertsStrategy, register_strategy
from .local_loop import LocalLoopStrategy


_CAPACITY_OVERFLOW_MODE = (
    os.environ.get("DSV4_MOE_CAPACITY_OVERFLOW_MONITOR", "count").strip().lower()
)
if _CAPACITY_OVERFLOW_MODE not in ("off", "count"):
    raise ValueError(
        "invalid DSV4_MOE_CAPACITY_OVERFLOW_MONITOR="
        f"{_CAPACITY_OVERFLOW_MODE!r}; expected off|count"
    )


class _CapacityOverflowMonitor:
    """Count the expert routes the fixed capacity silently discards.

    Observation only: it does not change which routes are dropped.  The fixed
    capacity (128 by default) is >8x the measured mean, but that headroom only
    holds at the current decode batch size -- and growing that batch is exactly
    what the decode optimisation work does.  Slot 1 (max load seen) is the
    early-warning signal: it shows the remaining headroom while slot 0 is still
    zero.

    Graph safety is the binding constraint: ``record`` uses device ops only --
    no ``.item()``, ``.cpu()``, ``print`` or ``assert`` -- so it is safe inside
    a captured graph.  ``read`` syncs and must only be called from outside
    capture/replay.
    """

    __slots__ = ("_stats", "_per_expert")

    def __init__(self) -> None:
        self._stats: Optional[torch.Tensor] = None
        # Per-expert token counts, allocated only while the outside-graph
        # readout is enabled (DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY > 0).  Device-only
        # accumulation, like ``_stats``: captured into the graph so replays
        # advance it with zero host involvement.  Never synced per step.
        self._per_expert: Optional[torch.Tensor] = None

    def record(self, counts: torch.Tensor, capacity: int) -> None:
        if _CAPACITY_OVERFLOW_MODE == "off":
            return
        if counts.numel() == 0:
            return
        # Lazy allocation happens BEFORE the warmup guard on purpose: the
        # first record must allocate OUTSIDE the capture (the eager warmup
        # forwards run first — cuda_graph_runner.cc:1319-1343).  A zeros()
        # fill kernel recorded inside the graph would zero these accumulators
        # on every replay.
        if self._stats is None:
            self._stats = torch.zeros(2, dtype=torch.int64, device=counts.device)
        if _OVF_OUTSIDE_EVERY > 0 and self._per_expert is None:
            self._per_expert = torch.zeros(
                counts.numel(), dtype=torch.int64, device=counts.device
            )
        # Warmup forwards run on dummy inputs just before each capture
        # (RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD=1).  Their routing counts can be
        # inflated and would poison the first readout window with fake
        # overflow warnings, so skip the accumulation — the allocation above
        # still runs so the captured graph records only the adds.
        if cuda_graph_warmup_forward_enabled():
            return
        c64 = counts.to(torch.int64)
        # dropped == sum(max(0, c - capacity)), reusing the clamp the caller needs
        self._stats[0] += c64.sum() - c64.clamp(max=capacity).sum()
        self._stats[1] = torch.maximum(self._stats[1], c64.max())
        if _OVF_OUTSIDE_EVERY > 0:
            self._per_expert += c64

    def read(self, reset: bool = True):
        """Return ``(dropped_routes, max_count_seen)`` or ``None``.

        Performs a device-to-host sync, which a CUDA graph capture forbids.
        Callers guard too, but guard here as well: this is a debug reader and a
        future caller reaching it from inside a capture would abort the engine,
        not just lose a statistic.
        """
        if self._stats is None:
            return None
        if _graph_capture_active():
            return None
        vals = self._stats.tolist()
        if reset:
            self._stats.zero_()
        return int(vals[0]), int(vals[1])

    def read_per_expert(self, reset: bool = True) -> "Optional[List[int]]":
        """Return per-expert accumulated counts as a host list, or ``None``.

        Same capture-safety contract as ``read``: one ``.tolist()`` sync,
        guarded against graph capture.  ``None`` when per-expert tracking is
        off (the default) or the monitor never recorded anything.
        """
        if self._per_expert is None:
            return None
        if _graph_capture_active():
            return None
        vals = self._per_expert.tolist()
        if reset:
            self._per_expert.zero_()
        return [int(v) for v in vals]


_CAPACITY_OVERFLOW_INSTANCES: "list" = []


def read_capacity_overflow_stats(reset: bool = True):
    """Aggregate every live monitor.  Host-side only (syncs); see ``read``."""
    dropped = 0
    max_seen = 0
    seen_any = False
    for mon in _CAPACITY_OVERFLOW_INSTANCES:
        got = mon.read(reset=reset)
        if got is None:
            continue
        seen_any = True
        dropped += got[0]
        max_seen = max(max_seen, got[1])
    if not seen_any:
        return None
    return dropped, max_seen


def _percentiles_from_counts(values, percentiles):
    """Nearest-rank percentiles of host-side per-expert counts.

    Only called on low-frequency readout paths where the counts are already
    on the host (one ``.tolist()``): sorting the host list avoids the second
    device sync a ``torch.quantile`` call would need.  "p50" of [0..31] is
    element 15 (0-based), p99 of a 32-vector is the max — the conservative
    end of the classic nearest-rank definition.
    """
    if not values:
        return []
    ordered = sorted(values)
    n = len(ordered)
    out = []
    for pct in percentiles:
        idx = max(0, math.ceil(int(pct) * n / 100.0) - 1)
        out.append(ordered[min(idx, n - 1)])
    return out


def drain_capacity_overflow_outside_graph() -> None:
    """Outside-graph readout of the capacity-overflow monitor.

    Under CUDA-graph decode the eager logger (``_maybe_log_capacity_overflow``)
    is silent: replay re-fires kernels without executing the Python forward,
    so its call counter never advances in production.  This hook is the
    out-of-graph Python seam every replay still crosses — C++
    ``CudaGraphRunner::prepareInputs`` calls the decode fmha impl's
    ``prepare_cuda_graph`` between replays, and the impls call this function
    from there (``decode/decode_fmha_impl.py``, ``fp8/decode/decode_fmha_impl.py``).

    ``DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY=K`` (K>0) reads the aggregated stats
    every K hook calls and logs one ``[CAPACITY_OVERFLOW]`` line covering a
    fresh window (stats reset on readout, so each line reports the last K
    calls).  K=1 is expensive — each readout costs 2 host syncs per monitor
    instance (``read`` + ``read_per_expert``) — so prefer K>=50.
    Per-expert counts are aggregated across monitor instances with an
    elementwise max (each instance is one MoE layer; the max keeps the
    "busiest layer" view consistent with how ``max_count_seen`` aggregates).
    Percentile fields are controlled by ``DSV4_MOE_ROUTER_PERCENTILES``.

    Default (K=0) is a no-op with zero overhead: byte-identical legacy
    behaviour.  The isolation testbench can also call this directly (or use
    ``read_capacity_overflow_stats``) without any graph.

    Thread safety: ``prepare_cuda_graph`` may run concurrently on the engine
    main thread and an AsyncRunner worker thread (cuda_graph_runner.cc:690-696),
    so everything past the disabled-early-exit (count increment, window check,
    readout, reset) happens under ``_OVF_DRAIN_LOCK``.
    """
    if _OVF_OUTSIDE_EVERY <= 0:
        return
    with _OVF_DRAIN_LOCK:
        if _graph_capture_active():
            return
        _OVF_OUTSIDE_CALLS[0] += 1
        if _OVF_OUTSIDE_CALLS[0] % _OVF_OUTSIDE_EVERY:
            return
        dropped = 0
        max_seen = 0
        seen_any = False
        per_expert_max: Optional[List[int]] = None
        for mon in _CAPACITY_OVERFLOW_INSTANCES:
            got = mon.read(reset=True)
            if got is None:
                continue
            seen_any = True
            dropped += got[0]
            max_seen = max(max_seen, got[1])
            counts = mon.read_per_expert(reset=True)
            if counts is None:
                continue
            if per_expert_max is None:
                per_expert_max = list(counts)
            else:
                if len(counts) > len(per_expert_max):
                    per_expert_max.extend([0] * (len(counts) - len(per_expert_max)))
                for i, c in enumerate(counts):
                    if c > per_expert_max[i]:
                        per_expert_max[i] = c
        if not seen_any:
            return
        active = sum(1 for c in (per_expert_max or ()) if c > 0)
        pctl = ""
        if per_expert_max is not None and _OVF_ROUTER_PERCENTILES:
            qs = _percentiles_from_counts(per_expert_max, _OVF_ROUTER_PERCENTILES)
            pctl = " " + " ".join(
                "p%d=%d" % (p, q)
                for p, q in zip(_OVF_ROUTER_PERCENTILES, qs)
            )
        logging.info(
            "[CAPACITY_OVERFLOW] ts=%.3f dropped_routes=%d max_count_seen=%d "
            "active_experts=%d%s",
            time.time(),
            dropped,
            max_seen,
            active,
            pctl,
        )


# 0 disables the log.  N>0 emits one line every N eager calls.
_OVF_LOG_EVERY = int(os.environ.get("DSV4_MOE_CAPACITY_OVERFLOW_LOG_EVERY", "0"))
_OVF_LOG_CALLS = [0]

# --- Outside-graph overflow readout (new switches; runtime_config surface) ---
# Production decode replays CUDA graphs: the captured Python forward (and
# with it ``_maybe_log_capacity_overflow``'s eager call counter) only runs
# during capture, so the overflow stats stay silent under replay.  The new
# switches below read through runtime_config (fail-loud + one [DSV4_CONFIG]
# audit line each); defaults keep byte-identical legacy behaviour.
#
# DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY=K (K>0): read the aggregated monitor
# stats every K outside-graph hook calls (``drain_capacity_overflow_outside_graph``,
# wired into the decode fmha impls' ``prepare_cuda_graph`` — the one host
# seam C++ ``CudaGraphRunner::prepareInputs`` crosses between replays).
# Enabling it also makes every monitor keep a per-expert token-count vector
# (one extra elementwise device add per layer per captured step — recorded
# into the graph, so replays keep it advancing; never synced per step).
#
# Name map (do not confuse the two overflow throttles):
#   * DSV4_MOE_CAPACITY_OVERFLOW_LOG_EVERY   (legacy)  — eager-path log
#     throttle: only advances while Python forwards run (silently inert under
#     graph replay).
#   * DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY     (new)    — graph-replay-path
#     readout throttle: counts ``prepare_cuda_graph`` seam calls, i.e. every
#     replayed decode step.
_OVF_OUTSIDE_EVERY = get_switch(
    "DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY", 0, parse_nonneg_int
)
_OVF_OUTSIDE_CALLS = [0]
# ``prepare_cuda_graph`` runs on both the engine main thread and an
# AsyncRunner worker thread (cuda_graph_runner.cc:690-696), so the drain's
# count-then-maybe-readout sequence needs a lock: the += on the list above is
# a non-atomic read-modify-write, and an interleaving can lose an increment
# or double-trigger a window (the latecomer reads post-reset zeros and logs
# a fake empty window).
_OVF_DRAIN_LOCK = threading.Lock()
# DSV4_MOE_ROUTER_PERCENTILES="p50,p99": which per-expert routing percentiles
# to append on readout, applied both to the eager router-stats line below and
# to the outside-graph [CAPACITY_OVERFLOW] line.  Unset/empty == off.
_OVF_ROUTER_PERCENTILES = get_switch(
    "DSV4_MOE_ROUTER_PERCENTILES", (), parse_percentile_list
)
# DSV4_MOE_ROUTER_PERCENTILE_LOG_EVERY=K (K>0): independent throttle for the
# eager router-stats percentiles — computing them needs a whole-vector
# ``.tolist()`` host sync, so they can be sampled sparser than the legacy
# 5-scalar line.  0 == never (legacy lines stay byte-identical).
# NOTE: only takes effect when the legacy eager log itself is on, i.e.
# DSV4_MOE_CAPACITY_OVERFLOW_LOG_EVERY > 0 — this throttle lives on the same
# line; on a graph-replay decode arm use DSV4_MOE_OVERFLOW_REPLAY_LOG_EVERY
# + DSV4_MOE_ROUTER_PERCENTILES instead.
_OVF_ROUTER_PCTL_EVERY = get_switch(
    "DSV4_MOE_ROUTER_PERCENTILE_LOG_EVERY", 0, parse_nonneg_int
)
_OVF_ROUTER_PCTL_CALLS = [0]


def _graph_capture_active() -> bool:
    """True while a CUDA graph capture is in flight on the current stream.

    Reading the monitor syncs device->host, which the capture forbids.  Ask the
    runtime rather than trusting the caller to only enable the log on eager
    runs: production decode captures graphs, so a caller-side rule is one
    forgotten export away from aborting the engine during warmup.
    """
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def _maybe_log_capacity_overflow(capacity, counts=None):
    """Emit overflow stats every ``_OVF_LOG_EVERY`` eager calls.

    Sits on the forward path, so it must be a no-op during graph capture; the
    call counter only advances on calls that could actually log, otherwise the
    period would depend on how many capture passes ran first.
    """
    if _OVF_LOG_EVERY <= 0:
        return
    if _graph_capture_active():
        return
    _OVF_LOG_CALLS[0] += 1
    if _OVF_LOG_CALLS[0] % _OVF_LOG_EVERY:
        return
    got = read_capacity_overflow_stats(reset=False)
    if got is None:
        return
    dist = ""
    if counts is not None and counts.numel() > 0:
        c64 = counts.to(torch.int64)
        # One sync for the whole tuple; capture already ruled out above.
        vals = torch.stack(
            [
                c64.sum(),
                c64.max(),
                c64.argmax().to(torch.int64),
                (c64 > 0).sum(),
                c64.min(),
            ]
        ).tolist()
        dist = (
            " n_local_experts=%d routes_sum=%d step_max=%d step_argmax=%d"
            " active_experts=%d step_min=%d"
            % (counts.numel(), vals[0], vals[1], vals[2], vals[3], vals[4])
        )
        # Per-expert routing percentiles (p50/p99 by convention): computed
        # here — on the throttled readout only — from the current step's
        # per-expert counts, so the percentile fields share the same window
        # as the max/min/sum/active fields above.  Both switches default to
        # off, keeping the legacy line byte-identical.
        if _OVF_ROUTER_PCTL_EVERY > 0 and _OVF_ROUTER_PERCENTILES:
            _OVF_ROUTER_PCTL_CALLS[0] += 1
            if not _OVF_ROUTER_PCTL_CALLS[0] % _OVF_ROUTER_PCTL_EVERY:
                qs = _percentiles_from_counts(
                    c64.tolist(), _OVF_ROUTER_PERCENTILES
                )
                dist += " " + " ".join(
                    "p%d=%d" % (p, q)
                    for p, q in zip(_OVF_ROUTER_PERCENTILES, qs)
                )
    logging.info(
        "[CAPACITY_OVERFLOW] calls=%d capacity=%d dropped_routes=%d max_count_seen=%d%s",
        _OVF_LOG_CALLS[0],
        capacity,
        got[0],
        got[1],
        dist,
    )


def _ppu_deepep_compact_copy_2d_enabled() -> bool:
    return os.environ.get("DSV4_PPU_DEEPEP_COMPACT_COPY_2D", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def _ppu_grouped_fp4_enabled() -> bool:
    return os.environ.get("DSV4_PPU_GROUPED_FP4", "0").strip().lower() in (
        "1",
        "true",
        "on",
        "yes",
    )


def _select_ppu_grouped_fp4_capacity(
    configured_capacity: int,
    num_recv_tokens_per_expert: Sequence[int],
    n_local_experts: int,
    *,
    fixed_shape: bool,
) -> int:
    """Choose an aligned grouped-GEMM capacity without truncating eager Prefill.

    Decode graph capture/replay must retain the configured fixed shape. Eager
    Prefill already receives exact per-expert counts from DeepEP's dynamic
    dispatch CPU result, so grow the capacity to cover the busiest local
    expert and round it to the kernel's ``E * capacity`` alignment.
    """
    if configured_capacity <= 0 or n_local_experts <= 0:
        raise ValueError("grouped-FP4 capacity and local expert count must be positive")
    alignment = 128 // math.gcd(128, n_local_experts)
    if configured_capacity % alignment:
        raise ValueError(
            "DSV4_PPU_GROUPED_FP4_CAPACITY must make "
            "n_local_experts*capacity divisible by 128"
        )
    if fixed_shape:
        return configured_capacity
    if not num_recv_tokens_per_expert:
        raise ValueError("eager grouped-FP4 requires DeepEP CPU expert counts")
    required = max(int(count) for count in num_recv_tokens_per_expert)
    if required < 0:
        raise ValueError("DeepEP expert token counts must be non-negative")
    selected = max(configured_capacity, required)
    return ((selected + alignment - 1) // alignment) * alignment


# Spend the whole DeepEP LL slot on compute instead of compacting a prefix out of
# it.  0 = compact (current behaviour).  The slot's geometry already covers the
# worst case -- every token routed to one expert -- so the clamp that guards the
# compact path can no longer bind and no route can be dropped, at the cost of
# running the experts over rows the dispatch never filled.
_LL_NO_COMPACT = os.environ.get("DSV4_MOE_LL_NO_COMPACT", "0").strip().lower() in (
    "1",
    "true",
    "on",
    "yes",
)
_LL_NO_COMPACT_LOGGED = [False]


# The masked SwiGLU+MXFP4 kernel computes only the rows below each expert's
# count, so the activation stops scaling with the compute slot.  It is wired next
# to ``_LL_NO_COMPACT`` because the two are one candidate, not two: in one
# in-situ decode measurement the activation is 27.8 us compacted, 436.1 us with
# no-compact alone, and 13.8 us with no-compact plus this kernel.  Enabling only
# the former is a net loss.
_MASKED_SILU = os.environ.get("DSV4_MOE_MASKED_SILU", "0").strip().lower() in (
    "1",
    "true",
    "on",
    "yes",
)
_MASKED_SILU_LOGGED = [False]
_MASKED_SILU_SKIP_LOGGED = [False]
# The kernel emits scale bytes for the block-padded hidden width.  When the width
# is a multiple of the block the padding is empty and the returned view carries
# the same strides the compact path materializes; otherwise the expert stride
# grows, and no masked GEMM has been run against that stride.
_MASKED_SILU_BLOCK_N = 256


def _tensor_byte_span(t) -> Tuple[int, int]:
    """Exact ``[start, end)`` byte range a strided tensor can touch.

    ``nbytes`` would understate a non-contiguous view and overstate a narrowed
    one; the aliasing check below needs the real reach, so walk the strides.
    """
    if t.numel() == 0:
        return (0, 0)
    start = t.data_ptr()
    last = sum((s - 1) * st for s, st in zip(t.shape, t.stride()))
    return (start, start + (last + 1) * t.element_size())


def _spans_overlap(a, b) -> bool:
    a0, a1 = _tensor_byte_span(a)
    b0, b1 = _tensor_byte_span(b)
    if a1 == 0 or b1 == 0:
        return False
    return a0 < b1 and b0 < a1


# ACCL-EP's intranode dispatch kernel has a compile-time switch over
# ``num_topk`` that only covers {2, 4, 8, 16} (asserts false on others —
# intranode.cu:2237 "Unsupported num_topk"). V4-Flash uses
# ``n_activated_experts = 6``; we pad both ``indices`` and ``weights``
# up to 8 slots with ``-1`` and ``0.0`` so the dispatch accepts them,
# and the padding slots are silently dropped by the per-expert loop
# (``torch.where(idx == -1)`` never matches a real expert index).
_DEEPEP_SUPPORTED_TOPK = (2, 4, 8, 16)


@register_strategy
class DeepEPStrategy(RoutedExpertsStrategy):
    name = "deepep"

    def __init__(self, cfg: MoeCfg):
        super().__init__(cfg)
        # Composition: hold a LocalLoopStrategy instance for the per-expert
        # local compute on dispatched recv tokens. Registered as a child
        # nn.Module so its ``experts`` ModuleList propagates through
        # ``MoE.to(device)`` / state_dict.
        self._local = LocalLoopStrategy(cfg)
        self._ppu_grouped_fp4 = False
        # Fixed-capacity overflow observability; see _CapacityOverflowMonitor.
        self._ovf = _CapacityOverflowMonitor()
        _CAPACITY_OVERFLOW_INSTANCES.append(self._ovf)
        self._ppu_deepep_compact_copy_2d = _ppu_deepep_compact_copy_2d_enabled()
        if self._ppu_deepep_compact_copy_2d:
            from ._compact_prefix_copy import prepare_compact_prefix_copy

            # Resolve the cudart entry points now: doing it lazily would put a
            # dlopen on the first captured forward.
            prepare_compact_prefix_copy()

    @classmethod
    def can_handle(cls, cfg: MoeCfg) -> bool:
        # ep_size > 1. Mega-vs-DeepEP priority is enforced by registry order
        # (Mega registered first).
        return cfg.ep_size > 1

    def setup_weights(self, layer_weights: Dict) -> None:
        """Delegates to ``LocalLoopStrategy.setup_weights`` — DeepEP has no
        weights of its own; it dispatches recv tokens to the per-expert loop
        owned by the inner ``LocalLoopStrategy``.
        """
        if not _ppu_grouped_fp4_enabled():
            self._local.setup_weights(layer_weights)
            return

        # M890P checkpoints keep routed experts as packed MXFP4. Preserve the
        # native payloads and prepare E8M0 checkpoint scales once, then execute
        # all local experts with two grouped GEMMs instead of 3*E small GEMMs.
        # The opt-in is deliberately strict: other storage geometries continue
        # to use LocalLoopStrategy and cannot silently enter this platform path.
        from rtp_llm.utils.model_weight import W

        w1 = layer_weights.pop(W.v4_routed_w1_w)
        s1 = layer_weights.pop(W.v4_routed_w1_s)
        w2 = layer_weights.pop(W.v4_routed_w2_w)
        s2 = layer_weights.pop(W.v4_routed_w2_s)
        w3 = layer_weights.pop(W.v4_routed_w3_w)
        s3 = layer_weights.pop(W.v4_routed_w3_s)
        cfg = self.cfg
        expected_w1 = (cfg.n_local_experts, cfg.moe_inter_dim, cfg.dim // 2)
        expected_w2 = (cfg.n_local_experts, cfg.dim, cfg.moe_inter_dim // 2)
        if tuple(w1.shape) != expected_w1 or tuple(w3.shape) != expected_w1:
            raise ValueError(
                f"PPU grouped-FP4 w1/w3 must have shape {expected_w1}, got "
                f"{tuple(w1.shape)}/{tuple(w3.shape)}"
            )
        if tuple(w2.shape) != expected_w2:
            raise ValueError(
                f"PPU grouped-FP4 w2 must have shape {expected_w2}, got {tuple(w2.shape)}"
            )
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if any(w.dtype not in (torch.int8, torch.uint8) for w in (w1, w2, w3)):
            raise TypeError("PPU grouped-FP4 requires packed int8/uint8 routed weights")
        if e8m0_dtype is None or any(s.dtype != e8m0_dtype for s in (s1, s2, s3)):
            raise TypeError("PPU grouped-FP4 requires float8_e8m0fnu checkpoint scales")
        if not w1.is_cuda or torch.cuda.get_device_name(w1.device) != "ZW-M890P":
            raise RuntimeError("DSV4_PPU_GROUPED_FP4=1 requires ZW-M890P weights")

        from internal_source.rtp_llm.models_py.kernels.ppu_mxfp4 import (
            prepare_fp4_weight_scale_mxfp4,
        )

        self.register_buffer(
            "_ppu_w13",
            torch.cat((w1, w3), dim=1).view(torch.uint8).contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "_ppu_s13",
            prepare_fp4_weight_scale_mxfp4(torch.cat((s1, s3), dim=1).contiguous()),
            persistent=False,
        )
        self.register_buffer(
            "_ppu_w2", w2.view(torch.uint8).contiguous(), persistent=False
        )
        self.register_buffer(
            "_ppu_s2", prepare_fp4_weight_scale_mxfp4(s2.contiguous()),
            persistent=False,
        )
        self._ppu_grouped_fp4 = True

    def _forward_ppu_grouped_fp4(
        self,
        recv_x: torch.Tensor,
        recv_topk_weights: torch.Tensor,
        recv_topk_idx: torch.Tensor,
        capacity: int,
    ) -> torch.Tensor:
        """Fixed-capacity, CUDA-graph-safe grouped MXFP4 local expert compute."""
        import deep_gemm

        from internal_source.rtp_llm.models_py.kernels.ppu_mxfp4 import (
            downcast_to_mxfp4,
        )
        from rtp_llm.models_py.modules.dsv4.moe.expert import require_silu_mul_split
        from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
            ep_gather,
            ep_scatter_v2,
            recompute_topk_ids_sum_expert_count,
        )

        cfg = self.cfg
        M, D = recv_x.shape
        E = cfg.n_local_experts
        inter = cfg.moe_inter_dim
        if capacity <= 0 or (E * capacity) % 128:
            raise ValueError(
                "DSV4_PPU_GROUPED_FP4_CAPACITY must be positive and "
                "n_local_experts*capacity must be divisible by 128"
            )

        local_idx = recv_topk_idx.to(torch.int64).contiguous()
        adjusted_idx, actual_counts = recompute_topk_ids_sum_expert_count(
            local_idx, current_expert_start_id=0, num_local_experts=E
        )

        total = E * capacity
        safe_counts = actual_counts.clamp(max=capacity).to(torch.int32).contiguous()
        expert_start = torch.empty(E, dtype=torch.int32, device=recv_x.device)
        output_index = torch.full_like(adjusted_idx, -1, dtype=torch.int64)
        scatter_x = torch.zeros((total, D), dtype=recv_x.dtype, device=recv_x.device)
        # ep_scatter is dtype-generic. Dummy scales let us reuse its fixed
        # expert-layout/index kernel for BF16; MXFP4 quantization follows.
        dummy_in_scale = torch.zeros(
            (M, D // 128), dtype=torch.float32, device=recv_x.device
        )
        dummy_out_scale = torch.zeros(
            (E, capacity, D // 128), dtype=torch.float32, device=recv_x.device
        )
        ep_scatter_v2(
            recv_x,
            dummy_in_scale,
            adjusted_idx,
            capacity,
            expert_start,
            scatter_x,
            dummy_out_scale,
            output_index,
            scale_ue8m0=False,
        )

        scatter_fp4, scatter_scale = downcast_to_mxfp4(scatter_x.contiguous())
        scatter_fp4_grouped = scatter_fp4.view(E, capacity, -1)
        scatter_scale_grouped = scatter_scale.as_strided(
            (E, capacity, scatter_scale.size(1)),
            (capacity, 1, total),
        )
        # GroupedMasked assumes each expert owns a compact [K/64, M]
        # scale block; the flat packer instead owns one global [K/64, E*M]
        # block. Materialize the expert-major physical layout while retaining
        # the logical [E, M, K/64] mn-major view required by DeepGEMM.
        scatter_scale_grouped = (
            scatter_scale_grouped.permute(0, 2, 1)
            .contiguous()
            .permute(0, 2, 1)
        )
        gate_up_grouped = torch.empty(
            (E, capacity, 2 * inter),
            dtype=torch.bfloat16,
            device=recv_x.device,
        )
        expected_m = max(
            1,
            (
                cfg.max_tokens_per_rank
                * cfg.ep_size
                * cfg.n_activated_experts
                + cfg.n_routed_experts
                - 1
            )
            // cfg.n_routed_experts,
        )
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (scatter_fp4_grouped, scatter_scale_grouped),
            (self._ppu_w13, self._ppu_s13),
            None,
            gate_up_grouped,
            safe_counts,
            expected_m,
        )
        gate_up = gate_up_grouped.view(total, 2 * inter)
        hidden = require_silu_mul_split()(
            gate_up[:, :inter].float().contiguous(),
            gate_up[:, inter:].float().contiguous(),
            clamp_limit=cfg.swiglu_limit,
        ).to(torch.bfloat16).contiguous()
        hidden_fp4, hidden_scale = downcast_to_mxfp4(hidden)
        hidden_fp4_grouped = hidden_fp4.view(E, capacity, -1)
        hidden_scale_grouped = hidden_scale.as_strided(
            (E, capacity, hidden_scale.size(1)),
            (capacity, 1, total),
        )
        hidden_scale_grouped = (
            hidden_scale_grouped.permute(0, 2, 1)
            .contiguous()
            .permute(0, 2, 1)
        )
        down_grouped = torch.empty(
            (E, capacity, D), dtype=torch.bfloat16, device=recv_x.device
        )
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (hidden_fp4_grouped, hidden_scale_grouped),
            (self._ppu_w2, self._ppu_s2),
            None,
            down_grouped,
            safe_counts,
            expected_m,
        )
        down = down_grouped.view(total, D)
        gathered = torch.empty((M, D), dtype=torch.bfloat16, device=recv_x.device)
        ep_gather(
            down,
            adjusted_idx,
            recv_topk_weights.contiguous(),
            output_index,
            gathered,
        )
        return gathered.float()

    def _compute_ppu_grouped_fp4_packed(
        self,
        expert_x,
        expert_num_tokens: torch.Tensor,
        out: "Optional[torch.Tensor]" = None,
    ) -> torch.Tensor:
        """Run grouped MXFP4 experts on DeepEP LL's compact payload.

        ``out``, when its geometry matches exactly, receives the down-projection
        directly so the caller can skip a full-slot copy.  A mismatch is not an
        error: it just means the compact path is active and the buffer is the
        wrong shape, so allocate as before.
        """
        import deep_gemm

        from internal_source.rtp_llm.models_py.kernels.ppu_mxfp4 import (
            downcast_to_mxfp4,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        cfg = self.cfg
        packed_dispatch = isinstance(expert_x, tuple)
        if packed_dispatch:
            if len(expert_x) != 2:
                raise ValueError("DeepEP LL MXFP4 dispatch must return data and scales")
            packed_x, packed_scale = expert_x
            if packed_x.dim() != 3 or packed_scale.dim() != 3:
                raise ValueError("DeepEP LL MXFP4 tensors must both be rank 3")
            E, ll_capacity, packed_D = packed_x.shape
            D = packed_D * 2
            device = packed_x.device
        else:
            if not isinstance(expert_x, torch.Tensor) or expert_x.dim() != 3:
                raise ValueError(
                    "DeepEP LL input must be BF16 [E, M, D] or an MXFP4 tuple"
                )
            E, ll_capacity, D = expert_x.shape
            device = expert_x.device
        if (E, D) != (cfg.n_local_experts, cfg.dim):
            raise ValueError(
                f"grouped-FP4 packed input must be "
                f"[{cfg.n_local_experts}, M, {cfg.dim}], got E={E}, M={ll_capacity}, D={D}"
            )
        compute_capacity = int(
            os.environ.get("DSV4_PPU_GROUPED_FP4_CAPACITY", "128")
        )
        if _LL_NO_COMPACT:
            # Take the slot as-is.  ``safe_counts`` below then clamps against a
            # bound the counts cannot exceed, so the overflow the monitor watches
            # for becomes unreachable rather than merely rare.  The validation
            # below still applies: an LL geometry that breaks the kernel's
            # alignment must fail loudly, not silently fall back to compacting.
            compute_capacity = ll_capacity
            if not _LL_NO_COMPACT_LOGGED[0]:
                _LL_NO_COMPACT_LOGGED[0] = True
                logging.info(
                    "[LL_NO_COMPACT] on: E=%d ll_capacity=%d -> compute_capacity=%d",
                    E,
                    ll_capacity,
                    compute_capacity,
                )
        if (
            compute_capacity <= 0
            or compute_capacity > ll_capacity
            or (E * compute_capacity) % 128
        ):
            raise ValueError(
                "grouped-FP4 compute capacity must be positive, no larger than "
                "the DeepEP LL slot, and E*capacity divisible by 128"
            )
        inter = cfg.moe_inter_dim
        total = E * compute_capacity
        if expert_num_tokens.numel() != E:
            raise ValueError(
                f"grouped-FP4 expert counts must have {E} elements, "
                f"got {expert_num_tokens.numel()}"
            )
        self._ovf.record(expert_num_tokens, compute_capacity)
        _maybe_log_capacity_overflow(compute_capacity, expert_num_tokens)
        safe_counts = (
            expert_num_tokens.clamp(min=0, max=compute_capacity)
            .to(torch.int32)
            .contiguous()
        )
        # DeepEP LL reserves ``max_tokens_per_rank * ep_size`` rows per expert
        # for a theoretical all-to-one route.  Quantizing that entire slot
        # would erase the LL communication win.  The production grouped-MXFP4
        # path already uses a conservative fixed expert capacity (128 by
        # default, >8x the measured B80 mean); copy just that prefix into a
        # compact graph-stable tensor and keep the original LL-shaped output
        # for combine.
        if packed_dispatch:
            x_fp4_grouped = packed_x[:, :compute_capacity, :].contiguous()
            # DeepEP exposes logical [E, LL_M, K/64] scales with mn-major
            # stride. Compact the physical [E, K/64, M] storage, then restore
            # the same logical view with the smaller M stride.
            x_scale_grouped = (
                packed_scale.permute(0, 2, 1)[:, :, :compute_capacity]
                .contiguous()
                .permute(0, 2, 1)
            )
        else:
            compact_x = expert_x[:, :compute_capacity, :].contiguous()
            flat_x = compact_x.view(total, D)
            x_fp4, x_scale = downcast_to_mxfp4(flat_x)
            x_fp4_grouped = x_fp4.view(E, compute_capacity, -1)
            x_scale_grouped = x_scale.as_strided(
                (E, compute_capacity, x_scale.size(1)),
                (compute_capacity, 1, total),
            )
            x_scale_grouped = (
                x_scale_grouped.permute(0, 2, 1).contiguous().permute(0, 2, 1)
            )
        gate_up_grouped = torch.empty(
            (E, compute_capacity, 2 * inter),
            dtype=torch.bfloat16,
            device=device,
        )
        expected_m = max(
            1,
            (
                cfg.max_tokens_per_rank
                * cfg.ep_size
                * cfg.n_activated_experts
                + cfg.n_routed_experts
                - 1
            )
            // cfg.n_routed_experts,
        )
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (x_fp4_grouped, x_scale_grouped),
            (self._ppu_w13, self._ppu_s13),
            None,
            gate_up_grouped,
            safe_counts,
            expected_m,
        )
        swiglu_limit = cfg.swiglu_limit if cfg.swiglu_limit > 0 else None
        use_masked_silu = _MASKED_SILU
        if use_masked_silu and inter % _MASKED_SILU_BLOCK_N:
            if not _MASKED_SILU_SKIP_LOGGED[0]:
                _MASKED_SILU_SKIP_LOGGED[0] = True
                logging.warning(
                    "[MASKED_SILU] inter=%d is not a multiple of %d, so the "
                    "scale view's expert stride would differ from the compact "
                    "path's; keeping the unmasked kernel",
                    inter,
                    _MASKED_SILU_BLOCK_N,
                )
            use_masked_silu = False
        if use_masked_silu:
            if not _MASKED_SILU_LOGGED[0]:
                _MASKED_SILU_LOGGED[0] = True
                logging.info(
                    "[MASKED_SILU] on: E=%d capacity=%d inter=%d",
                    E,
                    compute_capacity,
                    inter,
                )
            # Only the rows below ``safe_counts[e]`` are written; the rest keep
            # whatever the allocator left there.  That is already true of
            # ``gate_up_grouped`` above, and the masked GEMM below does not read
            # past an expert's count.  ``compute_capacity`` as the row hint means
            # "do not cap blocks per expert" -- it only trades occupancy.
            #
            # The scale comes back as the mn-major [E, M, S] view the GEMM wants,
            # because [E, S, M] is this kernel's native layout.  The unmasked
            # branch below has to pay a contiguous transpose to get there.
            hidden_fp4_grouped, hidden_scale_grouped = (
                rtp_llm_ops.ppu_silu_and_mul_masked_post_quant_mxfp4(
                    gate_up_grouped,
                    safe_counts,
                    swiglu_limit,
                    compute_capacity,
                )
            )
        else:
            gate_up = gate_up_grouped.view(total, 2 * inter)
            # Match the SGLang PPU runner: consume the grouped GEMM's BF16
            # gate/up output directly and fuse SwiGLU with MXFP4 packing.  The
            # old chain materialized two FP32 halves, an FP32 activation, and a
            # BF16 tensor before launching a separate quantizer on every routed
            # layer.
            hidden_fp4, hidden_scale = (
                rtp_llm_ops.ppu_silu_and_mul_post_quant_mxfp4(
                    gate_up, swiglu_limit
                )
            )
            hidden_fp4_grouped = hidden_fp4.view(E, compute_capacity, -1)
            hidden_scale_grouped = hidden_scale.as_strided(
                (E, compute_capacity, hidden_scale.size(1)),
                (compute_capacity, 1, total),
            )
            hidden_scale_grouped = (
                hidden_scale_grouped.permute(0, 2, 1)
                .contiguous()
                .permute(0, 2, 1)
            )
        if (
            out is not None
            and out.dtype == torch.bfloat16
            and tuple(out.shape) == (E, compute_capacity, D)
            and out.is_contiguous()
        ):
            compact_down = out
        else:
            compact_down = torch.empty(
                (E, compute_capacity, D),
                dtype=torch.bfloat16,
                device=device,
            )
        deep_gemm.m_grouped_gemm_fp4_fp4_bf16_nt_masked(
            (hidden_fp4_grouped, hidden_scale_grouped),
            (self._ppu_w2, self._ppu_s2),
            None,
            compact_down,
            safe_counts,
            expected_m,
        )
        return compact_down

    def _forward_ppu_grouped_fp4_low_latency(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
        wrapper,
    ) -> torch.Tensor:
        """DeepEP LL packed dispatch → grouped MXFP4 experts → combine."""
        dispatch_args = {
            "x": x.contiguous(),
            "topk_idx": indices.to(torch.int64).contiguous(),
            "num_max_dispatch_tokens_per_rank": wrapper.ll_num_max_token_per_rank,
            "num_experts": self.cfg.n_routed_experts,
            "use_fp8": False,
            "use_mxfp4": True,
            "mxfp4_scale_row_major": False,
            "quant_size": 32,
            "async_finish": False,
            "return_recv_hook": False,
        }
        expert_x, expert_num_tokens, handle, _, _ = (
            wrapper.buffer.low_latency_dispatch(**dispatch_args)
        )
        if not isinstance(expert_x, tuple) or len(expert_x) != 2:
            raise RuntimeError("DeepEP LL MXFP4 dispatch must return (data, scale)")
        # Claim the combine buffer BEFORE the experts run so the down-projection
        # GEMM can write into it directly -- DeepEP documents this buffer as one
        # the caller fills itself, precisely to avoid a staging copy.  Only under
        # no-compact, where the compute shape equals the slot shape; and only
        # when its bytes are disjoint from the dispatch payload the first GEMM
        # still reads, since the two can share one RDMA arena.
        expert_y_pre = None
        if _LL_NO_COMPACT:
            candidate = wrapper.buffer.get_next_low_latency_combine_buffer(handle)
            payload, scale = expert_x
            if _spans_overlap(candidate, payload) or _spans_overlap(candidate, scale):
                logging.warning(
                    "[LL_NO_COMPACT] combine buffer overlaps the dispatch payload; "
                    "falling back to the compacting copy"
                )
            else:
                expert_y_pre = candidate
        compact_expert_y = self._compute_ppu_grouped_fp4_packed(
            expert_x, expert_num_tokens, out=expert_y_pre
        )
        # The LL RDMA buffer already owns the required full slot geometry.  Do
        # not allocate another [E, ll_capacity, D] tensor (768 MiB for V4 at
        # gamma3); copy only the compact valid prefix into the next combine
        # buffer and let the handle's receive counts delimit the rows consumed.
        expert_y = (
            expert_y_pre
            if expert_y_pre is not None
            else wrapper.buffer.get_next_low_latency_combine_buffer(handle)
        )
        # Equal pointers mean the GEMM already landed in place; copying a tensor
        # onto itself is not merely wasteful here, it is an aliased overlap.
        # Whether to copy is decided here; how to copy is the flag below.
        if compact_expert_y.data_ptr() != expert_y.data_ptr():
            if self._ppu_deepep_compact_copy_2d:
                from ._compact_prefix_copy import compact_to_strided_prefix

                compact_to_strided_prefix(compact_expert_y, expert_y)
            else:
                expert_y[:, : compact_expert_y.size(1), :].copy_(compact_expert_y)
        combine_args = {
            "x": expert_y,
            "topk_idx": dispatch_args["topk_idx"],
            "topk_weights": weights.contiguous(),
            "handle": handle,
            "zero_copy": True,
            "async_finish": False,
            "return_recv_hook": False,
        }
        combined_x, _, _ = wrapper.buffer.low_latency_combine(**combine_args)
        return combined_x.float()

    @staticmethod
    def _pad_topk_for_deepep(
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad ``(indices, weights)`` to the nearest supported topk width.

        See ``_DEEPEP_SUPPORTED_TOPK`` docstring above.
        """
        n_act = indices.size(-1)
        if n_act in _DEEPEP_SUPPORTED_TOPK:
            return indices, weights
        pad_to = next((k for k in _DEEPEP_SUPPORTED_TOPK if k > n_act), None)
        if pad_to is None:
            raise RuntimeError(
                f"n_activated_experts={n_act} exceeds largest DeepEP-supported "
                f"topk ({max(_DEEPEP_SUPPORTED_TOPK)})"
            )
        N = indices.size(0)
        pad_n = pad_to - n_act
        pad_idx = torch.full((N, pad_n), -1, dtype=indices.dtype, device=indices.device)
        pad_w = torch.zeros((N, pad_n), dtype=weights.dtype, device=weights.device)
        return (
            torch.cat([indices, pad_idx], dim=-1),
            torch.cat([weights, pad_w], dim=-1),
        )

    def forward(
        self,
        x: torch.Tensor,        # [N, D] local rank's tokens (BF16)
        weights: torch.Tensor,  # [N, k] fp32
        indices: torch.Tensor,  # [N, k] int64 global expert IDs
    ) -> torch.Tensor:
        """DP+EP path: DeepEP normal dispatch → local per-expert compute
        → DeepEP combine. Requires ``init_deepep_wrapper`` to have been
        called by the engine (``backend_manager.py``).
        """
        from rtp_llm.models_py.distributed.deepep_wrapper import (
            DeepEPMode,
            DeepEPWrapper,
        )

        if DeepEPWrapper._instance is None:
            raise RuntimeError(
                "DeepEPWrapper not initialised; ep_size>1 requires "
                "init_deepep_wrapper() at engine startup (enable via "
                "--use_deepep_moe 1)."
            )
        wrapper = DeepEPWrapper._instance
        buf = wrapper.buffer
        cfg = self.cfg

        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        graph_warmup = cuda_graph_warmup_forward_enabled()
        if graph_warmup:
            sync_cuda_graph_warmup_ranks("deepep_before_dispatch", x.device)

        # Pad topk to nearest supported value (V4's 6 → 8).
        indices_p, weights_p = self._pad_topk_for_deepep(indices, weights)

        if wrapper.mode == DeepEPMode.LOW_LATENCY:
            if not getattr(self, "_ppu_grouped_fp4", False):
                raise RuntimeError(
                    "DSV4 DeepEP low-latency requires the M890P grouped-FP4 "
                    "executor (set DSV4_PPU_GROUPED_FP4=1)"
                )
            y_combined = self._forward_ppu_grouped_fp4_low_latency(
                x, weights_p, indices_p, wrapper
            )
            if graph_warmup:
                sync_cuda_graph_warmup_ranks("deepep_after_combine", x.device)
            return y_combined
        if wrapper.mode != DeepEPMode.NORMAL:
            raise RuntimeError(f"unsupported DeepEP mode for DSV4: {wrapper.mode}")

        # 1. Dispatch layout. indices cast to int64 already.
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            _,
        ) = buf.get_dispatch_layout(indices_p, cfg.n_routed_experts)

        # 2. Dispatch the BF16 tokens + topk scaffolding.
        (
            recv_x,
            recv_topk_idx,
            recv_topk_weights,
            num_recv_tokens_per_expert_list,
            handle,
            _,
        ) = buf.dispatch(
            x,
            None,
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            is_token_in_rank,
            num_tokens_per_expert,
            indices_p,
            weights_p,
            expert_alignment=1,
            # DeepEP's normal dispatch otherwise synchronizes the dynamic
            # receive count through the CPU. Fixed worst-case capacity keeps
            # warmup, capture, and replay shapes identical and uses the
            # graph-safe no-CPU-sync kernel path. DP ranks can have different
            # local batches (real batch 2 beside fake batch 1), so capacity
            # must use the common startup budget rather than this rank's x.
            # Otherwise a rank captured at batch 1 reserves 8 rows although
            # the EP group can send it 9, which deadlocks ACCL-EP replay.
            num_worst_tokens=(int(cfg.max_tokens_per_rank) * cfg.ep_size)
            if (graph_warmup or capturing)
            else 0,
        )

        # 3. Local per-expert compute. ACCL-EP's dispatch returns
        # ``recv_topk_idx`` in the LOCAL index space ``[0, n_local_experts)``
        # (with -1 for tokens not destined for any local expert), NOT the
        # global expert id. Shift to global so the per-expert loop in
        # ``LocalLoopStrategy`` indexes ``self._local.experts[global_i]``
        # correctly. Also force int64 and contiguous — the ACCL tensor
        # sometimes comes back with a non-standard dtype that triggers
        # ``torch.where(idx == i)`` with "unknown parameter type".
        M = recv_x.size(0)
        if M > 0 and getattr(self, "_ppu_grouped_fp4", False):
            configured_capacity = int(
                os.environ.get("DSV4_PPU_GROUPED_FP4_CAPACITY", "128")
            )
            grouped_capacity = _select_ppu_grouped_fp4_capacity(
                configured_capacity,
                num_recv_tokens_per_expert_list,
                cfg.n_local_experts,
                fixed_shape=(graph_warmup or capturing),
            )
            y_local = self._forward_ppu_grouped_fp4(
                recv_x.contiguous(),
                recv_topk_weights.contiguous(),
                recv_topk_idx.contiguous(),
                grouped_capacity,
            )
        elif M > 0:
            global_topk_idx = recv_topk_idx.to(torch.int64).contiguous()
            # Shift local→global; keep -1 as -1 (won't match any expert id).
            global_topk_idx = torch.where(
                global_topk_idx == -1,
                global_topk_idx,
                global_topk_idx + cfg.local_expert_start,
            )
            # _local.forward() allocates its own y_local buffer (its
            # _local_y_buf), runs the [local_start, local_end) loop, and
            # returns the fp32 accumulator. We pass through.
            y_local = self._local._forward_into_buf(
                recv_x.contiguous(),
                recv_topk_weights.contiguous(),
                global_topk_idx,
                local_start=cfg.local_expert_start,
                local_end=cfg.local_expert_end,
            )
        else:
            # M == 0: no recv tokens this rank — produce a fresh empty
            # fp32 accumulator so combine still has a valid tensor to send.
            y_local = torch.zeros(M, cfg.dim, dtype=torch.float32, device=recv_x.device)

        # 4. Combine back to source ranks. combine expects the tensor
        # dtype to match x (BF16) — cast the fp32 accumulator.
        y_combined, _, _ = buf.combine(
            y_local.to(x.dtype),
            handle,
        )
        if graph_warmup:
            sync_cuda_graph_warmup_ranks("deepep_after_combine", x.device)
        return y_combined.float()
