"""Cross-rank alignment of DeepEP low-latency dispatch call counts.

``low_latency_dispatch`` / ``low_latency_combine`` are EP collectives: every
rank must issue the same number of calls in the same order, or a rank ends up
spin-waiting on the device for peer data that never arrives (on PPU the symptom
is a permanently stuck ``hggc::_HGcontext::streamSynchronize``, i.e. a fence
that never signals).

Two prefill chunk loops derive their iteration count from the *local* token
count and therefore break that invariant as soon as DP > 1:

* ``MoE._forward_chunked``            — chunks of ``max_tokens_per_rank``
* ``DeepEPLowLatencyStrategy.forward`` — chunks of ``ll_num_max_token_per_rank``

With DP=8 a rank holding a 1005-token prefill issues 2 dispatches per layer
while the ranks holding the 1-token fake prefill issue 1, and the extra
dispatch hangs the whole DP group.

This module keeps both loops as they are and equalises their *count*: one
all-reduce(MAX) per prefill forward yields the global call count, and the ranks
below it issue no-op single-token dispatches to catch up.

Two ordering rules are load-bearing:

1. The vote is unconditional for the whole prefill forward. Deciding per rank
   "do I need to sync?" (e.g. only when ``tokens > cap``) is itself a cross-rank
   divergence and deadlocks the vote — same lesson as the prefill vote in
   ``NormalEngine::mayAddFakeStream``.
2. The vote is issued *before* the first layer's real chunks. Voting afterwards
   would let a rank sit inside its second DeepEP dispatch while its peers wait
   in the all-reduce for that very rank.

Decode never reaches the alignment path: ``MoE._should_chunk`` keeps decode
batches under the chunk threshold, CUDA-graph replay does not execute this
Python at all, and graph eligibility is per-rank (a rank with an uncaptured
batch size falls back to eager) — so a vote on the decode path would not be
rank-uniform.

A third gate is equally load-bearing: **the vote requires an MTP serve**
(``set_mtp_enabled(True)`` from ``DeepSeekV4MtpModel.__init__``). Only the MTP
path of ``NormalEngine::mayAddFakeStream`` injects a 1-token fake *prefill* on
ranks without a real context stream, so only there do all DP ranks execute a
Python prefill forward every cycle and the vote is guaranteed a matching
partner on every rank. A non-MTP serve injects only a fake *decode* stream
(graph replay — no Python, no vote), so a voting rank would spin in the
all-reduce forever. Verified on hardware: non-MTP + ALIGN=1 hangs even a
1-request paris probe, while ALIGN=0 serves paris and len=1000 fine (the
asymmetric dispatch itself is tolerated there; no fake prefill means no
count-divergence deadlock either).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch

_ALIGN_ENV = "DSV4_LL_CHUNK_ALIGN"
_DEBUG_ENV = "DSV4_LL_CHUNK_DEBUG"

_LL_STRATEGY_NAME = "deepep_low_latency"

# Cumulative number of real + padding LL dispatch rounds issued by this process.
# The strategy bumps it; the layer plan uses deltas to pad by what actually
# happened instead of trusting the analytic mirror of the chunk loops.
_LL_CALLS = 0

_FORMULA_MISMATCH_LOGGED = False

# True once a DeepSeekV4MtpModel (draft) has been constructed in this process —
# i.e. we are on an MTP serve. Rank-uniform: every rank of an MTP serve builds
# the draft model, no rank of a non-MTP serve does. See module docstring for
# why the vote is only safe under MTP's fake-prefill injection.
_MTP_ENABLED = False


def set_mtp_enabled(enabled: bool) -> None:
    """Mark this process as an MTP serve (called from the draft model ctor)."""
    global _MTP_ENABLED
    _MTP_ENABLED = bool(enabled)


def mtp_enabled() -> bool:
    return _MTP_ENABLED


def alignment_enabled() -> bool:
    """Chunk-count alignment is on by default; ``0`` restores the old behaviour."""
    return os.environ.get(_ALIGN_ENV, "1") != "0"


def debug_enabled() -> bool:
    return os.environ.get(_DEBUG_ENV, "0") == "1"


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class _ForwardState:
    """Per-model-forward state, reset by ``begin_forward``."""

    generation: int = 0
    is_prefill: bool = False
    global_calls: Optional[int] = None
    logged_generation: int = -1


@dataclass
class _LayerPlan:
    strategy: object
    local_calls: int
    global_calls: int
    calls_before: int


_STATE = _ForwardState()


def begin_forward(is_prefill: bool) -> None:
    """Open a new model forward. Called from ``DeepSeekV4Model.forward``.

    ``is_prefill`` must be true only for the always-eager context branch: the
    verify/decode branches may run from a captured CUDA graph on some ranks and
    eagerly on others, so they are not a safe place for a collective.
    """
    _STATE.generation += 1
    _STATE.is_prefill = bool(is_prefill)
    _STATE.global_calls = None


def note_ll_call() -> None:
    global _LL_CALLS
    _LL_CALLS += 1


def local_ll_calls(tokens: int, cap: int, chunk_tokens: int, chunked: bool) -> int:
    """Number of LL dispatch rounds one layer performs for ``tokens`` tokens.

    Mirrors ``MoE._forward_chunked`` (outer loop) composed with
    ``DeepEPLowLatencyStrategy.forward`` (inner loop). The composition is not
    ``ceil(tokens / cap)``: each MoE chunk restarts the inner loop, so its tail
    chunk is charged a full round.
    """
    if tokens <= 0 or cap <= 0:
        return 0
    if chunked and chunk_tokens > 0 and tokens > chunk_tokens:
        calls = 0
        for start in range(0, tokens, chunk_tokens):
            calls += _ceil_div(min(chunk_tokens, tokens - start), cap)
        return calls
    return _ceil_div(tokens, cap)


def _ll_cap() -> int:
    try:
        from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper

        wrapper = DeepEPWrapper._instance
        if wrapper is None:
            return 0
        return int(wrapper.ll_num_max_token_per_rank)
    except Exception:
        return 0


def _vote_max(local_calls: int, device: torch.device) -> int:
    """All-reduce MAX over the EP span (= DP_AND_TP / WORLD group).

    ``collective_torch.all_reduce`` is SUM-only, so go through
    ``torch.distributed`` with the group that module already registered.
    """
    from rtp_llm.models_py.distributed import collective_torch

    flag = torch.tensor([int(local_calls)], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(
        flag,
        op=torch.distributed.ReduceOp.MAX,
        group=collective_torch._get_group(collective_torch.Group.DP_AND_TP),
    )
    return int(flag.item())


def _maybe_log(tokens: int, cap: int, local_calls: int, global_calls: Optional[int]) -> None:
    if not debug_enabled() or _STATE.logged_generation == _STATE.generation:
        return
    _STATE.logged_generation = _STATE.generation
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else -1
    )
    logging.info(
        "[ll-chunk] rank=%d forward=%d tokens=%d cap=%d local_calls=%d global_calls=%s",
        rank,
        _STATE.generation,
        tokens,
        cap,
        local_calls,
        "off" if global_calls is None else str(global_calls),
    )


def begin_layer(
    layer, tokens: int, chunked: bool, device: torch.device
) -> Optional[_LayerPlan]:
    """Vote (once per forward) and return the plan for this MoE layer.

    Returns ``None`` when alignment does not apply, in which case
    ``finish_layer`` is a no-op. Every gate below is a rank-uniform fact
    (env var, strategy identity, prefill-ness, MTP-ness, world size) so all
    ranks either all vote or all skip.
    """
    if not _STATE.is_prefill:
        return None
    if not _MTP_ENABLED:
        # Non-MTP serves have no fake-prefill injection (mayAddFakeStream only
        # adds a fake *decode* stream there), so the ranks without a real
        # context stream never reach this vote — issuing it would deadlock.
        return None
    strategy = getattr(layer, "_strategy", None)
    if getattr(strategy, "name", None) != _LL_STRATEGY_NAME:
        return None
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    if torch.distributed.get_world_size() <= 1:
        return None
    cap = _ll_cap()
    if cap <= 0:
        return None

    local_calls = local_ll_calls(
        int(tokens), cap, int(layer.max_tokens_per_rank), bool(chunked)
    )
    if not alignment_enabled():
        # Phase-0 / bisection mode: still surface the per-rank call count so the
        # divergence is visible in the log, but do not touch the collectives.
        _maybe_log(int(tokens), cap, local_calls, None)
        return None

    global_calls = _STATE.global_calls
    if global_calls is None:
        global_calls = _vote_max(local_calls, device)
        _STATE.global_calls = global_calls
    _maybe_log(int(tokens), cap, local_calls, global_calls)

    return _LayerPlan(
        strategy=strategy,
        local_calls=local_calls,
        global_calls=global_calls,
        calls_before=_LL_CALLS,
    )


def finish_layer(plan: Optional[_LayerPlan]) -> None:
    """Issue the no-op dispatch rounds this rank owes the group."""
    global _FORMULA_MISMATCH_LOGGED
    if plan is None:
        return
    actual = _LL_CALLS - plan.calls_before
    if actual != plan.local_calls and not _FORMULA_MISMATCH_LOGGED:
        _FORMULA_MISMATCH_LOGGED = True
        logging.warning(
            "[ll-chunk] predicted %d LL calls but observed %d; padding follows the "
            "observed count (chunk loops changed?)",
            plan.local_calls,
            actual,
        )
    pad = plan.global_calls - actual
    if pad > 0:
        plan.strategy.pad_ll_calls(pad)
