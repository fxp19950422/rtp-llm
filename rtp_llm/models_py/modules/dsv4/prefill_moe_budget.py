"""Independent, default-off prefill MoE workspace/chunk budget."""

from __future__ import annotations

import logging

from rtp_llm.models_py.modules.dsv4.runtime_config import get_switch

PREFILL_MOE_MAX_TOKENS_SWITCH = "DSV4_PREFILL_MOE_MAX_TOKENS"


def _parse_prefill_moe_max_tokens(raw, _default):
    value = int(raw.strip())
    if value not in (0, 16384):
        raise ValueError("expected 0 (legacy budget) or 16384")
    return value


def resolve_prefill_moe_max_tokens(
    legacy_budget: int,
    parallelism_config,
    *,
    is_decode_role: bool,
    chunked_moe: bool,
    configured=None,
) -> int:
    """Override only MoE allocation/chunking after the runtime role is known.

    The default preserves the legacy context/chunk-derived budget. Decode
    ignores an inherited prefill setting and keeps its batch/verify budget.
    This helper has no dependency on the optional shared-TP4 implementation:
    it can run with either replicated or sharded shared experts.
    """
    if configured is None:
        configured = get_switch(
            PREFILL_MOE_MAX_TOKENS_SWITCH, 0, _parse_prefill_moe_max_tokens
        )
    if configured not in (0, 16384):
        raise ValueError(f"{PREFILL_MOE_MAX_TOKENS_SWITCH} only supports 0 or 16384")
    if configured == 0 or is_decode_role:
        return int(legacy_budget)

    role = getattr(parallelism_config, "role_type", None)
    role_name = str(getattr(role, "name", role)).upper().split(".")[-1]
    if role_name != "PREFILL":
        raise ValueError(
            f"{PREFILL_MOE_MAX_TOKENS_SWITCH}=16384 requires an explicit PREFILL role"
        )
    cp = parallelism_config.prefill_cp_config
    rank = int(parallelism_config.get_attn_tp_rank())
    valid = (
        int(parallelism_config.get_attn_tp_size()) == 4
        and int(parallelism_config.tp_size) == 4
        and int(parallelism_config.ep_size) == 1
        and int(parallelism_config.dp_size) == 1
        and int(parallelism_config.world_size) == 4
        and int(getattr(cp, "prefill_cp_size", 1)) == 1
        and not cp.is_enabled()
        and 0 <= rank < 4
    )
    if not valid:
        raise ValueError(
            f"{PREFILL_MOE_MAX_TOKENS_SWITCH}=16384 only supports PREFILL TP4/CP1/EP1/DP1"
        )
    if not chunked_moe:
        raise ValueError(
            f"{PREFILL_MOE_MAX_TOKENS_SWITCH}=16384 requires enabled MoE chunking"
        )
    logging.info(
        "[DSV4 MoE] independent prefill token budget=%d legacy_budget=%d "
        "rank=%d; context/KV and other chunk budgets unchanged",
        configured,
        legacy_budget,
        rank,
    )
    return int(configured)
