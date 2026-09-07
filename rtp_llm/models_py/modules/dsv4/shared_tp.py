"""Prefill-only TP4 shared-expert layout and byte-preserving FP8 slicing."""

from __future__ import annotations

import logging
from typing import NamedTuple

from rtp_llm.models_py.modules.dsv4.runtime_config import get_switch, parse_bool

SHARED_TP4_SWITCH = "DSV4_PREFILL_SHARED_TP4"


class SharedTpLayout(NamedTuple):
    size: int = 1
    rank: int = 0


def resolve_prefill_shared_tp4(parallelism_config, *, enabled=None) -> SharedTpLayout:
    """Resolve from framework topology, never from a role environment variable.

    Decode always keeps replicated shared weights, including when a common
    launcher passes the prefill switch to both roles. The MoE constructor also
    checks the actual PPU strategy and linear implementation before execution.
    """
    if enabled is None:
        enabled = get_switch(SHARED_TP4_SWITCH, False, parse_bool)
    if not enabled:
        return SharedTpLayout()
    role = getattr(parallelism_config, "role_type", None)
    role_name = str(getattr(role, "name", role)).upper().split(".")[-1]
    if role_name == "DECODE":
        logging.info("DSV4_SHARED_TP role=DECODE size=1 rank=0")
        return SharedTpLayout()
    if role_name != "PREFILL":
        raise ValueError(f"{SHARED_TP4_SWITCH}=1 requires an explicit PREFILL/DECODE role")
    tp = int(parallelism_config.get_attn_tp_size())
    rank = int(parallelism_config.get_attn_tp_rank())
    cp = parallelism_config.prefill_cp_config
    cp_size = int(getattr(cp, "prefill_cp_size", 1))
    valid = (
        tp == 4
        and int(parallelism_config.tp_size) == 4
        and int(parallelism_config.ep_size) == 1
        and int(parallelism_config.dp_size) == 1
        and int(parallelism_config.world_size) == 4
        and cp_size == 1
        and not cp.is_enabled()
        and 0 <= rank < 4
    )
    if not valid:
        raise ValueError(f"{SHARED_TP4_SWITCH}=1 only supports PREFILL TP4/CP1/EP1/DP1")
    logging.info("DSV4_SHARED_TP role=PREFILL size=4 rank=%d", rank)
    return SharedTpLayout(4, rank)


def shared_fp8_tp_slice(
    t, *, projection: str, tp_size: int, tp_rank: int, is_scale: bool = False, **_
):
    """Slice raw [N,K] E4M3 or raw [N/128,K/128] UE8M0; no requantization.

    W13 is [gate; up]: shard each half, then rejoin. W2 shards K.
    Extra kwargs permit use as an AtomicWeight split function without reusing
    the unrelated ffn_tp_size/rank passed by the generic loader.
    """
    import torch

    from rtp_llm.utils.model_weight import concat_0, sp_0, sp_neg1

    if tp_size not in (1, 4) or not 0 <= tp_rank < tp_size:
        raise ValueError("shared FP8 split requires TP1 or TP4 and a valid rank")
    if projection not in ("w13", "w2") or t.dim() != 2:
        raise ValueError("shared FP8 split requires a 2D w13/w2 tensor")
    expected_dtype = torch.float8_e8m0fnu if is_scale else torch.float8_e4m3fn
    if t.dtype != expected_dtype:
        raise ValueError(f"shared raw {'scale' if is_scale else 'weight'} must be {expected_dtype}")
    alignment = 1 if is_scale else 128
    if projection == "w13":
        if t.shape[0] % (2 * tp_size * alignment) or t.shape[1] % alignment:
            raise ValueError("shared W13 halves must divide TP on 128-element block boundaries")
        if tp_size == 1:
            return t
        gate, up = t.chunk(2, dim=0)
        return concat_0([sp_0(gate, tp_size, tp_rank), sp_0(up, tp_size, tp_rank)])
    if t.shape[1] % (tp_size * alignment) or t.shape[0] % alignment:
        raise ValueError("shared W2 K must divide TP on 128-element block boundaries")
    if tp_size == 1:
        return t
    return sp_neg1(t, tp_size, tp_rank).contiguous()
