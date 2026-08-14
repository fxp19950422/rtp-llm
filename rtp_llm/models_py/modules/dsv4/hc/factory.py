"""Factory for DeepSeek-V4 Hyper-Connection implementations."""

from __future__ import annotations

import os

import torch

from rtp_llm.models_py.modules.dsv4.hc.base import HCHeadBase, HCMode, HCUnitBase
from rtp_llm.models_py.modules.dsv4.hc.utils import maybe_squeeze_hc_1d


def _tilelang_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("tilelang") is not None


def _mode_from_env() -> HCMode:
    raw = os.environ.get("DSV4_HC_IMPL")
    if raw is None:
        # Default: TileLang when the package is importable, else the PyTorch
        # reference fallback. Backends without tilelang (e.g. PPU) then work
        # out of the box instead of raising inside the TileLang kernel; an
        # explicit DSV4_HC_IMPL still overrides this auto-detection.
        return HCMode.TILELANG if _tilelang_available() else HCMode.FALLBACK
    try:
        return HCMode(raw.lower())
    except ValueError as exc:
        allowed = ", ".join(m.value for m in HCMode)
        raise ValueError(f"invalid DSV4_HC_IMPL={raw!r}; expected one of: {allowed}") from exc


def build_hc_unit(
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    dim: int,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    layer_id: int = -1,
    name: str = "",
) -> HCUnitBase:
    mode = _mode_from_env()
    scale = maybe_squeeze_hc_1d(scale)
    if mode is HCMode.TILELANG:
        from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit

        return TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            layer_id=layer_id,
            name=name,
        )
    from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import FallbackHCUnit

    return FallbackHCUnit(
        fn,
        base,
        scale,
        dim=dim,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        layer_id=layer_id,
        name=name,
    )


def build_hc_head(
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    dim: int,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
) -> HCHeadBase:
    mode = _mode_from_env()
    scale = maybe_squeeze_hc_1d(scale)
    if mode is HCMode.TILELANG:
        from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCHead

        return TileLangHCHead(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc_mult,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )
    from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import FallbackHCHead

    return FallbackHCHead(
        fn,
        base,
        scale,
        dim=dim,
        hc_mult=hc_mult,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
    )
