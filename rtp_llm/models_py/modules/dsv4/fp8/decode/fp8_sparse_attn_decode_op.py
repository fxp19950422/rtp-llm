"""V4 decode-arm FP8 sparse attention op.

Wraps FlashMLA's ``flash_mla_with_kvcache(is_fp8_kvcache=True)`` for
single- and dual-pool decode. The kernel reads the packed
``fp8_model1_mla`` KV cache directly (no dequant on the read path) and
outputs bf16 attention output.

Dual-pool support uses FlashMLA's ``extra_k_cache`` +
``extra_indices_in_kvcache`` parameters to attend over a second FP8 pool
(CSA / HCA compressor pool) in a single kernel call, with in-kernel
softmax merging across both pools (mirrors vLLM
``deepseek_v4_attention.py:849-865``). Replaces the legacy "dequant both
pools -> BF16 cat -> TileLang sparse_attn" path which was
bandwidth-bound on the dequant kernels.

FlashMLA wheel is required (CUDA >= 12.9). The op asserts wheel
availability at forward — there is no slow Python reference fallback
because all dev/CI/prod boxes carry flash_mla.
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any, Optional

import torch

# The GPU flash_mla wheel exposes attn_sink / topk_length / extra_* kwargs on
# ``flash_mla_with_kvcache`` and applies dual-pool softmax + attn-sink inside the
# kernel. The PPU wheel ships only the reduced sparse signature
# ``(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata,
# num_splits, softmax_scale, causal, is_fp8_kvcache, indices)`` and returns a
# 2-based (log2) softmax_lse -- so on PPU we mask topk_length via indices, and
# reproduce the sink + dual-pool merge in Python from the returned lse.
_FLASH_MLA_KVCACHE_ATTN_SINK: Optional[bool] = None


def _flash_mla_with_kvcache_supports_attn_sink() -> bool:
    """True for the GPU wheel (native attn_sink/dual-pool); False for the PPU
    wheel (reduced signature -> post-hoc sink + Python dual-pool merge)."""
    global _FLASH_MLA_KVCACHE_ATTN_SINK
    if _FLASH_MLA_KVCACHE_ATTN_SINK is None:
        supported = True
        try:
            from flash_mla import (  # type: ignore[import-not-found]
                flash_mla_with_kvcache as _fmk,
            )

            params = inspect.signature(_fmk).parameters
            if params:
                has_varkw = any(
                    pm.kind == inspect.Parameter.VAR_KEYWORD for pm in params.values()
                )
                supported = has_varkw or ("attn_sink" in params)
        except (ImportError, ValueError, TypeError):
            supported = True
        _FLASH_MLA_KVCACHE_ATTN_SINK = supported
    return _FLASH_MLA_KVCACHE_ATTN_SINK


_FLASH_MLA_AVAILABLE = False
try:
    from flash_mla import flash_mla_with_kvcache  # type: ignore[import-not-found] # noqa: F401
    from flash_mla import get_mla_metadata  # type: ignore[import-not-found] # noqa: F401

    _cuda = torch.version.cuda
    _ge_129 = bool(_cuda) and tuple(map(int, _cuda.split(".")[:2])) >= (12, 9)
    # PPU wheel reports CUDA 12.6 but ships the sparse FP8 decode kernel; detect
    # it via its reduced (no-attn_sink) signature so decode is enabled on PPU.
    if _ge_129 or not _flash_mla_with_kvcache_supports_attn_sink():
        _FLASH_MLA_AVAILABLE = True
except (ImportError, AttributeError, ValueError) as e:
    logging.warning("[dsv4-fp8] flash_mla wheel unavailable (%s)", e)


class SparseAttnV4DecodeFp8Op:
    """FP8 sparse attention decode op (single- or dual-pool).

    Args (forward):
      q          : ``[B, q_len, n_heads, head_dim]`` bf16
      kv_cache   : ``[num_blocks, block_size, 584]`` uint8 packed FP8
        primary pool (SWA in dual-pool mode).
      attn_sink  : ``[n_heads]`` fp32 — per-head learned sink
      topk_idxs  : ``[B, q_len, topk]`` int32 — per-request global slot
        ids into the primary pool.
      cache_seqlens : unused in sparse FP8 decode; accepted to keep the
        attention helper call sites uniform.
      block_table   : unused in sparse FP8 decode; FlashMLA consumes global
        slot ids from ``topk_idxs`` directly.
      topk_length        : optional ``[B]`` int32 — per-request leftmost
        valid length on ``topk_idxs``.
      extra_k_cache      : optional secondary FP8 pool (CMP). Triggers
        FlashMLA's dual-pool path.
      extra_topk_idxs    : optional ``[B, q_len, extra_topk]`` int32 —
        global slot ids into ``extra_k_cache``.
      extra_topk_length  : optional ``[B]`` int32 — per-request leftmost
        valid length on ``extra_topk_idxs``.
    """

    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        softmax_scale: float,
    ) -> None:
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.softmax_scale = softmax_scale

    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sched_meta: Any,
        cache_seqlens: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        topk_length: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single- or dual-pool sparse attention.

        ``sched_meta`` is the FlashMLA planner output (``get_mla_metadata``
        return) — owned by :class:`DSv4DecodeAttnMetadataFP8` and fetched via
        :func:`~decode_attn_metadata.get_or_build_sched_meta`. The op is a
        pure dispatcher; it does NOT cache sched_meta itself (was the
        iter2 anti-pattern that accumulated per-instance state across
        decode steps).

        Dual-pool: pass ``extra_k_cache`` (e.g. CMP pool 3D
        ``[num_blocks, block_size, 584]`` uint8) + ``extra_topk_idxs``
        (3D ``[B, q_len, extra_topk]`` int32 global slot ids) to attend
        over a second FP8 KV pool in a single FlashMLA invocation. The
        kernel merges softmax across both pools natively.
        """
        assert _FLASH_MLA_AVAILABLE, (
            "flash_mla wheel is required for FP8 sparse decode "
            "(install rtp_llm with cuda12_9 / cuda13 config)"
        )
        return self._forward_flash_mla(
            q,
            kv_cache,
            attn_sink,
            topk_idxs,
            sched_meta,
            cache_seqlens,
            block_table,
            topk_length,
            extra_k_cache,
            extra_topk_idxs,
            extra_topk_length,
        )

    def _forward_flash_mla(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sched_meta: Any,
        cache_seqlens: Optional[torch.Tensor],
        block_table: Optional[torch.Tensor],
        topk_length: Optional[torch.Tensor] = None,
        extra_k_cache: Optional[torch.Tensor] = None,
        extra_topk_idxs: Optional[torch.Tensor] = None,
        extra_topk_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from flash_mla import flash_mla_with_kvcache  # type: ignore[import-not-found]

        B, q_len, H, D = q.shape
        # FlashMLA expects 4D q ``(batch_size, seq_len_q, num_heads_q, head_dim)``
        # and 3D indices ``(batch_size, seq_len_q, topk)`` per the installed
        # wheel's ``flash_mla_interface.flash_mla_with_kvcache`` docstring.

        assert topk_idxs is not None, "FP8 sparse decode requires topk_idxs"

        # FlashMLA FP8 kernel requires 4D k_cache: [num_blocks, block_size, num_heads_k=1, kv_dim].
        kv_4d = kv_cache.unsqueeze(-2)
        extra_kv_4d = extra_k_cache.unsqueeze(-2) if extra_k_cache is not None else None

        # topk_idxs: [B, q_len, topk] preferred; collapse a stray num_heads_k axis if present.
        if topk_idxs.dim() == 4:
            topk_3d = topk_idxs.squeeze(2).contiguous()
        else:
            topk_3d = topk_idxs.contiguous()

        if extra_topk_idxs is not None:
            extra_topk_3d = (
                extra_topk_idxs.squeeze(2).contiguous()
                if extra_topk_idxs.dim() == 4
                else extra_topk_idxs.contiguous()
            )
        else:
            extra_topk_3d = None

        if not _flash_mla_with_kvcache_supports_attn_sink():
            return self._forward_flash_mla_ppu(
                q=q,
                kv_4d=kv_4d,
                topk_3d=topk_3d,
                attn_sink=attn_sink,
                topk_length=topk_length,
                extra_kv_4d=extra_kv_4d,
                extra_topk_3d=extra_topk_3d,
                extra_topk_length=extra_topk_length,
                B=B,
                q_len=q_len,
                H=H,
            )

        # Sparse FlashMLA consumes global slot ids from ``indices`` directly.
        # Its sparse branch does not pass block_table/cache_seqlens to the CUDA
        # kernel, so keep dense metadata disabled here.
        block_table = None
        cache_seqlens = None

        # DSv4 attn_sink is per-head fp32, loaded from ckpt (layers.*.attn.attn_sink
        # shape [n_heads], non-zero ~0.3..0.6 mean). FlashMLA kernel applies
        # output *= exp(lse) / (exp(lse) + exp(attn_sink)). Mirrors vLLM
        # ``deepseek_v4_attention.py:860`` (both single- and dual-pool calls).
        attn_out, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_4d,
            block_table=block_table,
            head_dim_v=self.head_dim,
            cache_seqlens=cache_seqlens,
            tile_scheduler_metadata=sched_meta,
            num_splits=None,
            is_fp8_kvcache=True,
            indices=topk_3d,
            softmax_scale=self.softmax_scale,
            topk_length=topk_length,
            attn_sink=attn_sink,
            extra_k_cache=extra_kv_4d,
            extra_indices_in_kvcache=extra_topk_3d,
            extra_topk_length=extra_topk_length,
        )

        return attn_out.view(B, q_len, H, self.head_dim).contiguous()

    # ------------------------------------------------------------------
    # PPU flash_mla_with_kvcache path (reduced signature, 2-based lse)
    # ------------------------------------------------------------------
    def _forward_flash_mla_ppu(
        self,
        *,
        q: torch.Tensor,
        kv_4d: torch.Tensor,
        topk_3d: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_length: Optional[torch.Tensor],
        extra_kv_4d: Optional[torch.Tensor],
        extra_topk_3d: Optional[torch.Tensor],
        extra_topk_length: Optional[torch.Tensor],
        B: int,
        q_len: int,
        H: int,
    ) -> torch.Tensor:
        """Reproduce the GPU dual-pool + attn-sink kernel on the PPU wheel.

        The PPU ``flash_mla_with_kvcache`` lacks the ``attn_sink`` /
        ``topk_length`` / ``extra_*`` kwargs and returns a 2-based (log2)
        ``softmax_lse``.  Each verified primitive:

          * ``topk_length`` -> mask indices at/after the per-request length to
            -1 (the kernel ignores -1 / >= total slots).
          * ``attn_sink``   -> post-hoc ``out *= sigmoid(lse*ln2 - sink)`` in
            natural base (matches ``utils._sparse_attn`` / the prefill
            ``_raw_q_merge_apply_sink`` correction).
          * dual pool       -> two single-pool calls merged with a 2-based
            online-softmax, equivalent to one call over the concatenated pool.

        Sparse indices are global physical slot ids, so per-pool metadata is
        rebuilt locally with an identity ``block_table`` + full-length
        ``cache_seqlens`` (the PPU kernel requires both non-None), preserving
        the GPU "indices consumed directly" semantics.
        """
        o_main, lse_main = self._ppu_sparse_pool(q, kv_4d, topk_3d, topk_length, H)
        if extra_kv_4d is None:
            out = self._ppu_apply_sink(o_main, lse_main, attn_sink)
        else:
            o_extra, lse_extra = self._ppu_sparse_pool(
                q, extra_kv_4d, extra_topk_3d, extra_topk_length, H
            )
            o_merged, lse_merged = self._ppu_merge_pools(
                o_main, lse_main, o_extra, lse_extra
            )
            out = self._ppu_apply_sink(o_merged, lse_merged, attn_sink)
        return out.view(B, q_len, H, self.head_dim).contiguous()

    @staticmethod
    def _ppu_mask_topk_length(
        indices: torch.Tensor, topk_length: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mask indices at column >= per-request ``topk_length`` to -1."""
        if topk_length is None:
            return indices
        topk = indices.shape[-1]
        col = torch.arange(topk, device=indices.device).view(1, 1, topk)
        keep = col < topk_length.to(indices.device).view(-1, 1, 1)
        return torch.where(keep, indices, torch.full_like(indices, -1))

    def _ppu_sparse_pool(
        self,
        q: torch.Tensor,
        k_cache_4d: torch.Tensor,
        indices: torch.Tensor,
        topk_length: Optional[torch.Tensor],
        H: int,
    ) -> tuple:
        """One PPU sparse ``flash_mla_with_kvcache`` call.

        Returns ``(out[B, q_len, H, head_dim], lse[B, q_len, H, 1])`` where the
        lse is the kernel's 2-based (log2) softmax_lse reshaped for merge/sink.
        """
        from flash_mla import (  # type: ignore[import-not-found]
            flash_mla_with_kvcache,
            get_mla_metadata,
        )

        Bq, q_len = q.shape[0], q.shape[1]
        indices = self._ppu_mask_topk_length(indices, topk_length).contiguous()
        num_blocks = k_cache_4d.shape[0]
        page_block_size = k_cache_4d.shape[1]
        total_slots = num_blocks * page_block_size
        cache_seqlens = torch.full(
            (Bq,), total_slots, dtype=torch.int32, device=q.device
        )
        block_table = (
            torch.arange(num_blocks, dtype=torch.int32, device=q.device)
            .unsqueeze(0)
            .expand(Bq, num_blocks)
            .contiguous()
        )
        sched_meta, num_splits = get_mla_metadata(
            cache_seqlens,
            q_len * H,
            1,
            num_heads_q=H,
            is_fp8_kvcache=True,
            topk=indices.shape[-1],
        )
        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=k_cache_4d,
            block_table=block_table,
            cache_seqlens=cache_seqlens,
            head_dim_v=self.head_dim,
            tile_scheduler_metadata=sched_meta,
            num_splits=num_splits,
            softmax_scale=self.softmax_scale,
            is_fp8_kvcache=True,
            indices=indices,
        )
        # kernel lse: [B, H, q_len] -> [B, q_len, H, 1]
        lse_bqh1 = lse.permute(0, 2, 1).unsqueeze(-1).float()
        return out, lse_bqh1

    @staticmethod
    def _ppu_merge_pools(
        o1: torch.Tensor,
        lse1: torch.Tensor,
        o2: torch.Tensor,
        lse2: torch.Tensor,
    ) -> tuple:
        """2-based online-softmax merge of two single-pool results.

        ``lse*`` are 2-based [B, q_len, H, 1]; returns merged (out, 2-based lse).
        """
        m = torch.maximum(lse1, lse2)
        w1 = torch.exp2(lse1 - m)
        w2 = torch.exp2(lse2 - m)
        denom = w1 + w2
        merged = (w1 * o1.float() + w2 * o2.float()) / denom
        merged_lse = m + torch.log2(denom)
        return merged.to(o1.dtype), merged_lse

    def _ppu_apply_sink(
        self,
        out: torch.Tensor,
        lse_bqh1: torch.Tensor,
        attn_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """out[B, q_len, H, D] *= sigmoid(lse_natural - sink); lse is 2-based."""
        if attn_sink is None:
            return out
        sink = attn_sink.to(device=out.device, dtype=torch.float32).view(1, 1, -1, 1)
        lse_nat = lse_bqh1 * math.log(2.0)
        factor = torch.sigmoid(lse_nat - sink)
        factor = torch.where(
            torch.isfinite(lse_bqh1), factor, torch.zeros_like(factor)
        )
        return (out.float() * factor).to(out.dtype)
