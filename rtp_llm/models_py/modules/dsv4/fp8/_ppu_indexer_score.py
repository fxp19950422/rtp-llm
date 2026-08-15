"""PPU torch indexer score computation.

Replaces ``fp8_paged_indexer_score`` (deep_gemm, unavailable on PPU) +
``indexer_q_rope_fp8_quant_fold`` (triton float8e4nv, uncompilable on PPU).

The score math is:
  logits[b*q+n, t] = sum_h relu( sum_d Q_roped[b,n,h,d] * K[b,t,d] ) * w[b,n,h]

Where K is dequantized from the 132B block-grouped indexer pool.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


_INDEXER_HEAD_DIM = 128
_FP8_MAX = 448.0


def _dequant_pool_block_grouped(
    kv_pool: torch.Tensor,  # [num_blocks, block_size, 132] uint8
    block_table: torch.Tensor,  # [B, max_blocks] int32
    ctx_lens: torch.Tensor,  # [B, q_len] int32 (or [B] for q_len=1)
    max_ctx_len: int,
) -> torch.Tensor:
    """Dequantize the indexer pool into [B, max_ctx_len, 128] bf16.

    Reads the block-grouped layout (fp8-first, scales-after per block).
    Positions beyond ctx_lens are filled with zeros.
    """
    B = block_table.shape[0]
    bs = int(kv_pool.shape[1])  # tokens per cache block
    dev = kv_pool.device
    D = _INDEXER_HEAD_DIM

    # Flatten per-block to raw bytes for block-grouped access
    flat = kv_pool.view(kv_pool.shape[0], bs * (D + 4))

    # Build a gathered K tensor [B, max_ctx_len, D] bf16
    out = torch.zeros(B, max_ctx_len, D, dtype=torch.bfloat16, device=dev)
    for b in range(B):
        ctx = int(ctx_lens.flatten()[b])
        if ctx <= 0:
            continue
        t_end = min(ctx, max_ctx_len)
        for t in range(t_end):
            blk_in_seq = t // bs
            pos_in_blk = t % bs
            phys_blk = int(block_table[b, blk_in_seq])
            fp8_start = pos_in_blk * D
            fp8_raw = flat[phys_blk, fp8_start : fp8_start + D].contiguous()
            scale_start = bs * D + pos_in_blk * 4
            scale_raw = flat[phys_blk, scale_start : scale_start + 4].contiguous()
            fp8 = fp8_raw.view(torch.float8_e4m3fn)
            scale = scale_raw.view(torch.float32).item()
            out[b, t] = (fp8.float() * scale).to(torch.bfloat16)
    return out


def _apply_rope_torch(
    q: torch.Tensor,  # [B, S, H, D] bf16
    freqs_cis: torch.Tensor,  # complex [..., D/2] or [N, rope_dim/2]
    rope_head_dim: int,
) -> torch.Tensor:
    """Apply interleaved partial RoPE (last rope_head_dim dims) in torch."""
    D = q.shape[-1]
    nope_dim = D - rope_head_dim
    nope = q[..., :nope_dim]
    rope = q[..., nope_dim:]

    # freqs_cis broadcast: flatten q rows, tile freqs
    shape = rope.shape  # [..., rope_head_dim]
    rope_2d = rope.reshape(-1, rope_head_dim)
    T = rope_2d.shape[0]
    fc = freqs_cis.reshape(-1, rope_head_dim // 2)
    n_freq = fc.shape[0]
    if n_freq < T:
        repeat = T // n_freq
        fc = fc.repeat(repeat, 1)
    fc = fc[:T]

    even = rope_2d[:, 0::2]
    odd = rope_2d[:, 1::2]
    cos = fc.real.float()
    sin = fc.imag.float()
    new_even = even.float() * cos - odd.float() * sin
    new_odd = odd.float() * cos + even.float() * sin
    roped = torch.empty_like(rope_2d)
    roped[:, 0::2] = new_even.to(roped.dtype)
    roped[:, 1::2] = new_odd.to(roped.dtype)
    return torch.cat([nope, roped.view(shape)], dim=-1)


def ppu_indexer_score(
    q_bf16: torch.Tensor,  # [B, S, H, D=128] bf16 (pre-RoPE)
    weights: torch.Tensor,  # [B, S, H] bf16/fp32 — indexer per-head weights
    freqs_cis: torch.Tensor,  # complex [N, rope_dim/2]
    rope_head_dim: int,
    kv_pool: torch.Tensor,  # [num_blocks, block_size, 132] uint8
    block_table: torch.Tensor,  # [B, max_blocks] int32
    ctx_lens: torch.Tensor,  # [B, S] int32
    max_ctx_len: int,
) -> torch.Tensor:
    """PPU torch indexer score: RoPE Q + dequant K + einsum+relu+wsum.

    Returns ``[B*S, max_ctx_len] fp32`` logits, same contract as
    ``fp8_paged_indexer_score``.
    """
    B, S, H, D = q_bf16.shape
    # Apply RoPE
    q_roped = _apply_rope_torch(q_bf16, freqs_cis, rope_head_dim)
    # Dequant K
    K = _dequant_pool_block_grouped(kv_pool, block_table, ctx_lens, max_ctx_len)
    # [B, max_ctx_len, D]

    # Score: logits[b,s,t] = sum_h relu(sum_d q[b,s,h,d] * K[b,t,d]) * w[b,s,h]
    # einsum: Q[B,S,H,D] @ K[B,T,D]^T -> [B,S,H,T]
    dots = torch.einsum("bshd,btd->bsht", q_roped.float(), K.float())
    # ReLU + weighted sum over H
    scored = F.relu(dots) * weights.float().unsqueeze(-1)  # [B,S,H,T]
    logits = scored.sum(dim=2)  # [B,S,T]
    return logits.view(B * S, max_ctx_len)


def ppu_indexer_score_prefill(
    q_bf16: torch.Tensor,         # [M, H, D=128] bf16 (post-RoPE from caller)
    weights: torch.Tensor,        # [M, H] fp32 — per-head weights (scale folded)
    k_quant: torch.Tensor,        # [N, D] float8_e4m3fn — gathered flat K
    k_scale: torch.Tensor,        # [N] fp32 — per-token scale
    cu_seqlen_ks: torch.Tensor,   # [M] int32 — per-row K start
    cu_seqlen_ke: torch.Tensor,   # [M] int32 — per-row K end
) -> torch.Tensor:
    """PPU torch prefill indexer score (non-paged).

    Same contract as ``fp8_mqa_indexer_score``: returns ``[M, N] fp32``.
    Q is already roped by the caller. K is flat (already gathered from the
    pool by ``_gather_prefill_k_cache``).
    """
    M, H, D = q_bf16.shape
    N = k_quant.shape[0]
    # Dequant K: fp8 * scale -> bf16
    K_f32 = k_quant.float() * k_scale[:, None]  # [N, D]
    # einsum: Q[M,H,D] @ K[N,D]^T -> [M,H,N]
    dots = torch.einsum("mhd,nd->mhn", q_bf16.float(), K_f32)
    # ReLU + weighted sum over H
    scored = F.relu(dots) * weights.float().unsqueeze(-1)  # [M,H,N]
    logits = scored.sum(dim=1)  # [M, N]
    # Apply causal mask: positions outside [ks, ke) -> -inf (or just leave as-is
    # since the caller topk handles masking). Match deep_gemm behavior:
    # entries past ke are "untouched" (we leave them as computed; the topk
    # path applies its own causal cap).
    return logits
