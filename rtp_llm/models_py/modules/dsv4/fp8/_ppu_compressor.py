"""PPU DSV4 compressor core compute (pure torch).

Reproduces the compress math of the fused triton kernel
``_compressor_vllm_triton._fused_kv_compress_norm_rope_insert_*_attn`` (which
cannot compile on PPU: triton lacks ``fp8e4nv``). Per boundary token:

  1. per-dim softmax over the gathered window positions of ``gate_score`` (the
     caller must have already added the APE table, ``ape[pos % compress_ratio]``);
  2. weighted sum of the window KV -> compressed KV ``[head_dim]``;
  3. RMSNorm over the full head with ``rms_weight``;
  4. interleaved even/odd partial RoPE on the last ``rope_head_dim`` dims
     (``new_even = even*cos - odd*sin``, ``new_odd = odd*cos + even*sin``);

then packs the full head (Mapping A) into the PPU 656-byte FP8 KV slot via
``pack_ppu_kv_656``. The gather (state-pool / raw dispatch) is done by the
caller; this is the numeric core. Verified on ZW810E: RoPE matches a complex-
exponential reference (bf16 floor); softmax/RMSNorm are standard torch; pack is
covered by ``_ppu_kv_layout`` tests.
"""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv4.fp8._ppu_kv_layout import pack_ppu_kv_656


def compress_windows_ppu(
    kv_windows: torch.Tensor,     # [T, W, head_dim] bf16 — gathered KV per boundary token
    score_windows: torch.Tensor,  # [T, W, head_dim] fp32 — gate score + APE
    cos: torch.Tensor,            # [T, rope_head_dim // 2] fp32
    sin: torch.Tensor,            # [T, rope_head_dim // 2] fp32
    rms_weight: torch.Tensor,     # [head_dim] bf16
    eps: float,
    rope_head_dim: int,
) -> torch.Tensor:
    """Compress + RMSNorm + partial RoPE -> ``[T, head_dim]`` bf16."""
    T, W, head_dim = kv_windows.shape
    wts = torch.softmax(score_windows.float(), dim=1)            # softmax over W, per dim
    compressed = (kv_windows.float() * wts).sum(dim=1)           # [T, head_dim]
    var = (compressed * compressed).sum(dim=-1, keepdim=True) / head_dim
    normed = compressed * torch.rsqrt(var + eps) * rms_weight.float()  # [T, head_dim]

    nope_dim = head_dim - rope_head_dim
    nope = normed[:, :nope_dim]
    rope = normed[:, nope_dim:]                                  # [T, rope_head_dim]
    even = rope[:, 0::2]                                         # [T, rope//2]
    odd = rope[:, 1::2]
    new_even = even * cos - odd * sin
    new_odd = odd * cos + even * sin
    roped = torch.empty_like(rope)
    roped[:, 0::2] = new_even
    roped[:, 1::2] = new_odd
    return torch.cat([nope, roped], dim=-1).to(torch.bfloat16)   # [T, head_dim]


def compress_and_pack_ppu_656(
    kv_windows: torch.Tensor,
    score_windows: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    rope_head_dim: int,
) -> torch.Tensor:
    """``compress_windows_ppu`` then pack to ``[T, 656]`` uint8 (Mapping A)."""
    normed = compress_windows_ppu(
        kv_windows, score_windows, cos, sin, rms_weight, eps, rope_head_dim
    )
    return pack_ppu_kv_656(normed)


def save_partial_states_ppu(
    kv_flat: torch.Tensor,       # [N, coff*head_dim] fp32/bf16
    score_flat: torch.Tensor,    # [N, coff*head_dim] fp32/bf16
    ape: torch.Tensor,           # [compress_ratio, coff*head_dim] fp32
    positions: torch.Tensor,     # [N] int
    state_cache: torch.Tensor,   # [num_blocks, ring, 2*coff*head_dim] fp32
    slot_mapping: torch.Tensor,  # [N] int; -1 = skip
    compress_ratio: int,
) -> None:
    """Torch port of ``run_save_partial_states`` (no fp8; PPU-safe).

    Per token with ``slot_mapping[t] >= 0`` write ``kv`` into the first half
    and ``score + ape[pos % compress_ratio]`` into the second half of the
    fp32 state row at ``(slot // block_size, slot % block_size)``.
    """
    if int(slot_mapping.shape[0]) == 0:
        return
    block_size = int(state_cache.shape[1])
    sw = int(state_cache.shape[-1] // 2)
    hs = int(kv_flat.shape[-1])
    slot = slot_mapping.long()
    keep = slot >= 0
    if not bool(keep.any()):
        return
    idx = keep.nonzero(as_tuple=True)[0]
    s = slot[idx]
    blk = s // block_size
    off = s % block_size
    ape_rows = positions.long()[idx] % int(compress_ratio)
    kv_v = kv_flat[idx].to(state_cache.dtype)
    sc_v = score_flat[idx].to(state_cache.dtype) + ape[ape_rows].to(state_cache.dtype)
    state_cache[blk, off, :hs] = kv_v
    state_cache[blk, off, sw : sw + hs] = sc_v


def ppu_compress_kv_write(
    state_cache: torch.Tensor,       # [num_state_blocks, ring, 2*coff*head_dim] fp32
    token_to_req: torch.Tensor,      # [N] int
    positions: torch.Tensor,         # [N] int
    block_table: torch.Tensor,       # [B, stride] int — state-pool block table
    rms_weight: torch.Tensor,        # [head_dim] bf16
    rms_eps: float,
    cos_sin_cache: torch.Tensor,     # [max_pos, rope_head_dim] fp32 (cos|sin)
    kv_cache: torch.Tensor,          # [num_kv_blocks, kv_block_size, 656] uint8
    kv_slot_mapping: torch.Tensor,   # [N] int — KV-pool slot per token (-1 skip)
    kv_raw: torch.Tensor,            # [n_raw, coff*head_dim] — raw path source
    score_raw: torch.Tensor,         # [n_raw, coff*head_dim]
    ape: torch.Tensor,               # [compress_ratio, coff*head_dim] fp32
    seq_start: int,                  # abs pos of kv_raw[0] (ignored if disabled/varlen)
    disable_raw_path: bool,
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    state_tokens_per_block: int,
    seq_start_per_req: torch.Tensor = None,  # [B] int — varlen raw
    cu_seq_per_req: torch.Tensor = None,     # [B+1] int — varlen raw
) -> None:
    """Torch port of ``run_fused_compress_kv_write`` for the PPU 656 KV pool.

    Replaces the ``float8e4nv`` triton kernel (uncompilable on PPU). For each
    boundary token (``(pos+1) % compress_ratio == 0`` and ``kv_slot >= 0``)
    gathers the prior ``(1+overlap)*compress_ratio`` window positions from the
    raw arrays (prefill) or the fp32 state pool (prefix-cache / decode), softmax-
    reduces over positions, RMSNorms, partial-RoPEs, then packs Mapping-A into
    the 656-byte slot. Source dispatch and masking mirror the triton kernel
    exactly (raw / cache / batched-varlen; invalid positions -> softmax -inf).
    """
    dev = positions.device
    r = int(compress_ratio)
    cf = 1 + int(overlap)
    Wn = cf * r
    ring = int(state_cache.shape[1])
    sw = int(state_cache.shape[-1] // 2)
    stride = int(block_table.shape[1])
    num_sb = int(state_cache.shape[0])
    kv_bs = int(kv_cache.shape[1])
    half = int(rope_head_dim // 2)
    sblk = int(state_tokens_per_block)

    pos = positions.long()
    req = token_to_req.long()
    kslot = kv_slot_mapping.long()
    n_raw = 0 if disable_raw_path else int(kv_raw.shape[0])
    batched = (
        (not disable_raw_path)
        and seq_start_per_req is not None
        and cu_seq_per_req is not None
    )

    bnd = ((pos + 1) % r == 0) & (kslot >= 0)
    bi = bnd.nonzero(as_tuple=True)[0]
    if bi.numel() == 0:
        return
    T = int(bi.numel())
    bpos = pos[bi]
    breq = req[bi]
    bkslot = kslot[bi]

    toks = torch.arange(Wn, device=dev)
    gpos = bpos[:, None] - Wn + 1 + toks[None, :]      # [T, Wn]
    mask_pos = gpos >= 0
    seg = (toks >= r).long()                            # [Wn]
    # segment column index into a [., coff*head_dim] row -> [T, Wn, head_dim]
    seg_col = (seg[:, None] * head_dim + torch.arange(head_dim, device=dev)[None, :])
    seg_col = seg_col[None].expand(T, Wn, head_dim)

    # ---- raw path (prefill) ----
    if disable_raw_path:
        use_raw = torch.zeros((T, Wn), dtype=torch.bool, device=dev)
        kv_from_raw = torch.zeros((T, Wn, head_dim), dtype=torch.float32, device=dev)
        score_from_raw = torch.zeros((T, Wn, head_dim), dtype=torch.float32, device=dev)
    else:
        if batched:
            ssr = seq_start_per_req.long()[breq][:, None]
            clo = cu_seq_per_req.long()[breq][:, None]
            chi = cu_seq_per_req.long()[breq + 1][:, None]
            flat_in_req = gpos - ssr
            use_raw = mask_pos & (flat_in_req >= 0) & (flat_in_req < (chi - clo))
            flat_idx = clo + flat_in_req
        else:
            flat_idx = gpos - int(seq_start)
            use_raw = mask_pos & (flat_idx >= 0) & (flat_idx < n_raw)
        flat_safe = flat_idx.clamp(0, max(n_raw - 1, 0))
        kv_from_raw = torch.gather(kv_raw[flat_safe].float(), -1, seg_col)
        score_from_raw = torch.gather(score_raw[flat_safe].float(), -1, seg_col)
        ape_rows = (gpos % r).clamp_min(0)
        ape_from = torch.gather(ape[ape_rows].float(), -1, seg_col)
        score_from_raw = score_from_raw + ape_from

    # ---- cache path (prefix-cache hit / decode) ----
    use_cache = mask_pos & ~use_raw
    gsafe = gpos.clamp_min(0)
    blk_log = (gsafe // sblk) % stride
    block_num = block_table[breq[:, None].expand(T, Wn), blk_log].long()
    valid_block = use_cache & (block_num > 0) & (block_num < num_sb)
    bn_safe = torch.where(valid_block, block_num, torch.zeros_like(block_num))
    ring_off = gsafe % ring
    entry = state_cache[bn_safe, ring_off]              # [T, Wn, 2*sw]
    kv_from_cache = torch.gather(entry[..., :sw].float(), -1, seg_col)
    score_from_cache = torch.gather(entry[..., sw:].float(), -1, seg_col)

    # ---- combine + mask ----
    ur = use_raw[..., None]
    kv = torch.where(ur, kv_from_raw, kv_from_cache)    # [T, Wn, head_dim]
    score = torch.where(ur, score_from_raw, score_from_cache)
    final_valid = (use_raw | (use_cache & valid_block))[..., None]
    score = torch.where(final_valid, score, torch.full_like(score, float("-inf")))
    kv = torch.where(final_valid, kv, torch.zeros_like(kv))

    cpos = (bpos // r) * r
    cos = cos_sin_cache[cpos, :half]
    sin = cos_sin_cache[cpos, half:]
    normed = compress_windows_ppu(
        kv.to(torch.bfloat16), score, cos, sin, rms_weight, rms_eps, rope_head_dim
    )
    packed = pack_ppu_kv_656(normed)                    # [T, 656] uint8
    kblk = bkslot // kv_bs
    kpos = bkslot % kv_bs
    kv_cache[kblk, kpos] = packed


# ---------------------------------------------------------------------------
# Indexer 132B layout: full head_dim=128 in one fp8 quant block + fp32 scale.
# The block-grouped format (K-all-then-scale-all) is handled at scatter time;
# the per-token pack produces a contiguous 132-byte row [128 fp8 | 4 fp32].
# ---------------------------------------------------------------------------
_INDEXER_HEAD_DIM = 128
_INDEXER_ENTRY_BYTES = 132
_FP8_MAX = 448.0


def pack_indexer_132(normed: torch.Tensor) -> torch.Tensor:
    """Pack ``[T, 128]`` bf16 into ``[T, 132]`` uint8 (per-token fp8 + fp32 scale).

    Quantization mirrors ``_indexer_k_quant_kernel`` (one quant block, absmax/448
    scale) but uses torch ops (no triton float8e4nv). Result layout per token:
      bytes [0:128] = float8_e4m3fn quantized dims
      bytes [128:132] = fp32 scale (little-endian)
    """
    assert normed.dim() == 2 and normed.shape[1] == _INDEXER_HEAD_DIM
    T = normed.shape[0]
    dev = normed.device
    f32 = normed.float()
    absmax = f32.abs().amax(dim=-1)                         # [T]
    scale = (absmax / _FP8_MAX).clamp_min(1e-12)            # [T]
    scaled = f32 / scale[:, None]
    fp8 = scaled.to(torch.float8_e4m3fn)                    # clamp+RNE
    fp8_u8 = fp8.view(torch.uint8)                          # [T, 128]
    scale_u8 = scale.to(torch.float32).view(torch.uint8).view(T, 4)
    return torch.cat([fp8_u8, scale_u8], dim=-1)            # [T, 132]


def dequant_indexer_132(pool: torch.Tensor) -> torch.Tensor:
    """Dequantize ``[T, 132]`` uint8 → ``[T, 128]`` bf16."""
    assert pool.dim() == 2 and pool.shape[1] == _INDEXER_ENTRY_BYTES
    T = pool.shape[0]
    fp8_raw = pool[:, :_INDEXER_HEAD_DIM].contiguous()
    fp8 = fp8_raw.view(torch.float8_e4m3fn)
    scale = pool[:, _INDEXER_HEAD_DIM:].contiguous().view(torch.float32)  # [T, 1]
    return (fp8.float() * scale).to(torch.bfloat16)


def _scatter_indexer_132_block_grouped(
    packed: torch.Tensor,       # [T, 132] uint8 (per-token contiguous)
    kv_cache: torch.Tensor,     # [num_blocks, block_size, 132] uint8
    slots: torch.Tensor,        # [T] int64 — kv slot index per boundary token
) -> None:
    """Scatter per-token [128 fp8 | 4 scale] into the block-grouped KV pool.

    The pool's physical byte layout per block is NOT ``[block_size, 132]``
    per-token rows. It is block-grouped:
      bytes [0, block_size*128): fp8 data (token_0 || token_1 || ...)
      bytes [block_size*128, block_size*132): fp32 scales (4 each)
    This function writes into the correct byte positions.
    """
    bs = int(kv_cache.shape[1])
    # Work on a flat per-block view
    flat = kv_cache.view(kv_cache.shape[0], bs * _INDEXER_ENTRY_BYTES)
    for i in range(int(slots.shape[0])):
        s = int(slots[i].item())
        if s < 0:
            continue
        blk = s // bs
        pos = s % bs
        row = packed[i]                                        # [132]
        fp8_data = row[:_INDEXER_HEAD_DIM]                     # [128]
        scale_data = row[_INDEXER_HEAD_DIM:]                   # [4]
        flat[blk, pos * _INDEXER_HEAD_DIM : pos * _INDEXER_HEAD_DIM + _INDEXER_HEAD_DIM] = fp8_data
        flat[blk, bs * _INDEXER_HEAD_DIM + pos * 4 : bs * _INDEXER_HEAD_DIM + pos * 4 + 4] = scale_data


def ppu_compress_kv_write_indexer(
    state_cache: torch.Tensor,       # [num_state_blocks, ring, 2*coff*head_dim] fp32
    token_to_req: torch.Tensor,      # [N] int
    positions: torch.Tensor,         # [N] int
    block_table: torch.Tensor,       # [B, stride] int — state-pool block table
    rms_weight: torch.Tensor,        # [head_dim=128] bf16
    rms_eps: float,
    cos_sin_cache: torch.Tensor,     # [max_pos, rope_head_dim] fp32 (cos|sin)
    kv_cache: torch.Tensor,          # [num_kv_blocks, kv_block_size, 132] uint8
    kv_slot_mapping: torch.Tensor,   # [N] int — KV-pool slot per token (-1 skip)
    kv_raw: torch.Tensor,            # [n_raw, coff*head_dim]
    score_raw: torch.Tensor,         # [n_raw, coff*head_dim]
    ape: torch.Tensor,               # [compress_ratio, coff*head_dim] fp32
    seq_start: int,
    disable_raw_path: bool,
    *,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    state_tokens_per_block: int,
    seq_start_per_req: torch.Tensor = None,
    cu_seq_per_req: torch.Tensor = None,
) -> None:
    """Torch port of ``run_fused_compress_kv_write`` for the indexer 132B pool.

    Same gather+compress+RoPE as ``ppu_compress_kv_write`` (656B), but packs
    the FULL roped head_dim=128 result into one fp8 quant block + fp32 scale
    and scatters into the block-grouped layout that ``fp8_paged_mqa_logits``
    and ``_indexer_k_quant_kernel`` expect.
    """
    dev = positions.device
    r = int(compress_ratio)
    cf = 1 + int(overlap)
    Wn = cf * r
    ring = int(state_cache.shape[1])
    sw = int(state_cache.shape[-1] // 2)
    stride = int(block_table.shape[1])
    num_sb = int(state_cache.shape[0])
    kv_bs = int(kv_cache.shape[1])
    half = int(rope_head_dim // 2)
    sblk = int(state_tokens_per_block)

    pos = positions.long()
    req = token_to_req.long()
    kslot = kv_slot_mapping.long()
    n_raw = 0 if disable_raw_path else int(kv_raw.shape[0])
    batched = (
        (not disable_raw_path)
        and seq_start_per_req is not None
        and cu_seq_per_req is not None
    )

    bnd = ((pos + 1) % r == 0) & (kslot >= 0)
    bi = bnd.nonzero(as_tuple=True)[0]
    if bi.numel() == 0:
        return
    T = int(bi.numel())
    bpos = pos[bi]
    breq = req[bi]
    bkslot = kslot[bi]

    toks = torch.arange(Wn, device=dev)
    gpos = bpos[:, None] - Wn + 1 + toks[None, :]
    mask_pos = gpos >= 0
    seg = (toks >= r).long()
    seg_col = (seg[:, None] * head_dim + torch.arange(head_dim, device=dev)[None, :])
    seg_col = seg_col[None].expand(T, Wn, head_dim)

    if disable_raw_path:
        use_raw = torch.zeros((T, Wn), dtype=torch.bool, device=dev)
        kv_from_raw = torch.zeros((T, Wn, head_dim), dtype=torch.float32, device=dev)
        score_from_raw = torch.zeros((T, Wn, head_dim), dtype=torch.float32, device=dev)
    else:
        if batched:
            ssr = seq_start_per_req.long()[breq][:, None]
            clo = cu_seq_per_req.long()[breq][:, None]
            chi = cu_seq_per_req.long()[breq + 1][:, None]
            flat_in_req = gpos - ssr
            use_raw = mask_pos & (flat_in_req >= 0) & (flat_in_req < (chi - clo))
            flat_idx = clo + flat_in_req
        else:
            flat_idx = gpos - int(seq_start)
            use_raw = mask_pos & (flat_idx >= 0) & (flat_idx < n_raw)
        flat_safe = flat_idx.clamp(0, max(n_raw - 1, 0))
        kv_from_raw = torch.gather(kv_raw[flat_safe].float(), -1, seg_col)
        score_from_raw = torch.gather(score_raw[flat_safe].float(), -1, seg_col)
        ape_rows = (gpos % r).clamp_min(0)
        ape_from = torch.gather(ape[ape_rows].float(), -1, seg_col)
        score_from_raw = score_from_raw + ape_from

    use_cache = mask_pos & ~use_raw
    gsafe = gpos.clamp_min(0)
    blk_log = (gsafe // sblk) % stride
    block_num = block_table[breq[:, None].expand(T, Wn), blk_log].long()
    valid_block = use_cache & (block_num > 0) & (block_num < num_sb)
    bn_safe = torch.where(valid_block, block_num, torch.zeros_like(block_num))
    ring_off = gsafe % ring
    entry = state_cache[bn_safe, ring_off]
    kv_from_cache = torch.gather(entry[..., :sw].float(), -1, seg_col)
    score_from_cache = torch.gather(entry[..., sw:].float(), -1, seg_col)

    ur = use_raw[..., None]
    kv = torch.where(ur, kv_from_raw, kv_from_cache)
    score = torch.where(ur, score_from_raw, score_from_cache)
    final_valid = (use_raw | (use_cache & valid_block))[..., None]
    score = torch.where(final_valid, score, torch.full_like(score, float("-inf")))
    kv = torch.where(final_valid, kv, torch.zeros_like(kv))

    cpos = (bpos // r) * r
    cos = cos_sin_cache[cpos, :half]
    sin = cos_sin_cache[cpos, half:]
    normed = compress_windows_ppu(
        kv.to(torch.bfloat16), score, cos, sin, rms_weight, rms_eps, rope_head_dim
    )
    packed = pack_indexer_132(normed)
    _scatter_indexer_132_block_grouped(packed, kv_cache, bkslot)
