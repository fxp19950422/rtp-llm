"""PPU DSV4 FP8 KV pool layout (flashmla_ppu 656-byte per-token slot).

The PPU flash_mla wheel is compiled only for the DeepSeek-V3 MLA geometry and
reads a *per-token-contiguous* 656-byte FP8 KV slot -- different from the GPU
``fp8_model1_mla`` 584-byte *block-interleaved* layout
(``_swa_kv_insert_triton`` / ``_compressor_vllm_triton``). Per-token layout
(matches ``flashmla_ppu`` ``tests/quant.py::quantize_k_cache``):

    [0,   512)  NoPE  fp8_e4m3   (dv=512; 4 quant tiles x 128 elems)
    [512, 528)  scales fp32      (4 tiles; scale = abs(tile).max / 448.0)
    [528, 656)  RoPE  bf16       (64 elems)

DSV4 (head_dim=512 = 448 NoPE + 64 RoPE, value == full 512) maps on via
"Mapping A": the whole 512-d head is packed into the NoPE fp8 region and the
RoPE region is left zero; the decode op zero-pads q 512->576 so the padded
(kernel RoPE) dims contribute nothing to the scores. Verified against the real
PPU kernel (rel err ~0.03, fp8 floor). See dsv4/fp8/test/test_ppu_kv_layout.py.
"""

from __future__ import annotations

import torch

PPU_KV_DV = 512
PPU_KV_TILE = 128
PPU_KV_NUM_TILES = PPU_KV_DV // PPU_KV_TILE          # 4
PPU_KV_ROPE_ELEMS = 64
PPU_KV_ROPE_BYTES = PPU_KV_ROPE_ELEMS * 2            # 128 (bf16)
PPU_KV_SCALE_BYTES = PPU_KV_NUM_TILES * 4            # 16 (fp32)
PPU_KV_ENTRY_BYTES = PPU_KV_DV + PPU_KV_SCALE_BYTES + PPU_KV_ROPE_BYTES  # 656
_FP8_MAX = 448.0


def pack_ppu_kv_656(k_512: torch.Tensor) -> torch.Tensor:
    """[N, 512] bf16 DSV4 head -> [N, 656] uint8 PPU FP8 KV slot (Mapping A).

    The full 512-d head is fp8-quantized into the NoPE region (per-128 tile,
    fp32 scale = abs.max/448); the RoPE region is zeroed.
    """
    assert k_512.dim() == 2 and k_512.shape[1] == PPU_KV_DV, tuple(k_512.shape)
    n = k_512.shape[0]
    res = torch.empty(n, PPU_KV_ENTRY_BYTES, dtype=torch.float8_e4m3fn, device=k_512.device)
    nope = res[:, :PPU_KV_DV]
    scale = res[:, PPU_KV_DV : PPU_KV_DV + PPU_KV_SCALE_BYTES].view(torch.float32)
    rope = res[:, PPU_KV_DV + PPU_KV_SCALE_BYTES :].view(torch.bfloat16)
    rope.zero_()
    kf = k_512.float()
    for t in range(PPU_KV_NUM_TILES):
        sl = kf[:, t * PPU_KV_TILE : (t + 1) * PPU_KV_TILE]
        sinv = (sl.abs().amax(dim=-1) / _FP8_MAX).clamp_min(1e-4)
        scale[:, t] = sinv
        nope[:, t * PPU_KV_TILE : (t + 1) * PPU_KV_TILE] = (
            sl / sinv.unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
    return res.view(torch.uint8)


def dequant_ppu_kv_656(pool: torch.Tensor) -> torch.Tensor:
    """[N, 656] uint8 -> [N, 512] bf16 (inverse of pack_ppu_kv_656's NoPE)."""
    assert pool.shape[-1] == PPU_KV_ENTRY_BYTES, tuple(pool.shape)
    p = pool.reshape(-1, PPU_KV_ENTRY_BYTES)
    nope = p[:, :PPU_KV_DV].view(torch.float8_e4m3fn)
    scale = p[:, PPU_KV_DV : PPU_KV_DV + PPU_KV_SCALE_BYTES].view(torch.float32)
    out = torch.empty(p.shape[0], PPU_KV_DV, dtype=torch.bfloat16, device=pool.device)
    for t in range(PPU_KV_NUM_TILES):
        out[:, t * PPU_KV_TILE : (t + 1) * PPU_KV_TILE] = (
            nope[:, t * PPU_KV_TILE : (t + 1) * PPU_KV_TILE].to(torch.float32)
            * scale[:, t : t + 1]
        ).to(torch.bfloat16)
    return out
