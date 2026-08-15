"""Real-kernel end-to-end test for the PPU fp8 sparse decode op.

Runs SparseAttnV4DecodeFp8Op.forward through the ACTUAL PPU flash_mla kernel
with a real 656-byte fp8 KV pool (the flashmla_ppu layout: 512 nope-fp8 +
4 fp32 scales(tile=128) + 64 rope-bf16), DSV4 mapped via q-pad 512->576
(Mapping A: DSV4's full 512-d head packed in the fp8 nope region, rope region
zeroed). Verifies output vs a dequant + base-e-sink torch reference within the
fp8 quant floor. Skipped unless the PPU flash_mla wheel + a CUDA device exist.
"""
from __future__ import annotations

import unittest

import torch

import rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op as decode_mod
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)

_PPU_REAL = False
try:
    import flash_mla  # noqa: F401

    _PPU_REAL = (
        decode_mod._FLASH_MLA_AVAILABLE
        and torch.cuda.is_available()
        and not decode_mod._flash_mla_with_kvcache_supports_attn_sink()
    )
except Exception:  # noqa: BLE001
    _PPU_REAL = False

DV, TILE, D_KER = 512, 128, 576


def _quantize_k_cache_ppu(x576):
    """flashmla_ppu tests/quant.py layout -> [nb, bs, 1, 656] float8_e4m3fn."""
    num_tiles = DV // TILE
    nb, bs, _, d = x576.shape
    x = x576.squeeze(2)
    esz = x.element_size()
    res = torch.empty(
        (nb, bs, DV + num_tiles * 4 + esz * (d - DV)),
        dtype=torch.float8_e4m3fn,
        device=x.device,
    )
    nope = res[..., :DV]
    scale = res[..., DV : DV + num_tiles * 4].view(torch.float32)
    rope = res[..., DV + num_tiles * 4 :].view(x.dtype)
    rope[:] = x[..., DV:]
    for t in range(num_tiles):
        sinv = torch.abs(x[..., t * TILE : (t + 1) * TILE]).max(-1).values / 448.0
        scale[:, :, t] = sinv
        nope[..., t * TILE : (t + 1) * TILE] = (
            x[..., t * TILE : (t + 1) * TILE].float() / sinv.unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
    return res.view(nb, bs, 1, -1)


def _dequant_nope(pool):
    """[1,T,1,656] fp8 -> [T,512] bf16 (Mapping A: nope region = DSV4 full head)."""
    num_tiles = DV // TILE
    p = pool.view(pool.shape[0], pool.shape[1], -1)
    nope = p[..., :DV]
    scale = p[..., DV : DV + num_tiles * 4].view(torch.float32)
    out = torch.empty(p.shape[0], p.shape[1], DV, dtype=torch.bfloat16, device=p.device)
    for t in range(num_tiles):
        out[..., t * TILE : (t + 1) * TILE] = (
            nope[..., t * TILE : (t + 1) * TILE].to(torch.float32) * scale[..., t : t + 1]
        ).to(torch.bfloat16)
    return out[0]


@unittest.skipUnless(_PPU_REAL, "requires PPU flash_mla wheel (reduced sig) + CUDA")
class TestDecodePpuRealKernel(unittest.TestCase):
    def test_fp8_656_single_pool_end_to_end(self):
        torch.manual_seed(0)
        dev = "cuda"
        H, T, topk, HEAD = 64, 128, 128, 512
        kv = torch.randn(T, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        z = torch.zeros(T, D_KER - HEAD, dtype=torch.bfloat16, device=dev)
        pool4d = _quantize_k_cache_ppu(torch.cat([kv, z], -1).view(1, T, 1, D_KER))
        pool3d = pool4d[:, :, 0, :].contiguous()  # [1, T, 656] fp8 (forward iface)

        q = torch.randn(1, 1, H, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        sink = torch.rand(H, dtype=torch.float32, device=dev) * 0.5 + 0.2
        idx = torch.randint(0, T, (1, 1, topk), dtype=torch.int32, device=dev)

        op = SparseAttnV4DecodeFp8Op(n_heads=H, head_dim=HEAD, softmax_scale=HEAD ** -0.5)
        out = op.forward(q, pool3d, sink, idx, sched_meta=None)

        self.assertEqual(tuple(out.shape), (1, 1, H, HEAD))
        self.assertTrue(torch.isfinite(out).all().item())

        # dequant + base-e-sink torch reference (value = full 512, matches op sink)
        kv_dq = _dequant_nope(pool4d)              # [T, 512]
        sel = kv_dq[idx[0, 0].long()].float()      # [topk, 512]
        qf = q[0, 0].float()                       # [H, 512]
        sc = torch.einsum("hd,kd->hk", qf, sel) * (HEAD ** -0.5)
        m = sc.amax(-1, keepdim=True)
        e = torch.exp(sc - m)
        denom = e.sum(-1, keepdim=True) + torch.exp(sink.view(H, 1) - m)
        oref = torch.einsum("hk,kd->hd", e, sel) / denom
        err = ((out[0, 0].float() - oref).norm() / oref.norm()).item()
        self.assertLess(err, 0.1, f"PPU fp8 decode rel err {err} exceeds fp8 floor")


if __name__ == "__main__":
    unittest.main()
