"""Tests for the PPU compressor core compute (_ppu_compressor).

Verifies the compress math (per-dim softmax-over-window pooling + RMSNorm +
interleaved even/odd partial RoPE) against an independent reference:
  * softmax-pool + RMSNorm: standard torch;
  * interleaved RoPE: a complex-exponential rotation reference (cross-check of
    the even/odd formulation used by the fused triton kernel).
"""
from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8._ppu_compressor import (
    compress_and_pack_ppu_656,
    compress_windows_ppu,
    ppu_compress_kv_write,
    save_partial_states_ppu,
)
from rtp_llm.models_py.modules.dsv4.fp8._ppu_kv_layout import (
    PPU_KV_ENTRY_BYTES,
    dequant_ppu_kv_656,
)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device (fp8 pack)")
class TestPpuCompressorCompute(unittest.TestCase):
    def test_compress_matches_complex_exp_rope_reference(self):
        torch.manual_seed(0)
        dev = "cuda"
        T, W, HEAD, ROPE = 5, 8, 512, 64
        NOPE = HEAD - ROPE
        eps = 1e-6
        kv = torch.randn(T, W, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        score = torch.randn(T, W, HEAD, dtype=torch.float32, device=dev) * 0.5
        cos = torch.randn(T, ROPE // 2, device=dev)
        sin = torch.randn(T, ROPE // 2, device=dev)
        rms_w = torch.randn(HEAD, dtype=torch.bfloat16, device=dev) * 0.1 + 1.0

        out = compress_windows_ppu(kv, score, cos, sin, rms_w, eps, ROPE)
        self.assertEqual(tuple(out.shape), (T, HEAD))

        wts = torch.softmax(score.float(), dim=1)
        comp = (kv.float() * wts).sum(1)
        var = (comp * comp).sum(-1, keepdim=True) / HEAD
        nrm = comp * torch.rsqrt(var + eps) * rms_w.float()
        ref = nrm.clone()
        rp = nrm[:, NOPE:]
        c = torch.complex(rp[:, 0::2], rp[:, 1::2])
        rot = c * torch.complex(cos, sin)
        ref_rope = torch.empty_like(rp)
        ref_rope[:, 0::2] = rot.real
        ref_rope[:, 1::2] = rot.imag
        ref[:, NOPE:] = ref_rope
        err = ((out.float() - ref).norm() / ref.norm()).item()
        self.assertLess(err, 5e-3, f"compress compute rel err {err}")  # bf16 floor

    def test_compress_and_pack_shape(self):
        torch.manual_seed(0)
        dev = "cuda"
        T, W, HEAD, ROPE = 3, 8, 512, 64
        kv = torch.randn(T, W, HEAD, dtype=torch.bfloat16, device=dev) * 0.2
        score = torch.randn(T, W, HEAD, dtype=torch.float32, device=dev) * 0.5
        cos = torch.randn(T, ROPE // 2, device=dev)
        sin = torch.randn(T, ROPE // 2, device=dev)
        rms_w = torch.ones(HEAD, dtype=torch.bfloat16, device=dev)
        pool = compress_and_pack_ppu_656(kv, score, cos, sin, rms_w, 1e-6, ROPE)
        self.assertEqual(tuple(pool.shape), (T, PPU_KV_ENTRY_BYTES))
        self.assertEqual(pool.dtype, torch.uint8)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device (fp8 pack)")
class TestPpuCompressorWrite(unittest.TestCase):
    """save_partial_states_ppu + ppu_compress_kv_write vs an element-by-element
    reference. The real fused ``float8e4nv`` kernel cannot compile on PPU, so the
    reference gathers each window position in a Python loop (masking invalid
    negative / unallocated-block positions to softmax -inf, exactly as the
    triton kernel does) and reuses the verified ``compress_windows_ppu`` core."""

    HEAD, ROPE, RATIO, OVERLAP = 512, 64, 4, True
    RING, SBLK, NUM_SB, KV_BS, NUM_KV_BLOCKS = 64, 4, 8, 64, 4

    def _fixtures(self):
        self.dev = "cuda"
        self.coff = 1 + int(self.OVERLAP)
        self.Wn = self.coff * self.RATIO
        self.half = self.ROPE // 2
        self.sw = self.coff * self.HEAD
        self.eps = 1e-6
        torch.manual_seed(0)
        self.rms_w = torch.randn(self.HEAD, dtype=torch.bfloat16, device=self.dev) * 0.1 + 1.0
        self.cos_sin = torch.randn(self.RING, self.ROPE, device=self.dev)
        self.ape = torch.randn(self.RATIO, self.coff * self.HEAD, device=self.dev) * 0.1

    def _compress_ref(self, kv_rows, sc_rows, valids, bpos):
        kv = torch.stack(kv_rows)
        sc = torch.stack(sc_rows)
        v = torch.tensor(valids, device=self.dev)
        sc = torch.where(v[:, None], sc, torch.full_like(sc, float("-inf")))
        kv = torch.where(v[:, None], kv, torch.zeros_like(kv))
        cpos = (bpos // self.RATIO) * self.RATIO
        cos = self.cos_sin[cpos, : self.half]
        sin = self.cos_sin[cpos, self.half :]
        return compress_windows_ppu(
            kv.unsqueeze(0).to(torch.bfloat16), sc.unsqueeze(0),
            cos.unsqueeze(0), sin.unsqueeze(0), self.rms_w, self.eps, self.ROPE)[0]

    def _read_slot(self, kv_cache, slot):
        blk, off = slot // self.KV_BS, slot % self.KV_BS
        return dequant_ppu_kv_656(kv_cache[blk, off].unsqueeze(0))[0]

    def _zeros(self):
        return torch.zeros(self.HEAD, device=self.dev)

    def test_save_partial_states_roundtrip(self):
        self._fixtures()
        N = 8
        positions = torch.arange(N, dtype=torch.int64, device=self.dev)
        kv_flat = torch.randn(N, self.coff * self.HEAD, device=self.dev)
        score_flat = torch.randn(N, self.coff * self.HEAD, device=self.dev)
        state_cache = torch.zeros(self.NUM_SB, self.RING, 2 * self.sw,
                                  dtype=torch.float32, device=self.dev)
        slots = torch.tensor([1 * self.RING + i for i in range(4)]
                             + [2 * self.RING + 4 + i for i in range(4)],
                             dtype=torch.int64, device=self.dev)
        save_partial_states_ppu(kv_flat, score_flat, self.ape, positions,
                                state_cache, slots, self.RATIO)
        for t in range(N):
            blk, off = int(slots[t]) // self.RING, int(slots[t]) % self.RING
            self.assertTrue(torch.allclose(state_cache[blk, off, : self.sw], kv_flat[t]))
            ref_sc = score_flat[t] + self.ape[int(positions[t]) % self.RATIO]
            self.assertTrue(torch.allclose(state_cache[blk, off, self.sw :], ref_sc))

    def _run_write(self, disable_raw, kv_raw, score_raw, seq_start):
        N = 8
        state_cache = torch.randn(self.NUM_SB, self.RING, 2 * self.sw,
                                  dtype=torch.float32, device=self.dev) * 0.2
        block_table = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32, device=self.dev)
        positions = torch.arange(N, dtype=torch.int64, device=self.dev)
        token_to_req = torch.zeros(N, dtype=torch.int64, device=self.dev)
        kv_slot = torch.full((N,), -1, dtype=torch.int64, device=self.dev)
        kv_slot[3], kv_slot[7] = 0, 1
        kv_cache = torch.zeros(self.NUM_KV_BLOCKS, self.KV_BS, PPU_KV_ENTRY_BYTES,
                               dtype=torch.uint8, device=self.dev)
        ppu_compress_kv_write(
            state_cache, token_to_req, positions, block_table, self.rms_w, self.eps,
            self.cos_sin, kv_cache, kv_slot, kv_raw, score_raw, self.ape,
            seq_start, disable_raw, head_dim=self.HEAD, rope_head_dim=self.ROPE,
            compress_ratio=self.RATIO, overlap=self.OVERLAP,
            state_tokens_per_block=self.SBLK)
        return state_cache, block_table, kv_cache

    def test_compress_kv_write_cache_path(self):
        self._fixtures()
        dummy = torch.zeros(1, self.coff * self.HEAD, device=self.dev)
        state_cache, block_table, kv_cache = self._run_write(True, dummy, dummy, 0)

        def ref(bpos):
            kv_rows, sc_rows, valids = [], [], []
            for i in range(self.Wn):
                gp = bpos - self.Wn + 1 + i
                seg = 1 if i >= self.RATIO else 0
                blk = int(block_table[0, (gp // self.SBLK) % block_table.shape[1]]) if gp >= 0 else -1
                if gp < 0 or not (0 < blk < self.NUM_SB):
                    valids.append(False); kv_rows.append(self._zeros()); sc_rows.append(self._zeros()); continue
                ent = state_cache[blk, gp % self.RING]
                kv_rows.append(ent[seg * self.HEAD:(seg + 1) * self.HEAD].float())
                sc_rows.append(ent[self.sw + seg * self.HEAD:self.sw + (seg + 1) * self.HEAD].float())
                valids.append(True)
            return self._compress_ref(kv_rows, sc_rows, valids, bpos)

        for bpos, slot in [(3, 0), (7, 1)]:
            got = self._read_slot(kv_cache, slot)
            r = ref(bpos)
            err = (got.float() - r.float()).norm().item() / (r.float().norm().item() + 1e-9)
            self.assertLess(err, 0.06, f"cache pos={bpos} rel err {err}")

    def test_compress_kv_write_raw_path(self):
        self._fixtures()
        n_raw = 8
        kv_raw = torch.randn(n_raw, self.coff * self.HEAD, device=self.dev)
        score_raw = torch.randn(n_raw, self.coff * self.HEAD, device=self.dev)
        _, _, kv_cache = self._run_write(False, kv_raw, score_raw, 0)

        def ref(bpos):
            kv_rows, sc_rows, valids = [], [], []
            for i in range(self.Wn):
                gp = bpos - self.Wn + 1 + i
                seg = 1 if i >= self.RATIO else 0
                if gp >= 0 and 0 <= gp < n_raw:
                    kv_rows.append(kv_raw[gp, seg * self.HEAD:(seg + 1) * self.HEAD].float())
                    sc = score_raw[gp, seg * self.HEAD:(seg + 1) * self.HEAD] \
                        + self.ape[gp % self.RATIO, seg * self.HEAD:(seg + 1) * self.HEAD]
                    sc_rows.append(sc.float()); valids.append(True)
                else:
                    valids.append(False); kv_rows.append(self._zeros()); sc_rows.append(self._zeros())
            return self._compress_ref(kv_rows, sc_rows, valids, bpos)

        for bpos, slot in [(3, 0), (7, 1)]:
            got = self._read_slot(kv_cache, slot)
            r = ref(bpos)
            err = (got.float() - r.float()).norm().item() / (r.float().norm().item() + 1e-9)
            self.assertLess(err, 0.06, f"raw pos={bpos} rel err {err}")



if __name__ == "__main__":
    unittest.main()
