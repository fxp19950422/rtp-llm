"""Integration test: drive the real ``CompressorFP8._launch`` on a 656-byte
PPU KV pool so the PPU torch write branch (``save_partial_states_ppu`` +
``ppu_compress_kv_write``) is exercised end-to-end (pool binding, CompressorMeta
field usage, slot decomposition, detection flag). On PPU this is the only path
that can run at all — the GPU fused ``float8e4nv`` kernel cannot compile — so a
passing run also proves the branch was taken.

State + boundary-KV writes are compared against an independent torch reference.
``freqs_cis`` is set to unit complex so RoPE is identity (isolates gather/pool
wiring from the RoPE numerics already covered by test_ppu_compressor).
"""
from __future__ import annotations

import unittest

import torch

from rtp_llm.models_py.modules.dsv4.cp import _CP_ROLE_MAIN
from rtp_llm.models_py.modules.dsv4.fp8._compressor_consts import KV_HEAD_DIM
from rtp_llm.models_py.modules.dsv4.fp8._ppu_compressor import compress_windows_ppu
from rtp_llm.models_py.modules.dsv4.fp8._ppu_kv_layout import (
    PPU_KV_ENTRY_BYTES,
    dequant_ppu_kv_656,
)
from rtp_llm.models_py.modules.dsv4.fp8.compressor import (
    CompressorFP8,
    build_prefill_metadata,
)

DEVICE = "cuda"
TOKENS_PER_STATE_BLOCK = 256


def _build_compressor(dim, head_dim, rope_head_dim, compress_ratio):
    coff = 1 + (compress_ratio == 4)
    weights = {
        "ape": torch.randn(compress_ratio, coff * head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.1,
        "wkv": torch.randn(coff * head_dim, dim, dtype=torch.bfloat16, device=DEVICE) * 0.05,
        "wgate": torch.randn(coff * head_dim, dim, dtype=torch.bfloat16, device=DEVICE) * 0.05,
        "norm": (torch.randn(head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.1 + 1.0),
    }
    cmp = CompressorFP8(
        dim=dim, head_dim=head_dim, rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio, max_batch_size=1, cp_role=_CP_ROLE_MAIN,
        norm_eps=1e-6, compressor_weights=weights,
    )
    # unit complex -> cos=1, sin=0 -> identity RoPE
    cmp.freqs_cis = torch.ones(4096, rope_head_dim // 2, dtype=torch.complex64, device=DEVICE)
    return cmp


def _bind_ppu_pools(cmp, seqlen, head_dim, coff, compress_ratio):
    """Bind fake state + 656-byte PPU KV pools; returns (state_3d, kv_3d, kv_eb)."""
    state_eb = TOKENS_PER_STATE_BLOCK
    state_total_blocks = 1 + 2  # block 0 = unallocated sentinel
    hidden = 2 * coff * head_dim
    state_view_2d = torch.zeros(state_total_blocks * state_eb, hidden, dtype=torch.float32, device=DEVICE)
    state_block_table = torch.tensor([[1, 2]], dtype=torch.int32, device=DEVICE)

    kv_eb = TOKENS_PER_STATE_BLOCK // compress_ratio
    n_compressed = (seqlen + compress_ratio - 1) // compress_ratio
    kv_blocks_needed = max(1, (n_compressed + kv_eb - 1) // kv_eb)
    kv_pool_3d = torch.zeros(1 + kv_blocks_needed, kv_eb, PPU_KV_ENTRY_BYTES, dtype=torch.uint8, device=DEVICE)
    kv_block_table = torch.arange(1, 1 + kv_blocks_needed, dtype=torch.int32, device=DEVICE).reshape(1, kv_blocks_needed)

    cmp.set_pool_context(
        kv_pool_view=kv_pool_3d, kv_block_table=kv_block_table, kv_eb=kv_eb,
        state_pool_view=state_view_2d, state_block_table=state_block_table,
        state_eb=state_eb, state_tokens_per_block=state_eb,
        kv_tokens_per_block=kv_eb * compress_ratio,
    )
    return state_view_2d.view(state_total_blocks, state_eb, hidden), kv_pool_3d, kv_eb


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA/PPU device")
class TestPpuCompressorLaunch(unittest.TestCase):
    HEAD, ROPE, RATIO, DIM, SEQLEN = KV_HEAD_DIM, 64, 4, 128, 16

    def test_launch_ppu_prefill(self):
        torch.manual_seed(0)
        coff = 1 + (self.RATIO == 4)
        cmp = _build_compressor(self.DIM, self.HEAD, self.ROPE, self.RATIO)
        state_3d, kv_3d, kv_eb = _bind_ppu_pools(cmp, self.SEQLEN, self.HEAD, coff, self.RATIO)

        N = self.SEQLEN
        sp = 0
        kv_flat = torch.randn(N, coff * self.HEAD, device=DEVICE) * 0.2
        score_flat = torch.randn(N, coff * self.HEAD, device=DEVICE) * 0.5
        meta = build_prefill_metadata(cmp, sp=sp, bsz=1, seqlen=self.SEQLEN, device=torch.device(DEVICE))

        # A 656-byte pool + head_dim==512 must select the PPU torch branch; on
        # PPU the alternative (triton float8e4nv) would raise, so reaching the
        # assertions already proves the branch. _launch(seq_start=sp) => raw path.
        cmp._launch(kv_flat, score_flat, meta, seq_start=sp)
        torch.cuda.synchronize()

        sw = coff * self.HEAD
        state_bt = torch.tensor([[1, 2]], device=DEVICE)
        # --- state pool write (save_partial_states_ppu through _launch) ---
        for t in range(N):
            pos = sp + t
            blk = int(state_bt[0, (pos // TOKENS_PER_STATE_BLOCK) % 2])
            inb = pos % TOKENS_PER_STATE_BLOCK
            self.assertTrue(torch.allclose(state_3d[blk, inb, :sw], kv_flat[t], atol=1e-4))
            ref_sc = score_flat[t] + cmp.ape[pos % self.RATIO].float()
            self.assertTrue(torch.allclose(state_3d[blk, inb, sw:], ref_sc, atol=1e-4))

        # --- boundary KV write (ppu_compress_kv_write through _launch) ---
        cos = torch.ones(1, self.ROPE // 2, device=DEVICE)
        sin = torch.zeros(1, self.ROPE // 2, device=DEVICE)
        wrote_any = False
        for t in range(N):
            pos = sp + t
            if (pos + 1) % self.RATIO != 0:
                continue
            slot = int(meta.kv_slots[t])
            self.assertGreaterEqual(slot, 0)
            blk, inb = slot // kv_eb, slot % kv_eb
            got = dequant_ppu_kv_656(kv_3d[blk, inb].unsqueeze(0))[0]
            self.assertTrue(kv_3d[blk, inb].any().item(), f"boundary pos={pos} slot empty")
            wrote_any = True

            Wn = coff * self.RATIO
            kv_rows, sc_rows, valids = [], [], []
            for i in range(Wn):
                gp = pos - Wn + 1 + i
                seg = 1 if i >= self.RATIO else 0
                if gp >= 0 and 0 <= gp - sp < N:
                    kv_rows.append(kv_flat[gp - sp, seg * self.HEAD:(seg + 1) * self.HEAD].float())
                    sc = score_flat[gp - sp, seg * self.HEAD:(seg + 1) * self.HEAD] \
                        + cmp.ape[gp % self.RATIO, seg * self.HEAD:(seg + 1) * self.HEAD]
                    sc_rows.append(sc.float()); valids.append(True)
                else:
                    valids.append(False)
                    kv_rows.append(torch.zeros(self.HEAD, device=DEVICE))
                    sc_rows.append(torch.zeros(self.HEAD, device=DEVICE))
            kv = torch.stack(kv_rows); sc = torch.stack(sc_rows)
            v = torch.tensor(valids, device=DEVICE)
            sc = torch.where(v[:, None], sc, torch.full_like(sc, float("-inf")))
            kv = torch.where(v[:, None], kv, torch.zeros_like(kv))
            ref = compress_windows_ppu(kv.unsqueeze(0).to(torch.bfloat16), sc.unsqueeze(0),
                                       cos, sin, cmp.norm.weight, cmp.norm_eps, self.ROPE)[0]
            err = (got.float() - ref.float()).norm().item() / (ref.float().norm().item() + 1e-9)
            self.assertLess(err, 0.06, f"boundary pos={pos} rel err {err}")
        self.assertTrue(wrote_any, "no boundary tokens written")


if __name__ == "__main__":
    unittest.main()
