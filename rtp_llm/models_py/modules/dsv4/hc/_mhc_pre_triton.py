"""Fused Triton kernels for the DeepSeek-V4 mHC (hyper-connection) pre-mixer
and head reduce.

The PyTorch fallback (``hc/fallback_impl.py``) runs the 20-iteration Sinkhorn
normalisation as ~40 tiny ``comb.sum(dim=...) / ...`` launches on a ``[T, hc,
hc]`` tensor (hc=4 -> 16 elements). On the wr5 PPU decode timeline that made
``dsv4.hc.*.pre`` ~17% of GPU time -- almost pure launch overhead. These
kernels collapse the whole post-``mixes`` computation into one launch and the
readout into another, matching the fallback math exactly:

  pre[h]   = sigmoid(mixes[h]      * scale[0] + base[h])       + eps
  post[h]  = 2 * sigmoid(mixes[hc+h] * scale[1] + base[hc+h])
  comb     = (mixes[2hc:] * scale[2] + base[2hc:]).view(hc, hc)
  comb     = softmax(comb, dim=-1) + eps
  comb    /= comb.sum(dim=-2, keepdim=True) + eps
  repeat sinkhorn_iters-1 times: /sum(dim=-1); /sum(dim=-2)
  y[d]     = sum_h pre_readout[h] * x[h, d]           (readout; pre for head)

``mixes`` is produced by ``mhc_rms_mul`` below -- ONE fused kernel that
does the RMS square-mean/rsqrt, the bf16 ``* rsqrt`` rescale and the fp32
upcast (the torch chain for that tail was 7 elementwise launches per HC call;
at decode it is ~600 launches/step for 43 layers x 2 units). The ``fn``
matmul itself stays in cuBLAS/torch bf16; only the launch-bound tail moves
into Triton.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_pre_sinkhorn_kernel(
    mixes_ptr,  # [T, MIX] fp32 contiguous, MIX = (HC + 2) * HC
    scale_ptr,  # [3] fp32
    base_ptr,  # [MIX] fp32
    pre_ptr,  # [T, HC] fp32 out
    post_ptr,  # [T, HC] fp32 out
    comb_ptr,  # [T, HC, HC] fp32 out
    T: tl.int32,
    EPS: tl.constexpr,
    ITERS: tl.constexpr,
    HC: tl.constexpr,
    MIX: tl.constexpr,
):
    """One program per token. Holds the [HC, HC] comb tile in registers and
    runs the full Sinkhorn loop there, so the entire pre mixer is one launch."""
    t = tl.program_id(axis=0)
    if t >= T:
        return

    scale0 = tl.load(scale_ptr + 0)
    scale1 = tl.load(scale_ptr + 1)
    scale2 = tl.load(scale_ptr + 2)

    # pre / post: 1-D [HC].
    h = tl.arange(0, HC)
    pre_lin = tl.load(mixes_ptr + t * MIX + h)
    post_lin = tl.load(mixes_ptr + t * MIX + HC + h)
    pre = tl.sigmoid(pre_lin * scale0 + tl.load(base_ptr + h)) + EPS
    post = 2.0 * tl.sigmoid(post_lin * scale1 + tl.load(base_ptr + HC + h))
    tl.store(pre_ptr + t * HC + h, pre)
    tl.store(post_ptr + t * HC + h, post)

    # comb: [HC, HC] tile. ii = row (dim=-2), jj = col (dim=-1).
    ii = tl.arange(0, HC)[:, None]
    jj = tl.arange(0, HC)[None, :]
    coff = ii * HC + jj
    cb = tl.load(mixes_ptr + t * MIX + 2 * HC + coff) * scale2 + tl.load(
        base_ptr + 2 * HC + coff
    )

    # softmax over dim=-1 (jj), numerically stable, then + eps.
    mx = tl.max(cb, axis=1)[:, None]
    e = tl.exp(cb - mx)
    cb = e / tl.sum(e, axis=1)[:, None]
    cb = cb + EPS
    # first normalise over dim=-2 (ii, rows).
    cb = cb / (tl.sum(cb, axis=0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        cb = cb / (tl.sum(cb, axis=1)[:, None] + EPS)  # dim=-1
        cb = cb / (tl.sum(cb, axis=0)[None, :] + EPS)  # dim=-2
    tl.store(comb_ptr + t * (HC * HC) + coff, cb)


@triton.jit
def _mhc_rms_mul_kernel(
    x_ptr,  # [T, D] bf16 contiguous, D = hc * dim
    gemm_ptr,  # [T, MIX] bf16 contiguous (F.linear(x_flat, fn_bf16) output)
    mixes_ptr,  # [T, MIX] fp32 out
    T: tl.int32,
    D: tl.int32,
    NORM_EPS: tl.constexpr,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,  # pow2 >= MIX for tl.arange
    BLOCK_D: tl.constexpr,
):
    """Fused tail of the pre-mixer, one program per token. Matches the
    fallback chain bit-for-bit (up to fp32 sum order in the mean):

      rsqrt  = bf16(1/sqrt(mean(x^2) + eps))          (torch: .float().square()
                                                        .mean(-1) -> rsqrt -> .to)
      mixes  = fp32(bf16(gemm * rsqrt))               (torch: (lin*rsqrt).float())
    """
    t = tl.program_id(axis=0)
    if t >= T:
        return
    row = t.to(tl.int64) * D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        idx = off + tl.arange(0, BLOCK_D)
        v = tl.load(x_ptr + row + idx, mask=idx < D, other=0.0).to(tl.float32)
        acc += v * v
    mean = tl.sum(acc, axis=0) / D
    rsqrt = tl.rsqrt(mean + NORM_EPS).to(tl.bfloat16).to(tl.float32)
    m = tl.arange(0, MIX_PAD)
    g = tl.load(gemm_ptr + t * MIX + m, mask=m < MIX, other=0.0).to(tl.float32)
    mixes = (g * rsqrt).to(tl.bfloat16).to(tl.float32)
    tl.store(mixes_ptr + t * MIX + m, mixes, mask=m < MIX)


def mhc_rms_mul(
    x_flat: torch.Tensor,  # [..., D] bf16, D = hc * dim
    gemm_out: torch.Tensor,  # [..., MIX] bf16 (F.linear(x_flat, fn_bf16))
    *,
    norm_eps: float,
) -> torch.Tensor:
    """Return ``mixes`` fp32 ``[T, MIX]``: fused RMS rsqrt + rescale + upcast.

    Replaces the 7-launch torch chain
    ``x.float() / .square() / .mean(-1) / rsqrt(+eps) / .to(dtype) /
    gemm*rsqrt / .float()`` with a single kernel per HC pre call.
    """
    D = x_flat.shape[-1]
    x2d = x_flat.reshape(-1, D)
    gemm2d = gemm_out.reshape(-1, gemm_out.shape[-1])
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    if not gemm2d.is_contiguous():
        gemm2d = gemm2d.contiguous()
    T = x2d.shape[0]
    mix = gemm2d.shape[1]
    mixes = torch.empty((T, mix), dtype=torch.float32, device=x2d.device)
    if T == 0 or mix == 0:
        return mixes
    BLOCK_D = 4096 if D >= 4096 else triton.next_power_of_2(D)
    _mhc_rms_mul_kernel[(T,)](
        x2d,
        gemm2d,
        mixes,
        T=T,
        D=D,
        NORM_EPS=float(norm_eps),
        MIX=mix,
        MIX_PAD=triton.next_power_of_2(mix),
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return mixes


@triton.jit
def _mhc_readout_kernel(
    pre_ptr,  # [T, HC] fp32
    x_ptr,  # [T, HC, D] input dtype contiguous
    y_ptr,  # [T, D] output dtype contiguous
    T: tl.int32,
    D: tl.int32,
    HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """y[t, d] = sum_h pre[t, h] * x[t, h, d]. One program per (token, D-block)."""
    t = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    if t >= T:
        return
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for hh in tl.static_range(HC):
        p = tl.load(pre_ptr + t * HC + hh)
        xv = tl.load(x_ptr + t * HC * D + hh * D + d_off, mask=mask, other=0.0)
        acc += p * xv.to(tl.float32)
    tl.store(y_ptr + t * D + d_off, acc.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mhc_head_kernel(
    mixes_ptr,  # [T, HC] fp32  (= F.linear(x_flat, fn) * rsqrt)
    scale_ptr,  # [HC] fp32
    base_ptr,  # [HC] fp32
    x_ptr,  # [T, HC, D] fp32 contiguous  (head runs in fp32)
    y_ptr,  # [T, D] fp32 out
    T: tl.int32,
    D: tl.int32,
    EPS: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused head: pre = sigmoid(mixes*scale + base) + eps, then
    y[t, d] = sum_h pre[t, h] * x[t, h, d]. All fp32 (matches fallback head)."""
    t = tl.program_id(axis=0)
    pid_d = tl.program_id(axis=1)
    if t >= T:
        return
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = d_off < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for hh in tl.static_range(HC):
        m = tl.load(mixes_ptr + t * HC + hh)
        s = tl.load(scale_ptr + hh)
        b = tl.load(base_ptr + hh)
        p = tl.sigmoid(m * s + b) + EPS
        xv = tl.load(x_ptr + t * HC * D + hh * D + d_off, mask=mask, other=0.0)
        acc += p * xv
    tl.store(y_ptr + t * D + d_off, acc, mask=mask)


def mhc_pre_sinkhorn(
    mixes: torch.Tensor,  # [T, MIX] fp32 contiguous
    scale: torch.Tensor,  # [3] fp32
    base: torch.Tensor,  # [MIX] fp32
    *,
    hc_mult: int,
    eps: float,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(pre [T, hc], post [T, hc], comb [T, hc, hc])`` in fp32."""
    T, mix = mixes.shape
    hc = hc_mult
    assert mix == (hc + 2) * hc, f"mixes width {mix} != (hc+2)*hc={(hc + 2) * hc}"
    scale = scale.to(torch.float32).contiguous()
    base = base.to(torch.float32).contiguous()
    pre = torch.empty((T, hc), dtype=torch.float32, device=mixes.device)
    post = torch.empty((T, hc), dtype=torch.float32, device=mixes.device)
    comb = torch.empty((T, hc, hc), dtype=torch.float32, device=mixes.device)
    if T == 0:
        return pre, post, comb
    _mhc_pre_sinkhorn_kernel[(T,)](
        mixes,
        scale,
        base,
        pre,
        post,
        comb,
        T=T,
        EPS=float(eps),
        ITERS=int(sinkhorn_iters),
        HC=hc,
        MIX=mix,
        num_warps=1,
        num_stages=1,
    )
    return pre, post, comb


def mhc_readout(pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``y[t, d] = sum_h pre[t, h] * x[t, h, d]`` -> ``[T, D]`` in x's dtype."""
    T, hc, D = x.shape
    y = torch.empty((T, D), dtype=x.dtype, device=x.device)
    if T == 0 or D == 0:
        return y
    BLOCK_D = 1024 if D >= 1024 else triton.next_power_of_2(D)
    grid = (T, triton.cdiv(D, BLOCK_D))
    _mhc_readout_kernel[grid](
        pre.contiguous(),
        x.contiguous(),
        y,
        T=T,
        D=D,
        HC=hc,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return y


def mhc_head(
    mixes: torch.Tensor,  # [T, HC] fp32  (= F.linear(x_flat, fn) * rsqrt)
    scale: torch.Tensor,  # [HC] fp32 (broadcast)
    base: torch.Tensor,  # [HC] fp32
    x: torch.Tensor,  # [T, HC, D] fp32 contiguous
    *,
    hc_mult: int,
    eps: float,
) -> torch.Tensor:
    """Fused head sigmoid + readout -> ``[T, D]`` fp32."""
    T, hc, D = x.shape
    assert hc == hc_mult, f"x hc {hc} != hc_mult {hc_mult}"
    scale = scale.to(torch.float32).reshape(-1).contiguous()
    base = base.to(torch.float32).reshape(-1).contiguous()
    # Head scale/base may be scalar -> broadcast to [HC] for the per-h load.
    if scale.numel() == 1:
        scale = scale.expand(hc).contiguous()
    if base.numel() == 1:
        base = base.expand(hc).contiguous()
    y = torch.empty((T, D), dtype=torch.float32, device=x.device)
    if T == 0 or D == 0:
        return y
    BLOCK_D = 1024 if D >= 1024 else triton.next_power_of_2(D)
    grid = (T, triton.cdiv(D, BLOCK_D))
    _mhc_head_kernel[grid](
        mixes.contiguous(),
        scale,
        base,
        x.contiguous(),
        y,
        T=T,
        D=D,
        EPS=float(eps),
        HC=hc,
        BLOCK_D=BLOCK_D,
        num_warps=4,
        num_stages=2,
    )
    return y
