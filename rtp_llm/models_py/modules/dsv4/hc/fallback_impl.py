"""PyTorch reference implementation for DeepSeek-V4 Hyper-Connections."""

from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.hc.base import HCHeadBase, HCUnitBase


def _tp_linear_mixes(
    module, x_flat: torch.Tensor, *, use_fp32: bool = False
) -> torch.Tensor:
    """Compute global mHC mixes from one hidden-dimension TP shard."""

    weight = module.fn if use_fp32 else module._fn_bf16()
    linear_input = x_flat.float() if use_fp32 else x_flat
    local_k = int(x_flat.shape[-1])
    global_k = int(weight.shape[-1])
    tp_size = int(getattr(module, "tp_size", 1))
    tp_rank = int(getattr(module, "tp_rank", 0))
    if tp_size == 1:
        if local_k != global_k:
            raise ValueError(
                f"mHC input K={local_k} disagrees with weight K={global_k}"
            )
        rsqrt = torch.rsqrt(
            x_flat.float().square().mean(-1, keepdim=True) + module.norm_eps
        ).to(x_flat.dtype)
        return (F.linear(linear_input, weight) * rsqrt).float()
    if tp_size <= 0 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid mHC TP geometry: size={tp_size}, rank={tp_rank}")
    if global_k != local_k * tp_size:
        raise ValueError(
            f"mHC TP input K={local_k} x tp_size={tp_size} does not match "
            f"weight K={global_k}"
        )

    hc_mult = int(module.hc_mult)
    local_dim = local_k // hc_mult
    global_dim = global_k // hc_mult
    if local_k % hc_mult or global_k % hc_mult:
        raise ValueError(
            f"mHC flattened K must be divisible by hc_mult={hc_mult}: "
            f"local={local_k}, global={global_k}"
        )
    start = tp_rank * local_dim
    weight_shard = (
        weight.view(weight.shape[0], hc_mult, global_dim)
        [:, :, start : start + local_dim]
        .reshape(weight.shape[0], local_k)
    )
    local_square_sum = x_flat.float().square().sum(-1, keepdim=True)
    local_mixes = F.linear(linear_input, weight_shard).float()
    combined = torch.cat((local_square_sum, local_mixes), dim=-1)
    from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce

    combined = all_reduce(combined, Group.TP)
    rsqrt = torch.rsqrt(combined[..., :1] / global_k + module.norm_eps)
    return combined[..., 1:] * rsqrt


def _hc_fallback_chunk_tokens() -> int:
    raw = os.environ.get("DSV4_HC_FALLBACK_CHUNK_TOKENS", "16384")
    try:
        return max(int(raw), 1)
    except (TypeError, ValueError):
        return 16384


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for the TileLang mHC pre mixer.

    Production uses the TileLang path in ``hc/tilelang_impl.py``. This helper
    only backs ``DSV4_HC_IMPL=fallback`` for CPU/dev/reference runs.
    """
    hc = hc_mult
    *batch, mix_hc = mixes.size()
    assert (
        mix_hc == (hc + 2) * hc
    ), f"mix_hc={mix_hc}, expected (hc+2)*hc={(hc + 2) * hc}"

    pre_raw = mixes[..., :hc] * hc_scale[0] + hc_base[:hc]
    pre = pre_raw.sigmoid() + eps

    post_raw = mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    post = 2.0 * post_raw.sigmoid()

    comb_raw = mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb_raw.view(*batch, hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


class FallbackHCUnit(HCUnitBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn_bf16 = self.fn.to(torch.bfloat16)

    def _fn_bf16(self) -> torch.Tensor:
        if self.fn_bf16.shape != self.fn.shape or self.fn_bf16.device != self.fn.device:
            self.fn_bf16 = self.fn.to(torch.bfloat16)
        return self.fn_bf16

    def _linear_mixes(self, x_flat: torch.Tensor) -> torch.Tensor:
        return _tp_linear_mixes(self, x_flat)

    def _pre_impl(self, x: torch.Tensor, dbg_tag=None):
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(-2)  # [..., hc*dim] view
        T = (
            x_flat.shape[0]
            if x_flat.dim() == 2
            else int(torch.tensor(x_flat.shape[:-1]).prod().item())
        )
        chunk = _hc_fallback_chunk_tokens()

        if x_flat.dim() == 2 and T > chunk:
            mixes_list = []
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                mixes_list.append(self._linear_mixes(x_flat[s:e]))
            mixes = torch.cat(mixes_list, dim=0).contiguous()
            del mixes_list
        else:
            mixes = self._linear_mixes(x_flat).contiguous()

        pre, post, comb = _hc_split_sinkhorn(
            mixes,
            self.scale,
            self.base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
        )

        if x_flat.dim() == 2 and T > chunk:
            x_view = x.view(*shape)
            y = torch.empty((T, shape[-1]), dtype=dtype, device=x.device)
            pre_dt = pre.to(dtype)
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                y[s:e] = torch.sum(pre_dt[s:e].unsqueeze(-1) * x_view[s:e], dim=-2)
            y = y.view(*shape[:-2], shape[-1])
        else:
            y = torch.sum(pre.to(dtype).unsqueeze(-1) * x.view(*shape), dim=-2)
        return y.to(dtype), post.unsqueeze(-1), comb

    def _post_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        post_b = post.squeeze(-1) if post.dim() == residual.dim() else post
        # Both branches construct ``[..., hc, dim]`` from ``post * x`` (outer
        # product on hc/dim) and ``comb @ residual`` along hc.  Doing the
        # ``comb`` term as a matmul instead of broadcast-then-sum avoids
        # materialising the ``[..., hc, hc, dim]`` intermediate that OOMs at
        # long context (e.g. 274K tokens × hc=4 × hc=4 × dim=4096 = 9 GB bf16).
        # Mathematically equivalent: result[t, h, d] = sum_{h2} comb[t, h2, h]
        # * residual[t, h2, d] = (comb.transpose(-1, -2) @ residual)[t, h, d].

        T = (
            residual.shape[0]
            if residual.dim() == 3
            else int(torch.tensor(residual.shape[:-2]).prod().item())
        )
        chunk = _hc_fallback_chunk_tokens()
        bf16_path = x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16

        def _compose_chunk(_x, _res, _post_b, _comb):
            if bf16_path:
                first = _post_b.to(x.dtype).unsqueeze(-1) * _x.unsqueeze(-2)
                second = torch.matmul(_comb.to(x.dtype).transpose(-1, -2), _res)
                return (first + second).to(x.dtype)
            y = _post_b.unsqueeze(-1) * _x.unsqueeze(-2) + torch.matmul(
                _comb.transpose(-1, -2), _res
            )
            return y.type_as(x)

        if residual.dim() == 3 and T > chunk:
            out = torch.empty(residual.shape, dtype=x.dtype, device=x.device)
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                out[s:e] = _compose_chunk(x[s:e], residual[s:e], post_b[s:e], comb[s:e])
            return out
        return _compose_chunk(x, residual, post_b, comb)


class FallbackHCHead(HCHeadBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn_bf16 = self.fn.to(torch.bfloat16)

    def _fn_bf16(self) -> torch.Tensor:
        if self.fn_bf16.shape != self.fn.shape or self.fn_bf16.device != self.fn.device:
            self.fn_bf16 = self.fn.to(torch.bfloat16)
        return self.fn_bf16

    def _head_impl(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        # ``x_flat.float()`` materialises a full fp32 copy ``[T, hc*dim]`` —
        # ~17 GiB at 274K tokens × hc=4 × dim=4096.  Chunk along T so the fp32
        # intermediate stays per-chunk-bounded and we never ride above the
        # KV-cache headroom.
        T = shape[0] if x.dim() == 3 else int(torch.tensor(shape[:-2]).prod().item())
        chunk = _hc_fallback_chunk_tokens()

        def _head_chunk(_x):
            _x_flat = _x.flatten(-2)
            _mixes = _tp_linear_mixes(self, _x_flat, use_fp32=True)
            _pre = torch.sigmoid(_mixes * self.scale + self.base) + self.hc_eps
            return torch.sum(_pre.unsqueeze(-1) * _x.float(), dim=-2)

        if x.dim() == 3 and T > chunk:
            y = torch.empty((T, shape[-1]), dtype=torch.float32, device=x.device)
            for s in range(0, T, chunk):
                e = min(s + chunk, T)
                y[s:e] = _head_chunk(x[s:e])
            y = y.view(*shape[:-2], shape[-1])
        else:
            y = _head_chunk(x)
        return y.to(dtype)
