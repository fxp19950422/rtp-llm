"""PPU DeepGEMM MoE helpers for GLM4.7 INT8 W8A8.

This module mirrors the SGLang/vLLM PPU fast path while keeping the RTP
surface small: GPU-side expert counting, contiguous nopad packing, and fused
unpermute/reduce. It intentionally avoids host-side sort, .item(), and
index_add_ in the decode hot path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _round_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


def compute_aligned_M(
    M: int,
    num_topk: int,
    local_num_experts: int,
    alignment: int = 1,
) -> int:
    """Conservative static upper bound for contiguous nopad MoE rows."""
    alignment = max(int(alignment), 1)
    m_sum = M * num_topk + local_num_experts * (alignment - 1)
    return _round_up(m_sum, alignment)


@triton.jit
def _count_expert_num_tokens_kernel(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts: tl.constexpr,
    topk_numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    curr_expert = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    topk_ptrs = topk_ids_ptr + offsets

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for block_start in range(0, topk_numel, BLOCK_SIZE):
        mask = offsets < (topk_numel - block_start)
        expert_ids = tl.load(topk_ptrs, mask=mask, other=-1)
        acc += tl.where(expert_ids == curr_expert, 1, 0)
        topk_ptrs += BLOCK_SIZE

    if curr_expert < num_experts:
        tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))


def count_expert_num_tokens(topk_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    assert topk_ids.dtype.is_signed, "topk_ids must use -1 for invalid experts"
    expert_num_tokens = torch.empty(
        (num_experts,), device=topk_ids.device, dtype=torch.int32
    )
    block_size = min(max(int(topk_ids.numel()), 1), 1024)
    block_size = triton.next_power_of_2(block_size)
    _count_expert_num_tokens_kernel[(num_experts,)](
        topk_ids,
        expert_num_tokens,
        num_experts=num_experts,
        topk_numel=topk_ids.numel(),
        BLOCK_SIZE=block_size,
    )
    return expert_num_tokens


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offsets = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offsets,
        mask=offsets < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert

    # Keep the offset in registers. This avoids the stale global-memory read
    # pattern that made early PPU scatter kernels flaky.
    cur_start = tl.sum(tl.where(offsets == cur_expert, cumsum, 0))
    tl.store(expert_start_loc + cur_expert, cur_start)
    tl.debug_barrier()

    cur_count = tl.load(num_recv_tokens_per_expert + cur_expert)
    off_expert = tl.arange(0, BLOCK_E)
    for start_m in tl.range(0, cur_count, BLOCK_E):
        offs = start_m + off_expert
        mask = offs < cur_count
        tl.store(m_indices + cur_start + offs, cur_expert, mask=mask)


@triton.jit
def _fwd_kernel_ep_scatter_2_optimal(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    with_scale: tl.constexpr,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    COPY_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    token_id_i32 = tl.program_id(0)
    token_id = token_id_i32.to(tl.int64)

    expert_offsets = tl.arange(0, BLOCK_SIZE)
    expert_mask = expert_offsets < topk_num
    expert_loc = tl.load(
        recv_topk + token_id * recv_topk_stride0 + expert_offsets,
        mask=expert_mask,
        other=-1,
    )

    route_mask = expert_mask & (expert_loc >= 0)
    if route_mask.sum() == 0:
        return

    dest_i32 = tl.atomic_add(expert_start_loc + expert_loc, 1, mask=route_mask)
    tl.store(
        output_index + token_id * output_index_stride0 + expert_offsets,
        dest_i32,
        mask=route_mask,
        eviction_policy="evict_last",
    )
    tl.debug_barrier()
    dest = tl.load(
        output_index + token_id * output_index_stride0 + expert_offsets,
        mask=route_mask,
    ).to(tl.int64)

    value_offsets = tl.arange(0, COPY_SIZE)
    for _ in tl.range(0, triton.cdiv(HIDDEN_SIZE, COPY_SIZE), num_stages=num_stages):
        copy_mask = value_offsets < HIDDEN_SIZE
        values = tl.load(
            recv_x + token_id * recv_x_stride0 + value_offsets,
            mask=copy_mask,
        )
        output_offsets = dest[:, None] * output_tensor_stride0 + value_offsets[None, :]
        values = values[None, :].broadcast_to(BLOCK_SIZE, COPY_SIZE)
        tl.store(
            output_tensor + output_offsets,
            values,
            mask=copy_mask[None, :] & route_mask[:, None],
        )
        value_offsets += COPY_SIZE

    if with_scale:
        # GLM4.7 INT8 uses per-token/per-channel scale, i.e. one activation
        # scale per routed row.
        scale_value = tl.load(recv_x_scale + token_id * recv_x_scale_stride0)
        scale_offsets = dest * output_tensor_scale_stride0
        tl.store(output_tensor_scale + scale_offsets, scale_value, mask=route_mask)


@torch.no_grad()
def ep_scatter_sail(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor | None,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor | None,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
    block_align: int = 1,
) -> None:
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    block_e = max(int(block_align), 1)

    assert m_indices.shape[0] % block_e == 0
    _fwd_kernel_ep_scatter_1[(num_experts,)](
        num_recv_tokens_per_expert,
        expert_start_loc,
        m_indices,
        num_experts=num_experts,
        BLOCK_E=block_e,
        BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        num_warps=8,
    )

    grid = lambda meta: (recv_x.shape[0],)
    _fwd_kernel_ep_scatter_2_optimal[grid](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        0 if recv_x_scale is None else recv_x_scale.stride(0),
        0 if recv_x_scale is None else recv_x_scale.stride(1),
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        0 if output_tensor_scale is None else output_tensor_scale.stride(0),
        0 if output_tensor_scale is None else output_tensor_scale.stride(1),
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        with_scale=(recv_x_scale is not None),
        topk_num=recv_topk.shape[1],
        HIDDEN_SIZE=hidden_size,
        SCALE_HIDDEN_SIZE=1,
        BLOCK_SIZE=triton.next_power_of_2(recv_topk.shape[1]),
        COPY_SIZE=512,
        num_stages=3,
        num_warps=8,
    )


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block_i32 = tl.program_id(0)
    cur_block = cur_block_i32.to(tl.int64)
    start_token_i32 = tl.program_id(1)
    grid_num = tl.num_programs(1)

    for token_i32 in range(start_token_i32, total_token_num, grid_num):
        token = token_i32.to(tl.int64)
        off_d = tl.arange(0, BLOCK_D)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for topk_i32 in range(0, topk_num):
            topk = topk_i32.to(tl.int64)
            expert_id = tl.load(recv_topk_ids + token * recv_topk_ids_stride0 + topk)
            if expert_id >= 0:
                src_i32 = tl.load(input_index + token * input_index_stride0 + topk)
                src = src_i32.to(tl.int64)
                weight = tl.load(
                    recv_topk_weight + token * recv_topk_weight_stride0 + topk
                )
                value = tl.load(
                    input_tensor
                    + src * input_tensor_stride0
                    + cur_block * BLOCK_D
                    + off_d
                )
                acc += value.to(tl.float32) * weight
        tl.store(
            output_tensor + token * output_tensor_stride0 + cur_block * BLOCK_D + off_d,
            acc.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
) -> None:
    hidden_size = input_tensor.shape[1]
    block_d = 128 if hidden_size % 1024 != 0 else 1024
    while hidden_size % block_d:
        block_d //= 2
    assert hidden_size % block_d == 0

    grid = (triton.cdiv(hidden_size, block_d), min(output_tensor.shape[0], 1024))
    _fwd_kernel_ep_gather[grid](
        output_tensor.shape[0],
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        BLOCK_D=block_d,
        num_warps=2,
    )


@torch.no_grad()
def deepgemm_moe_permute(
    aq: torch.Tensor,
    aq_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    local_num_experts: int,
    aq_out: torch.Tensor | None = None,
    block_align: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack routed INT8 activations into PPU DeepGEMM contiguous nopad layout."""
    assert aq.ndim == 2
    assert topk_ids.dtype.is_signed, "topk_ids must use -1 for invalid experts"

    hidden_size = aq.size(1)
    device = aq.device
    m_sum = compute_aligned_M(
        M=topk_ids.size(0),
        num_topk=topk_ids.size(1),
        local_num_experts=local_num_experts,
        alignment=block_align,
    )

    if aq_out is None:
        aq_out = torch.empty((m_sum, hidden_size), device=device, dtype=aq.dtype)
    else:
        assert aq_out.shape == (m_sum, hidden_size)

    aq_scale_out = torch.empty((m_sum, 1), device=device, dtype=torch.float32)
    # DeepGEMM nopad skips negative m_indices. Keep padded rows invalid.
    expert_ids = torch.full((m_sum,), -1, device=device, dtype=torch.int32)
    inv_perm = torch.empty(topk_ids.shape, device=device, dtype=torch.int32)
    expert_start_loc = torch.empty(
        (local_num_experts,), device=device, dtype=torch.int32
    )
    expert_num_tokens = count_expert_num_tokens(topk_ids, local_num_experts)

    ep_scatter_sail(
        recv_x=aq,
        recv_x_scale=aq_scale,
        recv_topk=topk_ids.to(torch.int32),
        num_recv_tokens_per_expert=expert_num_tokens,
        expert_start_loc=expert_start_loc,
        output_tensor=aq_out,
        output_tensor_scale=aq_scale_out,
        m_indices=expert_ids,
        output_index=inv_perm,
        block_align=block_align,
    )
    return aq_out, aq_scale_out, expert_ids, inv_perm, expert_num_tokens


@torch.no_grad()
def deepgemm_unpermute_and_reduce(
    input_tensor: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    inv_perm: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    ep_gather(input_tensor, topk_ids, topk_weights, inv_perm, output)
    return output
