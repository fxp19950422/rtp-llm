"""SGLang Triton fused MoE kernel ported for INT8 W8A8 per-channel quantization.

This module provides a Triton-based fused MoE implementation that fuses
per-token INT8 quantization into the GEMM kernel, reducing kernel launches
from 15+ to 5 for the MoE forward pass.

Ported from SGLang's fused_moe_triton_kernels.py and moe_align_block_size.py.
Key difference: simplified to only handle INT8 W8A8 with per-channel quant.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# Try to import our fused quantization kernels
try:
    from rtp_llm.ops.compute_ops import per_token_group_quant_int8

    _HAS_FUSED_QUANT = True
except (ImportError, AttributeError):
    _HAS_FUSED_QUANT = False

# Try to import v2 fused kernel (SiLU+mul+INT8 quant in 1 kernel)
try:
    from rtp_llm.ops.compute_ops import per_token_group_quant_int8_v2

    _HAS_FUSED_QUANT_V2 = True
except (ImportError, AttributeError):
    _HAS_FUSED_QUANT_V2 = False


# =============================================================================
# moe_align_block_size: token sorting for MoE (replaces sort+bincount+cumsum)
# =============================================================================


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align token distribution across experts for block GEMM.

    Produces sorted flat positions, expert IDs per block, and padded token count.
    No CPU-GPU sync (.item()) — uses upper bounds for allocation.
    CUDA Graph compatible.

    Args:
        topk_ids: [M, top_k] expert indices for each token
        block_size: block size for GEMM tiling
        num_experts: total number of experts
    Returns:
        sorted_token_ids: flat positions [0, M*top_k) sorted by expert, padded.
                          Padding entries = total_expanded (masked in kernel).
        expert_ids: expert index for each block
        num_tokens_post_padded: total tokens after padding (1-element GPU tensor)
    """
    M, top_k = topk_ids.shape
    device = topk_ids.device
    total_expanded = M * top_k

    # Flatten and sort by expert id
    flat_ids = topk_ids.reshape(-1)  # [M*top_k]
    sorted_ids, sort_perm = flat_ids.sort(stable=True)

    # Count tokens per expert
    # torch.bincount() is NOT graph-safe (internal sync on some backends).
    # Use scatter_add_ for CUDA Graph compatibility.
    clamped_ids = flat_ids.clamp(min=0, max=num_experts - 1).long()
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    expert_counts.scatter_add_(0, clamped_ids, torch.ones_like(clamped_ids))

    # Pad each expert's count to block_size alignment
    padded_counts = triton.cdiv(expert_counts, block_size) * block_size

    # Cumsum of padded counts (GPU tensor, no .item() sync)
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
    cumsum[1:] = padded_counts.cumsum(0).to(torch.int32)

    # total_padded as GPU scalar (no .item() sync)
    total_padded = cumsum[-1].clone()  # scalar GPU tensor

    # Upper bound for allocation (avoids .item() sync)
    # Each expert adds at most block_size-1 padding tokens
    max_total_padded = total_expanded + num_experts * (block_size - 1)

    # Create sorted_token_ids with upper bound size
    # Padding tokens have index = total_expanded (out of range, masked in kernel)
    sorted_token_ids = torch.full(
        (max_total_padded,), total_expanded, dtype=torch.int32, device=device
    )

    # Store FLAT POSITIONS (not token indices) in sorted_token_ids.
    # The kernel uses offs_token // top_k to index into A (token-level),
    # and offs_token directly to index into topk_weights.
    #
    # We need to place each token into its expert's PADDED block.
    #   expert_offsets  = padded block starts (from cumsum of padded_counts)
    #   unpadded_offsets = start position of each expert in the SORTED order
    #   within_expert   = position within expert's group (in sorted order)
    #   write_pos       = expert's padded block start + within-expert position
    expert_offsets = cumsum[:num_experts]
    unpadded_offsets = torch.zeros(num_experts, dtype=torch.int32, device=device)
    unpadded_offsets[1:] = expert_counts[:-1].cumsum(0).to(torch.int32)
    within_expert = (
        torch.arange(total_expanded, device=device, dtype=torch.int32)
        - unpadded_offsets[sorted_ids.long()]
    )
    write_pos = expert_offsets[sorted_ids.long()] + within_expert
    sorted_token_ids[write_pos] = sort_perm.to(torch.int32)

    # Create expert_ids: which expert each block belongs to (vectorized, no Python loop)
    # block_boundaries[e] = cumsum[e+1] // block_size = exclusive end block for expert e
    max_num_blocks = triton.cdiv(max_total_padded, block_size)
    block_indices = torch.arange(max_num_blocks, device=device, dtype=torch.int32)
    block_boundaries = cumsum[1:].to(torch.int32) // block_size  # [E]
    # searchsorted right=True: for each block, find which expert it belongs to
    expert_ids = torch.searchsorted(block_boundaries, block_indices, right=True).to(
        torch.int32
    )
    expert_ids = expert_ids.clamp(max=num_experts - 1)

    return sorted_token_ids, expert_ids, total_padded.reshape(1)


# =============================================================================
# Triton fused MoE kernel (adapted from SGLang)
# =============================================================================


@triton.jit
def _fused_moe_int8_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_bse,
    stride_bsn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
    A_IS_EXPANDED: tl.constexpr,
):
    """Triton fused MoE kernel for INT8 W8A8 with per-channel quantization.

    Reads INT8 activation and weight, does INT8 GEMM, applies per-token and
    per-channel scales in the epilogue, outputs BF16.

    A_IS_EXPANDED: if True, activation is [M*top_k, K] (per-token-expert pair,
        used for GEMM2 where intermediate is already expanded). If False,
        activation is [M, K] (per-token, used for GEMM1) and offs_token//top_k
        is used to index it.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    offs_token = offs_token.to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Pointers for A (activation, INT8) and B (weight, INT8)
    # When A_IS_EXPANDED (GEMM2): activation is [M*top_k, K], index by offs_token directly
    # When not expanded (GEMM1): activation is [M, K], index by offs_token // top_k
    if A_IS_EXPANDED:
        a_row_idx = offs_token
    else:
        a_row_idx = offs_token // top_k
    a_ptrs = a_ptr + (a_row_idx[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Load per-channel weight scale: [E, N, 1]
    b_scale_ptrs = (
        b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
    )
    b_scale = tl.load(b_scale_ptrs)

    # Load per-token activation scale: [M, 1] (GEMM1) or [M*top_k, 1] (GEMM2)
    a_scale_ptrs = a_scale_ptr + a_row_idx * stride_asm
    a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]

    # INT8 GEMM with FP32 accumulation
    # Cast INT8 to compute_type (bfloat16) for Tensor Core compatible dot product.
    # Scales are applied in the epilogue (per-token * per-channel).
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0)

        a = a.to(compute_type)
        b = b.to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply per-token and per-channel scales
    accumulator *= a_scale * b_scale

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # Write output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# =============================================================================
# invoke_fused_moe_kernel: Python wrapper for the Triton kernel
# =============================================================================

# Default config for INT8 W8A8 MoE
_DEFAULT_MOE_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}


def invoke_fused_moe_int8_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    mul_routed_weight: bool = True,
    a_is_expanded: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Invoke fused MoE INT8 GEMM kernel.

    A: INT8 activation. If a_is_expanded=False, shape is [M, K] (per-token,
       GEMM1). If a_is_expanded=True, shape is [M*top_k, K] (per-token-expert,
       GEMM2 intermediate).
    B: INT8 weight [E, N, K]
    C: BF16 output [M*top_k, N]
    A_scale: FP32 per-token scale. Shape matches A ([M, 1] or [M*top_k, 1]).
    B_scale: FP32 per-channel scale [E, N, 1]
    """
    if config is None:
        config = _DEFAULT_MOE_CONFIG

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    K = B.shape[2]
    even_Ks = K % config["BLOCK_SIZE_K"] == 0

    _fused_moe_int8_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2],  # N, K
        sorted_token_ids.shape[0],  # EM
        topk_ids.numel(),  # num_valid_tokens
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        A_scale.stride(0) if A_scale.ndim >= 2 else 0,
        B_scale.stride(0) if B_scale.ndim >= 2 else 0,
        B_scale.stride(1) if B_scale.ndim >= 2 else 0,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=tl.bfloat16,
        even_Ks=even_Ks,
        A_IS_EXPANDED=a_is_expanded,
        **config,
    )


# =============================================================================
# moe_sum_reduce: Triton kernel for expert output reduction
# =============================================================================


@triton.jit
def _moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)

    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]

    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)

    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def moe_sum_reduce_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> None:
    """Reduce expert outputs by summing across top-k dimension.

    input: [M, top_k, hidden_dim]
    output: [M, hidden_dim]
    """
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    # Dynamically select BLOCK_DIM based on hidden_dim for better occupancy.
    # GLM4.7 hidden=7168: BLOCK_DIM=8192 -> 1 block instead of 4 (BLOCK_DIM=2048).
    BLOCK_DIM = min(triton.next_power_of_2(hidden_dim), 8192)
    BLOCK_M = 1
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    _moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )


# =============================================================================
# Triton per-token INT8 quantization (from SGLang int8_kernel.py)
# =============================================================================


@triton.jit
def _per_token_quant_int8_triton(
    x_ptr,
    xq_ptr,
    scale_ptr,
    stride_x,
    stride_xq,
    N,
    BLOCK: tl.constexpr,
):
    """Triton kernel: per-token INT8 quantization (absmax → scale → quant → cast)."""
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row_id * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(x)), 1e-10)
    scale_x = absmax / 127.0
    x_q = x * (127.0 / absmax)
    x_q = tl.extra.cuda.libdevice.round(x_q).to(tl.int8)

    tl.store(xq_ptr + row_id * stride_xq + cols, x_q, mask=mask)
    tl.store(scale_ptr + row_id, scale_x)


def per_token_quant_int8_triton(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token INT8 quantization using Triton kernel.

    Args:
        x: BF16/FP16 tensor [M, K], must be contiguous
    Returns:
        x_q: INT8 tensor [M, K]
        x_s: FP32 tensor [M, 1]
    """
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_q = torch.empty_like(x, dtype=torch.int8)
    x_s = torch.empty(x.shape[:-1] + (1,), device=x.device, dtype=torch.float32)

    BLOCK = triton.next_power_of_2(N)
    num_warps = min(max(BLOCK // 256, 1), 8)

    _per_token_quant_int8_triton[(M,)](
        x,
        x_q,
        x_s,
        stride_x=x.stride(-2) if x.ndim >= 2 else 0,
        stride_xq=x_q.stride(-2) if x_q.ndim >= 2 else 0,
        N=N,
        BLOCK=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return x_q, x_s


# =============================================================================
# Fused MoE forward: complete 5-kernel MoE execution
# =============================================================================


def fused_moe_int8_forward(
    hidden_states: torch.Tensor,  # [M, K] BF16
    w1: torch.Tensor,  # [E, 2*inter, K] INT8
    w2: torch.Tensor,  # [E, hidden, inter] INT8
    w1_scale: torch.Tensor,  # [E, 2*inter, 1] or [E, 2*inter] FP32
    w2_scale: torch.Tensor,  # [E, hidden, 1] or [E, hidden] FP32
    topk_weights: torch.Tensor,  # [M, top_k] FP32
    topk_ids: torch.Tensor,  # [M, top_k] INT32
    inter_size: int,
    top_k: int,
    block_size: int = 64,
) -> torch.Tensor:
    """Complete fused MoE forward for INT8 W8A8 per-channel quantization.

    Kernel launches:
    1. moe_align_block_size (token sorting, PyTorch ops)
    2. per_token_group_quant_int8 (activation quantization, 1 CUDA kernel)
    3. invoke_fused_moe_int8_kernel (GEMM1: hidden -> gate+up, fused scale in epilogue)
    4. SiLU+mul+quant (1 v2 fused kernel or 3 separate ops)
    5. invoke_fused_moe_int8_kernel (GEMM2: intermediate -> output, fused routing weight)
    6. moe_sum_reduce (combine top-k, weights already applied)

    GLM4 convention: w1 stores [up_proj, gate_proj] (up first, gate second).
    Correct: silu(gate_proj) * up_proj = silu(second_half) * first_half.

    Returns: [M, K] BF16 output
    """
    M, K = hidden_states.shape
    E = w1.shape[0]
    N1 = w1.shape[1]  # 2 * inter_size
    N2 = w2.shape[1]  # hidden_size
    device = hidden_states.device

    # Ensure weight scales have shape [E, N, 1]
    if w1_scale.dim() == 2:
        w1_scale = w1_scale.unsqueeze(-1)
    if w2_scale.dim() == 2:
        w2_scale = w2_scale.unsqueeze(-1)

    # Step 1: moe_align_block_size (no CPU-GPU sync)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids.to(torch.int32), block_size, E
    )

    # Step 2: Quantize activation to INT8
    if _HAS_FUSED_QUANT and hidden_states.is_contiguous():
        a_int8 = torch.empty_like(hidden_states, dtype=torch.int8)
        a_scale = torch.empty(M, 1, device=device, dtype=torch.float32)
        per_token_group_quant_int8(
            hidden_states,
            a_int8,
            a_scale,
            group_size=K,
            eps=1e-12,
            int8_min=-128.0,
            int8_max=127.0,
            scale_ue8m0=False,
        )
    else:
        a_int8, a_scale = per_token_quant_int8_triton(hidden_states)

    # Step 3: GEMM1 (hidden -> gate+up, INT8 with fused scale in epilogue)
    output1 = torch.zeros(M * top_k, N1, device=device, dtype=torch.bfloat16)
    invoke_fused_moe_int8_kernel(
        a_int8,
        w1,
        output1,
        a_scale,
        w1_scale,
        topk_weights.reshape(-1),
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k=top_k,
        mul_routed_weight=False,
        a_is_expanded=False,  # a_int8 is [M, K] (per-token)
    )

    # Step 4: SiLU + mul + INT8 re-quantization
    # GLM4 convention: first half = up_proj, second half = gate_proj
    # Correct: silu(gate_proj) * up_proj = silu(second_half) * first_half
    if _HAS_FUSED_QUANT_V2 and inter_size in (16, 32, 64, 128):
        # Fused: swap halves for v2 kernel, then silu+mul+quant in 1 kernel
        # v2 kernel only supports group_size in {16, 32, 64, 128}.
        # GLM-4.7 has inter_size=12288, so this path is not used for that model.
        swapped = torch.cat(
            [output1[:, inter_size:], output1[:, :inter_size]],  # gate_proj  # up_proj
            dim=-1,
        )  # [M*top_k, 2*inter]
        b_int8 = torch.empty(M * top_k, inter_size, device=device, dtype=torch.int8)
        b_scale = torch.empty(M * top_k, 1, device=device, dtype=torch.float32)
        per_token_group_quant_int8_v2(
            swapped,
            b_int8,
            b_scale,
            group_size=inter_size,
            eps=1e-10,
            int8_min=-128.0,
            int8_max=127.0,
            scale_ue8m0=False,
            fuse_silu_and_mul=True,
            masked_m=None,
        )
        del output1, swapped
    else:
        # Separate ops: silu + mul + quant (3 kernel launches)
        up = output1[:, :inter_size]
        gate = output1[:, inter_size:]
        intermediate = torch.nn.functional.silu(gate) * up  # [M*top_k, inter]
        del output1, up, gate
        if _HAS_FUSED_QUANT and intermediate.is_contiguous():
            b_int8 = torch.empty_like(intermediate, dtype=torch.int8)
            b_scale = torch.empty(
                intermediate.shape[0], 1, device=device, dtype=torch.float32
            )
            per_token_group_quant_int8(
                intermediate,
                b_int8,
                b_scale,
                group_size=inter_size,
                eps=1e-12,
                int8_min=-128.0,
                int8_max=127.0,
                scale_ue8m0=False,
            )
        else:
            b_int8, b_scale = per_token_quant_int8_triton(intermediate)
        del intermediate

    # Step 5: GEMM2 (intermediate -> output, with fused routing weight in epilogue)
    output2 = torch.zeros(M * top_k, N2, device=device, dtype=torch.bfloat16)
    invoke_fused_moe_int8_kernel(
        b_int8,
        w2,
        output2,
        b_scale,
        w2_scale,
        topk_weights.reshape(-1),
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k=top_k,
        mul_routed_weight=True,  # apply routing weight in GEMM2 epilogue
        a_is_expanded=True,  # b_int8 is [M*top_k, inter] (per-token-expert)
    )
    del b_int8, b_scale

    # Step 6: moe_sum_reduce (combine top-k, routing weights already applied)
    output2_reshaped = output2.reshape(M, top_k, N2)
    final_output = torch.empty(M, N2, device=device, dtype=torch.bfloat16)
    moe_sum_reduce_triton(output2_reshaped, final_output, routed_scaling_factor=1.0)

    return final_output


# =============================================================================
# INT8 scaled GEMM with fused bias (for Dense layers)
# Replaces deep_gemm int8_gemm_nt + separate bias add (3 kernels -> 2)
# =============================================================================


@triton.jit
def _int8_scaled_mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """Triton INT8 scaled GEMM with fused per-token/per-channel scale + bias.

    Computes: C[M,N] = (A_int8[M,K] @ B_int8[N,K].T) * a_scale[M] * b_scale[N] + bias[N]
    Output is BF16.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Load B as [BLOCK_SIZE_K, BLOCK_SIZE_N] (transposed) for tl.dot(a[M,K], b[K,N])
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    mask_m = offs_m < M
    mask_n = offs_n < N

    # INT8 GEMM with FP32 accumulation (cast to bf16 for Tensor Core dot)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0)
        else:
            k_mask = offs_k[:, None] < K - k_start
            a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask, other=0)
            b = tl.load(b_ptrs, mask=k_mask & mask_n[None, :], other=0)

        a = a.to(tl.bfloat16)
        b = b.to(tl.bfloat16)
        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply per-token and per-channel scales
    a_scale = tl.load(a_scale_ptr + offs_m, mask=mask_m, other=0.0)
    b_scale = tl.load(b_scale_ptr + offs_n, mask=mask_n, other=0.0)
    accumulator *= a_scale[:, None] * b_scale[None, :]

    # Add bias (fused in epilogue)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        accumulator += bias[None, :]

    # Store as BF16
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, accumulator.to(tl.bfloat16), mask=c_mask)


# Default config for INT8 scaled GEMM
_INT8_GEMM_CONFIG = {
    "BLOCK_SIZE_M": 64,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    "num_warps": 4,
    "num_stages": 2,
}


def int8_scaled_mm_triton(
    a_int8: torch.Tensor,  # [M, K] INT8
    b_int8: torch.Tensor,  # [N, K] INT8 (weight, will be transposed)
    a_scale: torch.Tensor,  # [M] FP32 per-token scale
    b_scale: torch.Tensor,  # [N] FP32 per-channel scale
    bias: Optional[torch.Tensor] = None,  # [N] BF16/FP32 or None
    out: Optional[torch.Tensor] = None,  # [M, N] BF16 or None
) -> torch.Tensor:
    """INT8 scaled matmul with fused bias: C = (A @ B.T) * a_scale * b_scale + bias.

    Replaces deep_gemm int8_gemm_nt + separate bias add (3 kernels -> 2).
    Uses Triton BF16 Tensor Core dot (INT8 inputs cast to BF16 in registers).
    Best for memory-bound cases (small M, e.g. decode) where INT8 weight
    loading bandwidth advantage compensates for BF16 dot compute.

    Args:
        a_int8: [M, K] INT8 activation (already quantized)
        b_int8: [N, K] INT8 weight (will be transposed internally)
        a_scale: [M] or [M, 1] FP32 per-token scale
        b_scale: [N] or [N, 1] FP32 per-channel scale
        bias: [N] optional bias tensor
        out: [M, N] pre-allocated output, or None to allocate
    Returns:
        [M, N] BF16 output
    """
    M, K = a_int8.shape
    N = b_int8.shape[0]
    device = a_int8.device

    if a_scale.ndim > 1:
        a_scale = a_scale.reshape(-1)
    if b_scale.ndim > 1:
        b_scale = b_scale.reshape(-1)

    if out is None:
        out = torch.empty(M, N, device=device, dtype=torch.bfloat16)

    config = _INT8_GEMM_CONFIG
    even_Ks = K % config["BLOCK_SIZE_K"] == 0

    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )

    _int8_scaled_mm_kernel[grid](
        a_int8,
        b_int8,
        out,
        a_scale,
        b_scale,
        bias if bias is not None else a_int8,  # dummy ptr when no bias
        M,
        N,
        K,
        a_int8.stride(0),
        a_int8.stride(1),
        b_int8.stride(0),
        b_int8.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        even_Ks=even_Ks,
        **config,
    )
    return out
