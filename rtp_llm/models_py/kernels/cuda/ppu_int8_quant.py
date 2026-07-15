"""PPU-optimized dynamic per-token INT8 quantization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _per_token_quant_int8_kernel(
    input_ptr,
    output_q_ptr,
    output_s_ptr,
    stride_input,
    stride_output,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    values = tl.load(input_ptr + row * stride_input + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    absmax = tl.maximum(tl.max(tl.abs(values)), 1e-10)
    scale = absmax / 127.0
    quantized = tl.extra.cuda.libdevice.round(values / scale).to(tl.int8)
    tl.store(output_q_ptr + row * stride_output + offsets, quantized, mask=mask)
    tl.store(output_s_ptr + row, scale)


def per_token_quant_int8_triton(
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize each contiguous input row with a dynamic FP32 scale."""
    if input_tensor.ndim < 2:
        raise ValueError("input_tensor must have at least two dimensions")
    if not input_tensor.is_contiguous():
        raise ValueError("input_tensor must be contiguous")

    hidden_size = input_tensor.shape[-1]
    rows = input_tensor.numel() // hidden_size
    output_q = torch.empty_like(input_tensor, dtype=torch.int8)
    output_s = torch.empty(
        *input_tensor.shape[:-1], 1, device=input_tensor.device, dtype=torch.float32
    )
    block_size = triton.next_power_of_2(hidden_size)
    num_warps = min(max(block_size // 256, 1), 8)
    _per_token_quant_int8_kernel[(rows,)](
        input_tensor,
        output_q,
        output_s,
        input_tensor.stride(-2),
        output_q.stride(-2),
        hidden_size,
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
        num_stages=1,
    )
    return output_q, output_s
