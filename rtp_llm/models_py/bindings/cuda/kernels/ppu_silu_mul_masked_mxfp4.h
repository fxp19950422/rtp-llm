// Copyright 2026 Alibaba Group Holding Limited.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
namespace rtp_llm {
// The only instantiated block width.  Exposed because the caller sizes the
// scale buffer from it: the kernel writes one e8m0 byte per 32 hidden
// elements of the padded width, not of the real width.
constexpr int kPpuSiluMulMaskedMxfp4BlockN = 256;
// Masked variant of invokePpuSiluMulMxfp4: the input is the grouped
// [num_experts, capacity, 2 * hidden_size] output of a masked grouped GEMM and
// only the first masked_m[e] rows of each expert are computed.  Rows past that
// count are left untouched, so the caller must not read them.
//
// `masked_m` is a device int32 buffer of `num_experts` entries.
// Input strides count BF16 elements; output and scale strides count bytes.
// `max_masked_m` is an occupancy hint only -- passing a value below the real
// row count still computes every row, because the kernel strides over tokens.
void invokePpuSiluMulMaskedMxfp4(const void* input,
                                 uint8_t*    output,
                                 uint8_t*    scale,
                                 const int32_t* masked_m,
                                 int64_t     stride_input_e,
                                 int64_t     stride_input_t,
                                 int64_t     stride_output_e,
                                 int64_t     stride_output_t,
                                 int64_t     stride_scale_e_bytes,
                                 int64_t     stride_scale_p_bytes,
                                 int64_t     stride_scale_t_bytes,
                                 int         num_experts,
                                 int         hidden_size,
                                 int         capacity,
                                 int         max_masked_m,
                                 bool        apply_swiglu_limit,
                                 float       swiglu_limit,
                                 cudaStream_t stream);
}  // namespace rtp_llm
