// Copyright 2026 Alibaba Group Holding Limited.
// SPDX-License-Identifier: Apache-2.0
//
// Adapted from SGLang v0.5.14_release:
// python/sglang/jit_kernel/csrc/elementwise/silu_and_mul_masked_post_quant_mxfp4.cuh
// Frozen source SHA256:
// 093f2e2cfd2a63a8bb495760a8ce83a847635355fd0093fa506bc02d6a6995f5
// Modifications: removed TVM/JIT host wrapper and exposed an RTP AOT launcher.
// The device quantization algorithm and PPU instructions remain Apache-2.0.

#include "rtp_llm/models_py/bindings/cuda/kernels/ppu_silu_mul_masked_mxfp4.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

static_assert(sizeof(float) == 4, "bit-trick reciprocal assumes IEEE-754 binary32");

namespace rtp_llm {
namespace {

constexpr int kGroupSize = 32;
constexpr int kWarpThreads = 32;
constexpr int kElemPerThread = 8;
constexpr int kThreadsPerGroup = kGroupSize / kElemPerThread;
constexpr float kQuantMax = 6.0f;
constexpr float kAbsmaxFloor = 1e-10f;
constexpr int kStages = 2;
constexpr int kBlocksYZTarget = 2048;

// Upstream makes kBlockN / kStages / kSwizzle template parameters and the SGLang
// runner instantiates SiluMulMxfp4EP<256, true>.  That single instantiation is
// the one whose payload and scale bytes were verified bit-identical against our
// unmasked kernel at production geometry, so it is the only one built here --
// a wider block would be a silently different arm, not a free option.
constexpr int kBlockN = kPpuSiluMulMaskedMxfp4BlockN;

struct alignas(16) SiluMulMaskedMxfp4Params {
  const __nv_bfloat16* __restrict__ input;
  uint8_t* __restrict__ output;
  uint8_t* __restrict__ output_scale;
  const int32_t* __restrict__ masked_m;
  int64_t stride_input_e;
  int64_t stride_input_t;
  int64_t stride_output_e;
  int64_t stride_output_t;
  int64_t stride_scale_e_bytes;
  int64_t stride_scale_p_bytes;
  int64_t stride_scale_t_bytes;
  int32_t N;         // actual H (may not be multiple of kBlockN)
  int32_t N_padded;  // ceil to kBlockN (for grid sizing)
  float swiglu_limit;
};

__device__ __forceinline__ uint32_t
pack_4xe2m1x2(float q0, float q1, float q2, float q3, float q4, float q5, float q6, float q7) {
  uint32_t packed;
  asm volatile(
      "{\n\t"
      ".reg .b8 r0, r1, r2, r3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r0, %2, %1;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r1, %4, %3;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r2, %6, %5;\n\t"
      "cvt.rn.satfinite.e2m1x2.f32 r3, %8, %7;\n\t"
      "mov.b32 %0, {r0, r1, r2, r3};\n\t"
      "}\n"
      : "=r"(packed)
      : "f"(q0), "f"(q1), "f"(q2), "f"(q3), "f"(q4), "f"(q5), "f"(q6), "f"(q7));
  return packed;
}

template <bool kApplySwigluLimit>
__device__ __forceinline__ __nv_bfloat162
silu_and_mul(__nv_bfloat162 gate, __nv_bfloat162 up, float swiglu_limit, float* fp32_prod = nullptr) {
  if constexpr (kApplySwigluLimit) {
    // clamp in bf16 to match upstream sglang / DeepGEMM
    const __nv_bfloat16 lim = __float2bfloat16_rn(swiglu_limit);
    const __nv_bfloat16 nlim = __float2bfloat16_rn(-swiglu_limit);
    const __nv_bfloat162 lim2 = __halves2bfloat162(lim, lim);
    const __nv_bfloat162 nlim2 = __halves2bfloat162(nlim, nlim);
    gate = __hmin2(gate, lim2);
    up = __hmin2(__hmax2(up, nlim2), lim2);
    const float g0 = __bfloat162float(__low2bfloat16(gate));
    const float g1 = __bfloat162float(__high2bfloat16(gate));
    const float u0 = __bfloat162float(__low2bfloat16(up));
    const float u1 = __bfloat162float(__high2bfloat16(up));
    const float silu0 = g0 * __ppu_sgmdf(g0);
    const float silu1 = g1 * __ppu_sgmdf(g1);
    const float p0 = u0 * silu0;
    const float p1 = u1 * silu1;
    if (fp32_prod) {
      fp32_prod[0] = p0;
      fp32_prod[1] = p1;
    }
    return __floats2bfloat162_rn(p0, p1);
  }
  const float g0 = __bfloat162float(__low2bfloat16(gate));
  const float g1 = __bfloat162float(__high2bfloat16(gate));
  const float silu0 = g0 * __ppu_sgmdf(g0);
  const float silu1 = g1 * __ppu_sgmdf(g1);
  const __nv_bfloat162 silu = __floats2bfloat162_rn(silu0, silu1);
  return __hmul2(up, silu);
}

// Upstream hard-codes the two shuffle offsets for kThreadsPerGroup == 4; the
// loop form below unrolls to exactly those two shuffles and matches the
// unmasked sibling kernel.
__device__ __forceinline__ float group_reduce_max(float v) {
#pragma unroll
  for (int offset = kThreadsPerGroup >> 1; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
  }
  return v;
}

__device__ __forceinline__ void e8m0_scale(float absmax, float& quant_scale, uint8_t& scale_byte) {
  constexpr float kQuantMaxRcp = 1.0f / kQuantMax;
  const float dequant_scale = absmax * kQuantMaxRcp;
  const uint32_t ds_u32 = __float_as_uint(dequant_scale);
  const uint32_t ds_e8m0 = (ds_u32 + 0x007FFFFFu) & 0x7F800000u;
  quant_scale = __uint_as_float(0x7F000000u - ds_e8m0);
  scale_byte = static_cast<uint8_t>(ds_e8m0 >> 23);
}

__device__ __forceinline__ void cp_async_16B(void* smem_dst, const void* gmem_src) {
  const uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_src));
}

__device__ __forceinline__ void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

template <bool kApplySwigluLimit, bool kIsFullBlock>
__global__ __launch_bounds__(kBlockN / kElemPerThread, 1) void silu_and_mul_masked_post_quant_mxfp4_kernel(
    const SiluMulMaskedMxfp4Params __grid_constant__ params) {
  constexpr int kThreadsPerBlock = kBlockN / kElemPerThread;
  constexpr int kPairsPerThread = kElemPerThread / 2;
  static_assert(kBlockN % kGroupSize == 0, "kBlockN must be multiple of 32");
  static_assert(kElemPerThread % 2 == 0, "bf16x2 path needs even elements per thread");
  static_assert(kThreadsPerBlock <= kWarpThreads * 32, "block would exceed the launch limit");

  __shared__ alignas(16) __nv_bfloat16 smem_gate[kStages][kThreadsPerBlock][kElemPerThread];
  __shared__ alignas(16) __nv_bfloat16 smem_up[kStages][kThreadsPerBlock][kElemPerThread];

  // kSwizzle == true in the measured arm: expert on x, hidden on z.
  const int expert_id = blockIdx.x;
  const int token_block_id = blockIdx.y;
  const int hidden_block = blockIdx.z;
  const int tid = threadIdx.x;

  const int n_tokens = __ldg(params.masked_m + expert_id);
  if (n_tokens == 0) return;

  const int n_offset_elem = hidden_block * kBlockN;
  if (n_offset_elem >= params.N_padded) return;

  const int group_in_block = tid / kThreadsPerGroup;
  const int lane_in_group = tid % kThreadsPerGroup;
  const int thread_n_offset = n_offset_elem + group_in_block * kGroupSize + lane_in_group * kElemPerThread;

  const int group_index_in_full = (n_offset_elem / kGroupSize) + group_in_block;
  const int scale_pair_idx = group_index_in_full >> 1;
  const int scale_lo_hi = group_index_in_full & 1;

  const __nv_bfloat16* in_e = params.input + expert_id * params.stride_input_e;
  uint8_t* out_e = params.output + expert_id * params.stride_output_e;
  uint8_t* scl_e = params.output_scale + expert_id * params.stride_scale_e_bytes;

  const int token_stride = gridDim.y;

  auto issue_token_load = [&](int stage, int t) {
    if (t < n_tokens) {
      const __nv_bfloat16* row_base = in_e + t * params.stride_input_t;
      if constexpr (kIsFullBlock) {
        const __nv_bfloat16* gate_ptr = row_base + thread_n_offset;
        const __nv_bfloat16* up_ptr = row_base + params.N + thread_n_offset;
        cp_async_16B(&smem_gate[stage][tid][0], gate_ptr);
        cp_async_16B(&smem_up[stage][tid][0], up_ptr);
      } else {
#pragma unroll
        for (int i = 0; i < kElemPerThread; ++i) {
          const int elem_idx = thread_n_offset + i;
          if (elem_idx < params.N) {
            smem_gate[stage][tid][i] = row_base[elem_idx];
            smem_up[stage][tid][i] = row_base[params.N + elem_idx];
          } else {
            smem_gate[stage][tid][i] = __float2bfloat16_rn(0.0f);
            smem_up[stage][tid][i] = __float2bfloat16_rn(0.0f);
          }
        }
      }
    } else {
      if constexpr (!kIsFullBlock) {
#pragma unroll
        for (int i = 0; i < kElemPerThread; ++i) {
          smem_gate[stage][tid][i] = __float2bfloat16_rn(0.0f);
          smem_up[stage][tid][i] = __float2bfloat16_rn(0.0f);
        }
      }
    }
    cp_async_commit_group();
  };

  int issue_stage = 0;
  int t_load = token_block_id;
#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    issue_token_load(issue_stage, t_load);
    issue_stage = (issue_stage + 1) % kStages;
    t_load += token_stride;
  }

  int compute_stage = 0;
  for (int t = token_block_id; t < n_tokens; t += token_stride) {
    issue_token_load(issue_stage, t_load);
    issue_stage = (issue_stage + 1) % kStages;
    t_load += token_stride;

    cp_async_wait_group<kStages - 1>();
    // Every thread only reads its own smem slot, so with one warp per block this
    // barrier is not load-bearing -- it is kept because the measured arm had it.
    __syncthreads();

    auto* gate_pairs = reinterpret_cast<__nv_bfloat162*>(&smem_gate[compute_stage][tid][0]);
    auto* up_pairs = reinterpret_cast<__nv_bfloat162*>(&smem_up[compute_stage][tid][0]);

    alignas(16) __nv_bfloat16 prod_bf16[kElemPerThread];
    auto* prod_pairs = reinterpret_cast<__nv_bfloat162*>(prod_bf16);

    float local_absmax;
    float fp32_prods[kPairsPerThread][2];
    if constexpr (kApplySwigluLimit) {
      local_absmax = 0.0f;
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        prod_pairs[i] = silu_and_mul<kApplySwigluLimit>(gate_pairs[i], up_pairs[i], params.swiglu_limit, fp32_prods[i]);
        local_absmax = fmaxf(local_absmax, fmaxf(fabsf(fp32_prods[i][0]), fabsf(fp32_prods[i][1])));
      }
    } else {
      __nv_bfloat162 absmax_v2 = __float2bfloat162_rn(0.0f);
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        prod_pairs[i] = silu_and_mul<kApplySwigluLimit>(gate_pairs[i], up_pairs[i], params.swiglu_limit);
        absmax_v2 = __hmax2(absmax_v2, __habs2(prod_pairs[i]));
      }
      local_absmax = fmaxf(__bfloat162float(__low2bfloat16(absmax_v2)), __bfloat162float(__high2bfloat16(absmax_v2)));
    }

    local_absmax = fmaxf(group_reduce_max(local_absmax), kAbsmaxFloor);

    float quant_scale;
    uint8_t scale_byte;
    e8m0_scale(local_absmax, quant_scale, scale_byte);

    float q[kElemPerThread];
    if constexpr (kApplySwigluLimit) {
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        q[2 * i] = fp32_prods[i][0] * quant_scale;
        q[2 * i + 1] = fp32_prods[i][1] * quant_scale;
      }
    } else {
      const __nv_bfloat162 quant_scale_v2 = __float2bfloat162_rn(quant_scale);
      __nv_bfloat162 q_pairs_scaled[kPairsPerThread];
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        q_pairs_scaled[i] = __hmul2(prod_pairs[i], quant_scale_v2);
      }
#pragma unroll
      for (int i = 0; i < kPairsPerThread; ++i) {
        q[2 * i] = __bfloat162float(__low2bfloat16(q_pairs_scaled[i]));
        q[2 * i + 1] = __bfloat162float(__high2bfloat16(q_pairs_scaled[i]));
      }
    }
    const uint32_t packed = pack_4xe2m1x2(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);

    if constexpr (kIsFullBlock) {
      const int out_byte_offset = thread_n_offset / 2;
      *reinterpret_cast<uint32_t*>(out_e + t * params.stride_output_t + out_byte_offset) = packed;
    } else {
      const int valid_elems =
          (thread_n_offset >= params.N)
              ? 0
              : ((thread_n_offset + kElemPerThread <= params.N) ? kElemPerThread : (params.N - thread_n_offset));
      if (valid_elems == kElemPerThread) {
        const int out_byte_offset = thread_n_offset / 2;
        *reinterpret_cast<uint32_t*>(out_e + t * params.stride_output_t + out_byte_offset) = packed;
      } else if (valid_elems > 0) {
        // valid_elems is always even: N is even (host check) and
        // thread_n_offset is a multiple of kElemPerThread (8).
        const int valid_bytes = valid_elems / 2;
        uint8_t* byte_out = out_e + t * params.stride_output_t + thread_n_offset / 2;
        const uint8_t* packed_bytes = reinterpret_cast<const uint8_t*>(&packed);
        for (int b = 0; b < valid_bytes; ++b) {
          byte_out[b] = packed_bytes[b];
        }
      }
    }

    if (lane_in_group == 0) {
      uint8_t byte;
      if constexpr (kIsFullBlock) {
        byte = scale_byte;
      } else {
        byte = (group_index_in_full * kGroupSize < params.N) ? scale_byte : 0;
      }
      scl_e[scale_pair_idx * params.stride_scale_p_bytes + t * params.stride_scale_t_bytes + scale_lo_hi] = byte;
    }

    compute_stage = (compute_stage + 1) % kStages;
  }
}

template <bool kApplySwigluLimit>
void launchPpuSiluMulMaskedMxfp4(const SiluMulMaskedMxfp4Params& params,
                                 int num_experts,
                                 int capacity,
                                 int max_masked_m,
                                 cudaStream_t stream) {
    // Upstream's amortization: spread kBlocksYZTarget blocks over the experts,
    // then stop handing an expert more token blocks than it can have rows.  The
    // cap is a hint only; the token loop below strides, so an under-estimate
    // costs occupancy, never correctness.
    int blocks_per_expert = (kBlocksYZTarget + num_experts - 1) / num_experts;
    const int amort_cap = max_masked_m > 0 ? max_masked_m : capacity;
    blocks_per_expert = std::max(1, std::min(blocks_per_expert, amort_cap));

    const int hidden_blocks = params.N_padded / kBlockN;
    const dim3 grid(num_experts, blocks_per_expert, hidden_blocks);
    constexpr int threads = kBlockN / kElemPerThread;
    if (params.N == params.N_padded) {
        silu_and_mul_masked_post_quant_mxfp4_kernel<kApplySwigluLimit, true>
            <<<grid, threads, 0, stream>>>(params);
    } else {
        silu_and_mul_masked_post_quant_mxfp4_kernel<kApplySwigluLimit, false>
            <<<grid, threads, 0, stream>>>(params);
    }
}

}  // namespace

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
                                 cudaStream_t stream) {
    const int hidden_padded = (hidden_size + kBlockN - 1) / kBlockN * kBlockN;
    const SiluMulMaskedMxfp4Params params{
        static_cast<const __nv_bfloat16*>(input),
        output,
        scale,
        masked_m,
        stride_input_e,
        stride_input_t,
        stride_output_e,
        stride_output_t,
        stride_scale_e_bytes,
        stride_scale_p_bytes,
        stride_scale_t_bytes,
        hidden_size,
        hidden_padded,
        swiglu_limit,
    };
    if (apply_swiglu_limit) {
        launchPpuSiluMulMaskedMxfp4<true>(params, num_experts, capacity, max_masked_m, stream);
    } else {
        launchPpuSiluMulMaskedMxfp4<false>(params, num_experts, capacity, max_masked_m, stream);
    }
}

}  // namespace rtp_llm
