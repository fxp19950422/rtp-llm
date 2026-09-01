// Copyright 2026 Alibaba Group Holding Limited.
// SPDX-License-Identifier: Apache-2.0
#include "rtp_llm/models_py/bindings/cuda/PpuSiluMulMaskedMxfp4Op.h"
#ifdef USE_PPU
#include "rtp_llm/models_py/bindings/common/Torch_ext.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/ppu_silu_mul_masked_mxfp4.h"
#include <limits>
namespace rtp_llm {
std::tuple<torch::Tensor, torch::Tensor> PpuSiluAndMulMaskedPostQuantMxfp4(
    torch::Tensor gate_up,
    torch::Tensor masked_m,
    double        swiglu_limit,
    bool          apply_swiglu_limit,
    int64_t       max_masked_m) {
    TORCH_CHECK(gate_up.is_cuda(), "gate_up must be a PPU tensor");
    TORCH_CHECK(gate_up.scalar_type() == torch::kBFloat16, "gate_up must be BF16");
    TORCH_CHECK(gate_up.dim() == 3, "gate_up must have shape [E, capacity, 2H]");
    // The kernel reads the expert and token strides, so a sliced view is fine,
    // but the vectorized load needs 16B-aligned rows and a packed last dim.
    TORCH_CHECK(gate_up.stride(2) == 1, "gate_up last dim must be contiguous");
    TORCH_CHECK(gate_up.stride(1) % 8 == 0, "gate_up rows must be 16B aligned");
    TORCH_CHECK(masked_m.is_cuda() && masked_m.scalar_type() == torch::kInt32,
                "masked_m must be an int32 PPU tensor");
    TORCH_CHECK(masked_m.dim() == 1 && masked_m.is_contiguous(),
                "masked_m must be a contiguous 1-D tensor");
    const int64_t num_experts = gate_up.size(0);
    const int64_t capacity    = gate_up.size(1);
    const int64_t two_hidden  = gate_up.size(2);
    TORCH_CHECK(masked_m.size(0) == num_experts,
                "masked_m must have one entry per expert");
    TORCH_CHECK(two_hidden > 0 && two_hidden % 4 == 0,
                "2H must be positive and H must be even");
    const int64_t hidden = two_hidden / 2;
    TORCH_CHECK(hidden <= std::numeric_limits<int>::max() &&
                    capacity <= std::numeric_limits<int>::max() &&
                    num_experts <= std::numeric_limits<int>::max(),
                "shape exceeds PPU launcher limits");
    constexpr int64_t block_n = kPpuSiluMulMaskedMxfp4BlockN;
    const int64_t hidden_padded = (hidden + block_n - 1) / block_n * block_n;
    // The kernel emits scale bytes for the padded width; only the leading
    // groups that cover real hidden elements are handed back.
    const int64_t scale_alloc = hidden_padded / 64;
    const int64_t scale_valid = (hidden + 63) / 64;
    auto packed = torch::empty({num_experts, capacity, hidden / 2},
                               gate_up.options().dtype(torch::kUInt8));
    auto scale_storage = torch::empty({num_experts, scale_alloc, capacity},
                                      gate_up.options().dtype(torch::kUInt16));
    if (num_experts != 0 && capacity != 0) {
        StreamType stream = GET_CURRENT_STREAM();
        invokePpuSiluMulMaskedMxfp4(
            gate_up.data_ptr(),
            packed.data_ptr<uint8_t>(),
            reinterpret_cast<uint8_t*>(scale_storage.data_ptr<uint16_t>()),
            masked_m.data_ptr<int32_t>(),
            gate_up.stride(0),
            gate_up.stride(1),
            packed.stride(0) * packed.element_size(),
            packed.stride(1) * packed.element_size(),
            scale_storage.stride(0) * scale_storage.element_size(),
            scale_storage.stride(1) * scale_storage.element_size(),
            scale_storage.stride(2) * scale_storage.element_size(),
            static_cast<int>(num_experts),
            static_cast<int>(hidden),
            static_cast<int>(capacity),
            static_cast<int>(std::min<int64_t>(max_masked_m, capacity)),
            apply_swiglu_limit,
            static_cast<float>(swiglu_limit),
            stream);
    }
    // [E, S, capacity] is the kernel's native layout; the transpose below hands
    // back the mn-major [E, capacity, S] the masked GEMM wants, for free.
    auto scale = scale_storage.slice(1, 0, scale_valid).transpose(1, 2);
    return {packed, scale};
}
}  // namespace rtp_llm
#endif
