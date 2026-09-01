# Copyright 2026 Alibaba Group Holding Limited.
# SPDX-License-Identifier: Apache-2.0
"""Source contract for the masked PPU SwiGLU+MXFP4 kernel.

The measured evidence for this kernel (bit-identical payload and scale bytes,
18.0 us at capacity 3072) was collected on exactly one instantiation --
kBlockN=256, kStages=2, kSwizzle=true -- loaded from a .so that SGLang's JIT
built.  Bringing it in-tree as AOT source is what removes the out-of-repo
binary; these assertions are what keep the in-tree copy on the arm that was
actually measured, and keep the JIT loader from creeping back.
"""
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[5]
CUDA = ROOT / "rtp_llm/models_py/bindings/cuda"
KERNEL = CUDA / "kernels/ppu_silu_mul_masked_mxfp4.cu"
HEADER = CUDA / "kernels/ppu_silu_mul_masked_mxfp4.h"
NOTICE = CUDA / "kernels/ppu_silu_mul_masked_mxfp4.NOTICE"
OP = CUDA / "PpuSiluMulMaskedMxfp4Op.cc"
REGISTER = CUDA / "RegisterBaseBindings.hpp"
KERNEL_BUILD = CUDA / "kernels/BUILD"
CUDA_BUILD = CUDA / "BUILD"

class PpuSiluMulMaskedMxfp4SourceTest(unittest.TestCase):
    def test_frozen_apache_provenance(self):
        text, notice = KERNEL.read_text(), NOTICE.read_text()
        sha = "093f2e2cfd2a63a8bb495760a8ce83a847635355fd0093fa506bc02d6a6995f5"
        self.assertIn("SPDX-License-Identifier: Apache-2.0", text)
        self.assertIn(sha, text)
        self.assertIn(sha, notice)
        self.assertIn("SGLang v0.5.14_release", notice)
        self.assertIn("silu_and_mul_masked_post_quant_mxfp4.cuh", notice)

    def test_ppu_native_aot_and_stream_contract(self):
        text, op = KERNEL.read_text(), OP.read_text()
        self.assertIn("__ppu_sgmdf", text)
        self.assertIn("cvt.rn.satfinite.e2m1x2.f32", text)
        self.assertIn("cp.async.cg.shared.global", text)
        self.assertIn("const SiluMulMaskedMxfp4Params params", text)
        self.assertIn("GET_CURRENT_STREAM()", op)
        # An out-of-repo .so loaded at runtime is exactly what this port exists
        # to delete; a JIT loader creeping back would make the wheel a lie.
        # Comments are stripped first: the provenance header has to name the
        # upstream JIT path, and test_frozen_apache_provenance requires it.
        code = "\n".join(line.split("//")[0]
                         for line in (text + "\n" + op).splitlines())
        for banned in ("tvm_ffi", "tvm::ffi", "load_module", "dlopen",
                       ".so", "sglang", "jit"):
            self.assertNotIn(banned.lower(), code.lower())

    def test_pinned_to_the_measured_instantiation(self):
        text, header = KERNEL.read_text(), HEADER.read_text()
        # kBlockN lives in the header so the op can size the scale buffer from
        # the same number; a second literal here is how the two drift apart.
        self.assertIn("constexpr int kPpuSiluMulMaskedMxfp4BlockN = 256;", header)
        self.assertIn("constexpr int kBlockN = kPpuSiluMulMaskedMxfp4BlockN;", text)
        self.assertNotIn("kBlockN = 256", text)
        self.assertIn("constexpr int kStages = 2;", text)
        self.assertIn("constexpr int kElemPerThread = 8;", text)
        # kSwizzle == true means expert on x and hidden on z; swapping them
        # silently changes which arm runs.
        self.assertIn("const int expert_id = blockIdx.x;", text)
        self.assertIn("const int hidden_block = blockIdx.z;", text)
        self.assertIn("const dim3 grid(num_experts, blocks_per_expert, hidden_blocks)", text)
        # Upstream leaves kBlockN/kStages/kSwizzle as template parameters; the
        # port must not reintroduce untested widths behind a dispatch.
        self.assertNotIn("hidden_size % 512", text)
        self.assertNotIn("dispatchBlock", text)

    def test_masked_rows_are_the_only_rows_touched(self):
        text = KERNEL.read_text()
        self.assertIn("const int n_tokens = __ldg(params.masked_m + expert_id);", text)
        self.assertIn("if (n_tokens == 0) return;", text)
        # Every payload and scale write is indexed by `t`, and `t` only ever
        # comes from this loop, so rows at or past masked_m stay untouched.
        self.assertIn("for (int t = token_block_id; t < n_tokens; t += token_stride)", text)
        self.assertNotIn("for (int t = 0;", text)
        for write in ("*reinterpret_cast<uint32_t*>(out_e + t * params.stride_output_t",
                      "scl_e[scale_pair_idx * params.stride_scale_p_bytes + t *"):
            self.assertIn(write, text)
        self.assertNotIn("out_e[", text)

    def test_row_cap_is_an_occupancy_hint_not_a_bound(self):
        text, op = KERNEL.read_text(), OP.read_text()
        # Production passes an expected_m far below the measured peak; if the
        # cap ever became a bound, rows would be silently dropped.
        self.assertIn("const int amort_cap = max_masked_m > 0 ? max_masked_m : capacity;", text)
        self.assertIn("blocks_per_expert = std::max(1, std::min(blocks_per_expert, amort_cap));", text)
        self.assertIn("std::min<int64_t>(max_masked_m, capacity)", op)

    def test_non_ppu_fail_closed(self):
        op, register = OP.read_text(), REGISTER.read_text()
        kernel_build, cuda_build = KERNEL_BUILD.read_text(), CUDA_BUILD.read_text()
        self.assertIn("#ifdef USE_PPU", op)
        self.assertIn("PpuSiluAndMulMaskedPostQuantMxfp4", register)
        self.assertIn('py::arg("masked_m")', register)
        self.assertIn('py::arg("swiglu_limit") = py::none()', register)
        self.assertIn('py::arg("max_masked_m") = 0', register)
        self.assertIn('name = "ppu_silu_mul_masked_mxfp4"', kernel_build)
        self.assertIn('"@platforms//:incompatible"', kernel_build)
        self.assertIn("kernels:ppu_silu_mul_masked_mxfp4", cuda_build)
        self.assertNotIn("fallback", op.lower())

    def test_wrapper_fails_closed_on_dtype_shape_and_layout(self):
        op = OP.read_text()
        for check in ("gate_up.is_cuda()", "torch::kBFloat16", "gate_up.dim() == 3",
                      "gate_up.stride(2) == 1", "gate_up.stride(1) % 8 == 0",
                      "masked_m.scalar_type() == torch::kInt32",
                      "masked_m.size(0) == num_experts",
                      "two_hidden > 0 && two_hidden % 4 == 0"):
            self.assertIn(check, op)
        for banned in ("torch.cuda.synchronize", "cudaDeviceSynchronize",
                       ".cpu(", ".item(", ".tolist("):
            self.assertNotIn(banned.lower(), op.lower())

    def test_scale_layout_is_handed_back_transposed_for_free(self):
        op = OP.read_text()
        for token in ("torch::kUInt8", "torch::kUInt16", "hidden / 2",
                      "hidden_padded / 64", "(hidden + 63) / 64",
                      ".slice(1, 0, scale_valid).transpose(1, 2)"):
            self.assertIn(token, op)
        # The whole point of the masked kernel's native [E, S, M] layout is that
        # the caller stops paying for a contiguous transpose copy.
        self.assertNotIn(".contiguous()", op)
        self.assertNotIn("permute", op)

if __name__ == "__main__":
    unittest.main()
