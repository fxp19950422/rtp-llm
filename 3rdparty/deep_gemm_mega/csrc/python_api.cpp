#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <hggc_runtime.h>

#include "apis/mega_moe.hpp"
#include "apis/runtime.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    deep_gemm_mega::mega::register_apis(m);
    deep_gemm_mega::runtime::register_apis(m);
    m.def("preprocess_mxfp4_scales",
          &deep_gemm_mega_fp4_common::preprocess_mxfp4_scales);
    m.def("preprocess_mxfp4_weight_for_act_and_quant_fusing",
          &deep_gemm_mega_fp4_common::preprocess_mxfp4_weight_for_act_and_quant_fusing);
}
