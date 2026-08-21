#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_base.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace py = pybind11;
namespace rtp_llm {

namespace {

NumericalStatusScope resolveNumericalStatusScope(const py::object& py_instance) {
    PyObject* raw_value = PyObject_GetAttrString(py_instance.ptr(), "numerical_status_scope");
    if (raw_value == nullptr) {
        if (PyErr_ExceptionMatches(PyExc_AttributeError)) {
            PyErr_Clear();
            return NumericalStatusScope::NONE;
        }
        throw py::error_already_set();
    }
    py::object value = py::reinterpret_steal<py::object>(raw_value);
    if (PyCallable_Check(value.ptr())) {
        value = value();
    }
    if (!py::isinstance<py::str>(value)) {
        return value.cast<NumericalStatusScope>();
    }
    const auto scope = value.cast<std::string>();
    if (scope == "none") {
        return NumericalStatusScope::NONE;
    }
    if (scope == "batch") {
        return NumericalStatusScope::BATCH;
    }
    if (scope == "origin_row") {
        return NumericalStatusScope::ORIGIN_ROW;
    }
    throw std::invalid_argument("unknown numerical_status_scope: " + scope);
}

}  // namespace

// Single wrapper for both prefill and decode tests; init_prefill / init_decode
// build GraphParams and call CudaGraphRunner factory methods.
// Plain pybind11 class (no torch::jit::CustomClassHolder) so the module loads without
// depending on torch's registered CustomClassHolder type.
class CudaGraphTestRunner {
public:
    void init_prefill(py::object               py_instance,
                      int64_t                  max_context_batch_size,
                      int64_t                  max_seq_len,
                      int64_t                  tokens_per_block,
                      int64_t                  kernel_tokens_per_block,
                      std::vector<int>         prefill_capture_seq_lens,
                      int64_t                  hidden_size,
                      std::vector<std::string> group_tags,
                      int64_t                  num_tokens_per_bs,
                      int64_t                  hc_mult) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = true;
        params.is_prefill_cuda_graph_mode   = true;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.tokens_per_block             = static_cast<int>(tokens_per_block);
        params.kernel_tokens_per_block      = static_cast<int>(kernel_tokens_per_block);
        params.num_tokens_per_bs =
            static_cast<int>(num_tokens_per_bs > 0 ? num_tokens_per_bs : max_seq_len);
        params.max_context_batch_size       = static_cast<size_t>(max_context_batch_size);
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.input_hidden_size            = static_cast<size_t>(hidden_size * hc_mult);
        params.hc_mult                      = hc_mult;
        params.model_data_type              = c10::ScalarType::BFloat16;
        params.prefill_capture_seq_lens     = std::move(prefill_capture_seq_lens);
        params.kv_cache_group_tags          = std::move(group_tags);
        params.numerical_status_scope       = resolveNumericalStatusScope(py_instance);

        runner_ = CudaGraphRunner::createForPrefill(std::move(py_instance), std::move(params));
    }

    void init_decode(py::object               py_instance,
                     int64_t                  hidden_size,
                     int64_t                  max_seq_len,
                     int64_t                  tokens_per_block,
                     int64_t                  kernel_tokens_per_block,
                     std::vector<int>         decode_capture_batch_sizes,
                     std::vector<std::string> group_tags,
                     bool                     is_target_verify,
                     int64_t                  num_tokens_per_bs) {
        reset_runner();
        GraphParams params;
        params.enable_cuda_graph_debug_mode = false;
        params.is_prefill_cuda_graph_mode   = false;
        params.max_seq_len                  = static_cast<int>(max_seq_len);
        params.tokens_per_block             = static_cast<int>(tokens_per_block);
        params.kernel_tokens_per_block      = static_cast<int>(kernel_tokens_per_block);
        params.input_hidden_size            = static_cast<size_t>(hidden_size);
        params.num_tokens_per_bs            = static_cast<int>(num_tokens_per_bs);
        params.hidden_size                  = static_cast<size_t>(hidden_size);
        params.model_data_type              = c10::ScalarType::BFloat16;
        params.max_context_batch_size       = 128;
        params.decode_capture_batch_sizes   = std::move(decode_capture_batch_sizes);
        params.kv_cache_group_tags          = std::move(group_tags);
        params.is_target_verify             = is_target_verify;
        params.numerical_status_scope       = resolveNumericalStatusScope(py_instance);

        runner_ = CudaGraphRunner::createForDecode(std::move(py_instance), std::move(params));
    }

    bool canRun(torch_ext::PyModelInputs& inputs) {
        return runner_ != nullptr && runner_->canRun(inputs, state_);
    }

    torch_ext::PyModelOutputs forward(torch_ext::PyModelInputs& inputs) {
        // Production PyWrappedModel creates these device mirrors. Python tests
        // cannot assign them because the bindings intentionally expose them as
        // read-only, so reproduce that input-building step in the test wrapper.
        inputs.attention_inputs.input_lengths_device  = inputs.attention_inputs.input_lengths.cuda();
        inputs.attention_inputs.prefix_lengths_device = inputs.attention_inputs.prefix_lengths.cuda();
        refreshTaggedAttentionInputs(inputs);
        return runner_->forward(inputs, state_);
    }

    torch_ext::PyModelOutputs forwardWithNumericalStatusSnapshot(torch_ext::PyModelInputs& inputs) {
        auto outputs = forward(inputs);
        if (!outputs.numerical_status.defined()) {
            return outputs;
        }
        auto snapshot = torch::empty_like(outputs.numerical_status.values);
        snapshot.copy_(outputs.numerical_status.values, /*non_blocking=*/true);
        runner_->recordNumericalStatusSourceFence(state_);
        outputs.numerical_status.values = std::move(snapshot);
        return outputs;
    }

    int getCurrentRealGraphSize() {
        return runner_ != nullptr ? runner_->getCurrentRealGraphBs(state_) : 0;
    }

    ~CudaGraphTestRunner() {
        reset_runner();
    }

private:
    void reset_runner() {
        if (runner_ != nullptr) {
            delete runner_;
            runner_ = nullptr;
        }
    }

    CudaGraphRunner* runner_ = nullptr;
    CudaGraphState   state_{};
};

}  // namespace rtp_llm

PYBIND11_MODULE(libtest_cuda_graph_runner, m) {
    using namespace rtp_llm;
    py::class_<CudaGraphTestRunner>(m, "CudaGraphRunner")
        .def(py::init<>())
        .def("init_prefill",
             &CudaGraphTestRunner::init_prefill,
             py::arg("py_instance"),
             py::arg("max_context_batch_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("kernel_tokens_per_block"),
             py::arg("prefill_capture_seq_lens"),
             py::arg("hidden_size"),
             py::arg("group_tags")        = std::vector<std::string>{},
             py::arg("num_tokens_per_bs") = 0,
             py::arg("hc_mult")           = 1)
        .def("init_decode",
             &CudaGraphTestRunner::init_decode,
             py::arg("py_instance"),
             py::arg("hidden_size"),
             py::arg("max_seq_len"),
             py::arg("tokens_per_block"),
             py::arg("kernel_tokens_per_block"),
             py::arg("decode_capture_batch_sizes"),
             py::arg("group_tags")        = std::vector<std::string>{},
             py::arg("is_target_verify")  = false,
             py::arg("num_tokens_per_bs") = 1)
        .def("canRun", &CudaGraphTestRunner::canRun)
        .def("forward", &CudaGraphTestRunner::forward)
        .def("forward_with_numerical_status_snapshot",
             &CudaGraphTestRunner::forwardWithNumericalStatusSnapshot)
        .def("getCurrentRealGraphSize", &CudaGraphTestRunner::getCurrentRealGraphSize);
}
