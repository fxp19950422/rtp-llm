#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <tuple>

namespace py = pybind11;
using namespace rtp_llm;

namespace unittest {

// Test-only wrapper class to expose protected methods for unit testing
class ZigZagProcessorTestWrapper: public ZigZagProcessor {
public:
    ZigZagProcessorTestWrapper(): ZigZagProcessor(ParallelismConfig{}) {}
    explicit ZigZagProcessorTestWrapper(const ParallelismConfig& config): ZigZagProcessor(config) {}
    using ZigZagProcessor::plan;
    using ZigZagProcessor::generateQKVRestoreIndices;
    using ZigZagProcessor::generateQKVPaddingMask;
};

// Wrapper for ZigZagProcessor::plan that returns a tuple
std::tuple<bool, std::vector<int>, std::vector<int>>
zigzagProcessorPlanWrapper(const std::vector<int>& total_input_tokens,
                           std::vector<int>        input_tokens,
                           std::vector<int>        shuffle_indices,
                           int                     cp_rank,
                           int                     cp_size,
                           int                     cp_chunk_size,
                           int                     cp_padding_size) {
    input_tokens.resize(cp_chunk_size);
    shuffle_indices.resize(cp_chunk_size);

    ZigZagProcessorTestWrapper processor;
    bool                       result = processor.plan(
        total_input_tokens, input_tokens, shuffle_indices, cp_rank, cp_size, cp_chunk_size, cp_padding_size);

    return std::make_tuple(result, input_tokens, shuffle_indices);
}

// Wrapper for ZigZagProcessor::generateQKVRestoreIndices
torch::Tensor zigzagGenerateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) {
    ZigZagProcessorTestWrapper processor;
    return processor.generateQKVRestoreIndices(prefill_cp_chunk_lengths, cp_size);
}

// Wrapper for ZigZagProcessor::generateQKVPaddingMask
torch::Tensor zigzagGenerateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                           const torch::Tensor& prefill_cp_padding_lengths,
                                           int                  cp_size) {
    ZigZagProcessorTestWrapper processor;
    return processor.generateQKVPaddingMask(prefill_cp_chunk_lengths, prefill_cp_padding_lengths, cp_size);
}

torch::Tensor restoreContextParallelOutputs(const torch::Tensor& decode_hidden,
                                            const torch::Tensor& gathered_prefill_hidden,
                                            const torch::Tensor& restore_indices,
                                            const torch::Tensor& padding_mask) {
    return rtp_llm::detail::restoreContextParallelOutputs(
        decode_hidden, gathered_prefill_hidden, restore_indices, padding_mask);
}

torch::Tensor selectContextParallelHiddenStates(const torch::Tensor& global_hidden,
                                                size_t               num_decode_streams,
                                                const torch::Tensor& original_input_lengths,
                                                const torch::Tensor& prefill_chunk_lengths,
                                                const torch::Tensor& prefill_shuffle_indices) {
    return rtp_llm::detail::selectContextParallelHiddenStates(
        global_hidden, num_decode_streams, original_input_lengths, prefill_chunk_lengths, prefill_shuffle_indices);
}

std::tuple<torch::Tensor, torch::Tensor> handleContextParallelDecodeInputs(const torch::Tensor& combo_tokens,
                                                                           const torch::Tensor& hidden_states) {
    ParallelismConfig config;
    config.tp_size = 16;
    config.tp_rank = 0;
    ZigZagProcessorTestWrapper processor(config);

    GptModelInputs inputs;
    inputs.combo_tokens  = combo_tokens;
    inputs.input_lengths = torch::ones({combo_tokens.size(0)}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    inputs.sequence_lengths =
        torch::ones({combo_tokens.size(0)}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    inputs.last_hidden_states = hidden_states;
    torch_ext::PyContextParallelParams cp_params;
    processor.handleInputs(inputs, cp_params);
    return std::make_tuple(inputs.combo_tokens, inputs.last_hidden_states);
}

torch::Tensor handleContextParallelTargetVerifyInputs(const torch::Tensor& combo_tokens, int cp_rank) {
    ParallelismConfig config;
    config.tp_size = 16;
    config.tp_rank = cp_rank;
    ZigZagProcessorTestWrapper processor(config);

    GptModelInputs inputs;
    inputs.combo_tokens  = combo_tokens;
    inputs.input_lengths = torch::empty({1}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    inputs.input_lengths.data_ptr<int32_t>()[0] = combo_tokens.size(0);
    inputs.sequence_lengths = torch::empty({0}, torch::TensorOptions(torch::kInt32).pinned_memory(true));
    torch_ext::PyContextParallelParams cp_params;
    processor.handleInputs(inputs, cp_params);
    return inputs.combo_tokens;
}

PYBIND11_MODULE(libth_context_parallel_py_wrapper_test, m) {
    m.def("context_parallel_load_balance_split",
          &zigzagProcessorPlanWrapper,
          py::arg("total_input_tokens"),
          py::arg("input_tokens"),
          py::arg("shuffle_indices"),
          py::arg("cp_rank"),
          py::arg("cp_size"),
          py::arg("cp_chunk_size"),
          py::arg("cp_padding_size"),
          "Distribute input tokens across context parallel ranks with load balancing (legacy wrapper)");

    m.def("generate_qkv_restore_indices",
          &zigzagGenerateQKVRestoreIndices,
          py::arg("prefill_cp_chunk_lengths"),
          py::arg("cp_size"),
          "Generate indices to restore original token order after parallel processing (legacy wrapper)");

    m.def("generate_qkv_padding_mask",
          &zigzagGenerateQKVPaddingMask,
          py::arg("prefill_cp_chunk_lengths"),
          py::arg("prefill_cp_padding_lengths"),
          py::arg("cp_size"),
          "Generate padding mask for QKV tensors in context parallel scenarios (legacy wrapper)");

    m.def("restore_context_parallel_outputs",
          &restoreContextParallelOutputs,
          py::arg("decode_hidden"),
          py::arg("gathered_prefill_hidden"),
          py::arg("restore_indices"),
          py::arg("padding_mask"),
          "Preserve decode rows and restore valid gathered prefill rows");

    m.def("select_context_parallel_hidden_states",
          &selectContextParallelHiddenStates,
          py::arg("global_hidden"),
          py::arg("num_decode_streams"),
          py::arg("original_input_lengths"),
          py::arg("prefill_chunk_lengths"),
          py::arg("prefill_shuffle_indices"),
          "Apply the rank-local token mapping to MTP hidden states");

    m.def("handle_context_parallel_decode_inputs",
          &handleContextParallelDecodeInputs,
          py::arg("combo_tokens"),
          py::arg("hidden_states"),
          "Preserve device-resident tokens across decode-only MTP draft steps");

    m.def("handle_context_parallel_target_verify_inputs",
          &handleContextParallelTargetVerifyInputs,
          py::arg("combo_tokens"),
          py::arg("cp_rank"),
          "Shuffle and pad device-resident MTP target-verify tokens");
}

}  // namespace unittest
