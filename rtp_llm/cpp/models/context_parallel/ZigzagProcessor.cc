#include "rtp_llm/cpp/models/context_parallel/ZigzagProcessor.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <numeric>
#include <vector>

using namespace std;

namespace rtp_llm {

namespace detail {

torch::Tensor restoreContextParallelOutputs(const torch::Tensor& decode_hidden,
                                            const torch::Tensor& gathered_prefill_hidden,
                                            const torch::Tensor& restore_indices,
                                            const torch::Tensor& padding_mask) {
    if (gathered_prefill_hidden.size(0) == 0) {
        return decode_hidden;
    }

    auto valid_indices    = torch::nonzero(padding_mask).squeeze(-1);
    auto combined_indices = restore_indices.index_select(0, valid_indices);
    auto restored_prefill = gathered_prefill_hidden.index_select(0, combined_indices);
    if (decode_hidden.size(0) == 0) {
        return restored_prefill;
    }
    return torch::cat({decode_hidden, restored_prefill}, 0);
}

}  // namespace detail

bool ZigZagProcessor::plan(const std::vector<int>& total_input_tokens,
                           std::vector<int>&       input_tokens,
                           std::vector<int>&       shuffle_indices,
                           int                     cp_rank,
                           int                     cp_size,
                           int                     cp_chunk_size,
                           int                     cp_padding_size) {
    const int input_token_size      = static_cast<int>(total_input_tokens.size());
    const int padded_seq_token_size = input_token_size + cp_padding_size;
    RTP_LLM_CHECK(cp_rank >= 0 && cp_rank < cp_size);

    const int pair_size = padded_seq_token_size / (cp_size * 2);

    // Even pair (from start): indices are [cp_rank * pair_size, ...)
    const int even_source = cp_rank * pair_size;
    // Odd pair (from end): indices are [padded_seq_token_size - pair_size * (cp_rank + 1), ...)
    const int odd_source = padded_seq_token_size - pair_size * (cp_rank + 1);

    // Fill shuffle_indices
    std::iota(shuffle_indices.begin(), shuffle_indices.begin() + pair_size, even_source);
    std::iota(shuffle_indices.begin() + pair_size, shuffle_indices.begin() + pair_size * 2, odd_source);

    // Even pair: source indices [even_source, even_source + pair_size)
    if (even_source < input_token_size) {
        const int copy_size = std::min(pair_size, input_token_size - even_source);
        std::memcpy(input_tokens.data(), total_input_tokens.data() + even_source, copy_size * sizeof(int));
    }

    // Odd pair: source indices [odd_source, odd_source + pair_size)
    if (odd_source < input_token_size) {
        const int copy_size = std::min(pair_size, input_token_size - odd_source);
        std::memcpy(input_tokens.data() + pair_size, total_input_tokens.data() + odd_source, copy_size * sizeof(int));
    }
    return true;
}

torch::Tensor ZigZagProcessor::generateQKVRestoreIndices(const torch::Tensor& prefill_cp_chunk_lengths, int cp_size) {
    int           num_prefill_streams = prefill_cp_chunk_lengths.size(0);
    int           total_token_size    = torch::sum(prefill_cp_chunk_lengths).item<int>();
    torch::Tensor qkv_restore_indices =
        torch::empty({cp_size, total_token_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU));

    int* qkv_data = qkv_restore_indices.data_ptr<int>();

    // Optimized: Directly compute indices without generating full shuffle_indices each time
    int chunk_offset = 0;
    int seq_offset   = 0;
    for (int stream = 0; stream < num_prefill_streams; stream++) {
        int chunk_length    = prefill_cp_chunk_lengths[stream].item<int>();
        int prefill_qkv_len = chunk_length * cp_size;
        int pair_size       = chunk_length / 2;  // prefill_qkv_len / (cp_size * 2)

        // For each cp_rank, directly compute its indices without full shuffle generation
        for (int cp_rank = 0; cp_rank < cp_size; cp_rank++) {
            int* dst = qkv_data + cp_rank * total_token_size + chunk_offset;

            // Even pair (from start): indices are [cp_rank * pair_size, ...)
            const int even_source = cp_rank * pair_size + seq_offset;
            std::iota(dst, dst + pair_size, even_source);

            // Odd pair (from end): indices are [prefill_qkv_len - pair_size * (cp_rank + 1), ...)
            const int odd_source = prefill_qkv_len - pair_size * (cp_rank + 1) + seq_offset;
            std::iota(dst + pair_size, dst + pair_size * 2, odd_source);
        }
        chunk_offset += chunk_length;
        seq_offset += prefill_qkv_len;
    }
    torch::Tensor sorted_indices = torch::empty(
        {cp_size * total_token_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    int* indices_data = sorted_indices.data_ptr<int>();

    for (int flat_idx = 0; flat_idx < cp_size * total_token_size; flat_idx++) {
        int value           = qkv_data[flat_idx];
        indices_data[value] = flat_idx;
    }
    return sorted_indices;
}

torch::Tensor ZigZagProcessor::generateQKVPaddingMask(const torch::Tensor& prefill_cp_chunk_lengths,
                                                      const torch::Tensor& prefill_cp_padding_lengths,
                                                      int                  cp_size) {
    int num_prefill_streams = prefill_cp_chunk_lengths.size(0);

    // Calculate padded sequence lengths: chunk_length * cp_size
    auto padded_seq_lengths = prefill_cp_chunk_lengths * cp_size;

    // Calculate total mask size
    int total_size = torch::sum(padded_seq_lengths).item<int>();

    // Optimized: Initialize with 1s (valid tokens) first, then overwrite padding with 0s
    // This is faster than separate fill operations for large sequences
    torch::Tensor padding_mask =
        torch::empty({total_size}, torch::TensorOptions(torch::kInt32).device(torch::kCPU).pinned_memory(true));
    int* mask_data = padding_mask.data_ptr<int>();

    // Only fill padding regions (typically smaller than valid regions)
    int offset = 0;
    for (int i = 0; i < num_prefill_streams; i++) {
        int padded_length = padded_seq_lengths[i].item<int>();
        int padding_count = prefill_cp_padding_lengths[i].item<int>();
        int valid_count   = padded_length - padding_count;

        std::fill_n(mask_data + offset, valid_count, 1);

        if (padding_count > 0) {
            int valid_count = padded_length - padding_count;
            // Only overwrite padding tokens to 0
            std::fill_n(mask_data + offset + valid_count, padding_count, 0);
        }
        offset += padded_length;
    }
    return padding_mask;
}

size_t ZigZagProcessor::handleOutputs(torch::Tensor&                            hidden_states,
                                      const GptModelInputs&                     inputs,
                                      const torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
    return 0;
#else
    const int64_t num_decode_streams = inputs.sequence_lengths.size(0);
    const int64_t num_model_streams  = inputs.input_lengths.size(0);
    RTP_LLM_CHECK_WITH_INFO(num_model_streams >= num_decode_streams,
                            "input stream count %ld is smaller than decode stream count %ld",
                            num_model_streams,
                            num_decode_streams);
    RTP_LLM_CHECK_WITH_INFO(hidden_states.size(0) >= num_decode_streams,
                            "hidden rows %ld is smaller than decode stream count %ld",
                            hidden_states.size(0),
                            num_decode_streams);

    // Decode tokens are replicated across the CP ranks and must remain one row
    // per request. Only the local prefill suffix is CP-sharded and needs an
    // all-gather followed by zig-zag restoration.
    auto decode_hidden = hidden_states.narrow(0, 0, num_decode_streams);
    if (num_model_streams == num_decode_streams) {
        hidden_states = decode_hidden;
        return hidden_states.size(0);
    }

    auto local_prefill_hidden = hidden_states.narrow(0, num_decode_streams, hidden_states.size(0) - num_decode_streams);
    const int prefill_cp_size = parallelism_config_.tp_size;
    auto      gathered_prefill_hidden =
        torch::empty({local_prefill_hidden.size(0) * prefill_cp_size, hidden_states.size(1)}, hidden_states.options());
    execAllGather({{gathered_prefill_hidden}, ParallelMode::TP, {local_prefill_hidden}, false});

    hidden_states = detail::restoreContextParallelOutputs(decode_hidden,
                                                          gathered_prefill_hidden,
                                                          cp_params.prefill_qkv_restore_indice,
                                                          cp_params.prefill_qkv_padding_mask);
    return hidden_states.size(0);
#endif
}

}  // namespace rtp_llm
