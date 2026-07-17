#include "rtp_llm/cpp/models/context_parallel/ContextParallelProcessorBase.h"
#include "rtp_llm/models_py/bindings/core/ExecOps.h"
#include "rtp_llm/models_py/bindings/core/OpData.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"

namespace rtp_llm {

namespace detail {

torch::Tensor selectContextParallelRows(const torch::Tensor& global_rows,
                                        size_t               num_decode_streams,
                                        const torch::Tensor& original_input_lengths,
                                        const torch::Tensor& prefill_chunk_lengths,
                                        const torch::Tensor& prefill_shuffle_indices) {
    RTP_LLM_CHECK(global_rows.defined());
    RTP_LLM_CHECK(global_rows.dim() >= 1);
    RTP_LLM_CHECK(original_input_lengths.device().is_cpu());
    RTP_LLM_CHECK(prefill_chunk_lengths.device().is_cpu());
    RTP_LLM_CHECK(prefill_shuffle_indices.device().is_cpu());

    const size_t num_model_streams = original_input_lengths.numel();
    RTP_LLM_CHECK(num_model_streams >= num_decode_streams);
    const size_t num_prefill_streams = num_model_streams - num_decode_streams;
    RTP_LLM_CHECK(prefill_chunk_lengths.numel() == num_prefill_streams);

    const int64_t local_rows = static_cast<int64_t>(num_decode_streams) + prefill_chunk_lengths.sum().item<int64_t>();
    RTP_LLM_CHECK(prefill_shuffle_indices.numel() == local_rows - static_cast<int64_t>(num_decode_streams));
    auto indices = torch::empty({local_rows}, torch::TensorOptions(torch::kInt64).pinned_memory(true));
    auto valid   = torch::ones({local_rows}, torch::TensorOptions(torch::kBool).pinned_memory(true));
    auto idx_ptr = indices.data_ptr<int64_t>();
    auto val_ptr = valid.data_ptr<bool>();
    for (size_t row = 0; row < num_decode_streams; ++row) {
        idx_ptr[row] = static_cast<int64_t>(row);
    }

    const int32_t* original_lengths = original_input_lengths.data_ptr<int32_t>();
    const int32_t* chunk_lengths    = prefill_chunk_lengths.data_ptr<int32_t>();
    const int32_t* shuffle          = prefill_shuffle_indices.data_ptr<int32_t>();
    int64_t        global_offset    = static_cast<int64_t>(num_decode_streams);
    int64_t        local_offset     = static_cast<int64_t>(num_decode_streams);
    int64_t        shuffle_offset   = 0;
    for (size_t stream = 0; stream < num_prefill_streams; ++stream) {
        const int32_t input_length = original_lengths[num_decode_streams + stream];
        const int32_t chunk_length = chunk_lengths[stream];
        RTP_LLM_CHECK(input_length > 0);
        for (int32_t row = 0; row < chunk_length; ++row) {
            const int32_t source        = shuffle[shuffle_offset + row];
            const bool    is_valid      = source >= 0 && source < input_length;
            idx_ptr[local_offset + row] = global_offset + (is_valid ? source : 0);
            val_ptr[local_offset + row] = is_valid;
        }
        global_offset += input_length;
        local_offset += chunk_length;
        shuffle_offset += chunk_length;
    }
    RTP_LLM_CHECK(global_offset == global_rows.size(0));

    auto selected     = global_rows.index_select(0, indices.to(global_rows.device()));
    auto valid_device = valid.to(global_rows.device());
    for (int64_t dim = 1; dim < global_rows.dim(); ++dim) {
        valid_device = valid_device.unsqueeze(-1);
    }
    return torch::where(valid_device, selected, torch::zeros_like(selected));
}

torch::Tensor selectContextParallelHiddenStates(const torch::Tensor& global_hidden,
                                                size_t               num_decode_streams,
                                                const torch::Tensor& original_input_lengths,
                                                const torch::Tensor& prefill_chunk_lengths,
                                                const torch::Tensor& prefill_shuffle_indices) {
    RTP_LLM_CHECK(global_hidden.dim() == 2);
    return selectContextParallelRows(
        global_hidden, num_decode_streams, original_input_lengths, prefill_chunk_lengths, prefill_shuffle_indices);
}

}  // namespace detail

void IContextParallelProcessor::handleInputs(GptModelInputs&                     model_input,
                                             torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
#else
    int prefill_cp_rank = parallelism_config_.tp_rank;
    int prefill_cp_size = parallelism_config_.tp_size;
    int cp_align_size   = prefill_cp_size * 2;

    static const auto pinned_i32 = torch::TensorOptions(torch::kInt32).pinned_memory(true);

    auto& total_input_tokens       = model_input.combo_tokens;
    auto& input_lengths            = model_input.input_lengths;
    auto& sequence_lengths         = model_input.sequence_lengths;
    auto  input_lengths_cpu_tensor = input_lengths.clone().pin_memory();

    size_t num_decode_stream  = sequence_lengths.size(0);
    size_t num_prefill_stream = input_lengths.size(0) - num_decode_stream;

    auto prefill_cp_padding_lengths = torch::empty({(int64_t)num_prefill_stream}, pinned_i32);
    auto prefill_cp_chunk_lengths   = torch::empty({(int64_t)num_prefill_stream}, pinned_i32);
    int* padding_lengths            = prefill_cp_padding_lengths.data_ptr<int>();
    int* chunk_lengths              = prefill_cp_chunk_lengths.data_ptr<int>();

    size_t prefill_cp_split_tokens_size = 0;
    for (size_t p = 0; p < num_prefill_stream; ++p) {
        int num_prefill_token = input_lengths.data_ptr<int32_t>()[num_decode_stream + p];

        int padded_seq_len = ((num_prefill_token + cp_align_size - 1) / cp_align_size) * cp_align_size;
        int padding_size   = padded_seq_len - num_prefill_token;
        int chunk_size     = padded_seq_len / prefill_cp_size;

        prefill_cp_split_tokens_size += chunk_size;
        padding_lengths[p] = padding_size;
        chunk_lengths[p]   = chunk_size;
    }

    const bool input_tokens_on_device = total_input_tokens.device().is_cuda();
    // MTP replaces combo_tokens with sampled CUDA tokens after each draft step.
    // Build host shuffle metadata for target verify, then apply it on device.
    torch::Tensor cp_split_input_tokens;
    if (!input_tokens_on_device) {
        cp_split_input_tokens = torch::empty({(int64_t)(num_decode_stream + prefill_cp_split_tokens_size)}, pinned_i32);
    }
    auto prefill_shuffle_indices = torch::empty({(int64_t)prefill_cp_split_tokens_size}, pinned_i32);

    int* input_token_ptr             = input_tokens_on_device ? nullptr : cp_split_input_tokens.data_ptr<int>();
    int* input_length_ptr            = input_lengths.data_ptr<int32_t>();
    int* prefill_shuffle_indices_ptr = prefill_shuffle_indices.data_ptr<int>();

    int input_token_idx       = 0;
    int total_input_token_idx = 0;

    if (num_decode_stream > 0) {
        if (!input_tokens_on_device) {
            std::memcpy(input_token_ptr,
                        total_input_tokens.data_ptr<int32_t>() + total_input_token_idx,
                        num_decode_stream * sizeof(int));
        }
        input_token_idx += num_decode_stream;
        total_input_token_idx += num_decode_stream;
    }

    for (size_t p = 0; p < num_prefill_stream; ++p) {
        int input_chunk_length   = prefill_cp_chunk_lengths.data_ptr<int>()[p];
        int input_padding_length = prefill_cp_padding_lengths.data_ptr<int>()[p];
        int input_length         = input_lengths.data_ptr<int32_t>()[num_decode_stream + p];

        std::vector<int> total_input_token_vec;
        if (input_tokens_on_device) {
            total_input_token_vec.resize(input_length);
        } else {
            int* src_tokens = total_input_tokens.data_ptr<int32_t>() + total_input_token_idx;
            total_input_token_vec.assign(src_tokens, src_tokens + input_length);
        }
        std::vector<int> chunk_input_token(input_chunk_length, 0);
        std::vector<int> shuffle_index(input_chunk_length, -1);

        bool success = plan(total_input_token_vec,
                            chunk_input_token,
                            shuffle_index,
                            prefill_cp_rank,
                            prefill_cp_size,
                            input_chunk_length,
                            input_padding_length);
        RTP_LLM_CHECK_WITH_INFO(success, "Context parallel planning failed for prefill stream %zu", p);

        if (!input_tokens_on_device) {
            std::memcpy(input_token_ptr + input_token_idx, chunk_input_token.data(), input_chunk_length * sizeof(int));
        }
        std::memcpy(prefill_shuffle_indices_ptr + input_token_idx - num_decode_stream,
                    shuffle_index.data(),
                    input_chunk_length * sizeof(int));
        input_token_idx += input_chunk_length;
        total_input_token_idx += input_length;
        input_length_ptr[num_decode_stream + p] = input_chunk_length;
    }

    if (input_tokens_on_device) {
        cp_split_input_tokens = num_prefill_stream == 0 ? total_input_tokens :
                                                          detail::selectContextParallelRows(total_input_tokens,
                                                                                            num_decode_stream,
                                                                                            input_lengths_cpu_tensor,
                                                                                            prefill_cp_chunk_lengths,
                                                                                            prefill_shuffle_indices);
    }

    if (model_input.last_hidden_states.defined()) {
        model_input.last_hidden_states = detail::selectContextParallelHiddenStates(model_input.last_hidden_states,
                                                                                   num_decode_stream,
                                                                                   input_lengths_cpu_tensor,
                                                                                   prefill_cp_chunk_lengths,
                                                                                   prefill_shuffle_indices);
    }
    model_input.combo_tokens = cp_split_input_tokens;
    auto cp_padding_lengths  = prefill_cp_padding_lengths;
    auto cp_chunk_lengths    = prefill_cp_chunk_lengths;
    auto shuffle_indices     = prefill_shuffle_indices;

    auto qkv_restore_indice = generateQKVRestoreIndices(cp_chunk_lengths, prefill_cp_size);
    auto qkv_padding_mask   = generateQKVPaddingMask(cp_chunk_lengths, cp_padding_lengths, prefill_cp_size);

    cp_params.prefill_cp_padding_lengths       = cp_padding_lengths.cuda();
    cp_params.prefill_cp_chunk_lengths         = cp_chunk_lengths.cuda();
    cp_params.prefill_shuffle_indices          = shuffle_indices.cuda();
    cp_params.prefill_qkv_restore_indice       = qkv_restore_indice.cuda();
    cp_params.prefill_qkv_padding_mask         = qkv_padding_mask.cuda();
    cp_params.prefill_actual_input_lengths_cpu = input_lengths_cpu_tensor;
#endif
}

size_t IContextParallelProcessor::handleOutputs(torch::Tensor&                            hidden_states,
                                                const GptModelInputs&                     inputs,
                                                const torch_ext::PyContextParallelParams& cp_params) {
#if !USING_CUDA
    RTP_LLM_FAIL("Context parallel not supported on ROCm");
    return 0;
#else
    int prefill_cp_size = parallelism_config_.tp_size;

    auto all_hidden_t =
        torch::empty({hidden_states.size(0) * prefill_cp_size, hidden_states.size(1)}, hidden_states.options());
    execAllGather({{all_hidden_t}, ParallelMode::TP, {hidden_states}, false});

    int64_t num_valid_tokens = all_hidden_t.size(0);
    hidden_states            = all_hidden_t;
    return num_valid_tokens;
#endif
}

}  // namespace rtp_llm
