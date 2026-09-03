#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_device_shims.h"

namespace rtp_llm {
void CudaGraphRunner::replayDecode(int bs) {
    replayGraph(bs);
}

std::vector<int> CudaGraphRunner::getDecodeBatchSizesToCapture() {
    // If decode_capture_batch_sizes_ is provided from Python, use it directly
    if (!decode_capture_batch_sizes_.empty()) {
        RTP_LLM_LOG_INFO("Using decode capture batch sizes from Python: %zu sizes", decode_capture_batch_sizes_.size());
        // Sort in ascending order (from small to large)
        std::sort(decode_capture_batch_sizes_.begin(), decode_capture_batch_sizes_.end());
        auto result = decode_capture_batch_sizes_;
        // Draft-loop graphs are N-1x larger than single-step graphs.
        // Constrain to exactly 1 batch size to stay within VRAM budget.
        // Two graphs caused OOM on 144 GiB GPUs (~96 MiB shortfall);
        // keeping only the largest (most common online) saves ~400-600 MiB.
        if (is_draft_loop_ && result.size() > 1) {
            RTP_LLM_LOG_INFO("[draft-loop] trimming capture batch sizes from %zu to 1", result.size());
            result.erase(result.begin(), result.end() - 1);
        }
        return result;
    }

    // Otherwise, use default logic
    std::vector<int> capture_bs;
    int              max_generate_batch_size = max_bs_;
    RTP_LLM_LOG_INFO("max_generate_batch_size for cuda graph: %d", max_generate_batch_size);

    if (is_draft_loop_) {
        // Draft-loop mode: capture exactly 1 batch size (the max) to limit
        // memory.  Each graph stores N-1 steps of kernels, so a single
        // graph already consumes ~600+ MiB.  Two graphs caused OOM.
        capture_bs.push_back(max_generate_batch_size);
        return capture_bs;
    }

    // Keep the latency-sensitive small decode batches exact. In particular,
    // mapping live B2/B4 to a B8 graph expands persistent attention metadata
    // and has caused stale FP8 padding rows to affect MTP target verification.
    for (int i : {1, 2, 4, 8, 16, 24, 32}) {
        if (i <= max_generate_batch_size) {
            capture_bs.push_back(i);
        }
    }
    // Add range from 48 to max_generate_batch_size, stepping by 16
    for (int i = 48; i <= max_generate_batch_size; i += 16) {
        capture_bs.push_back(i);
    }
    if (capture_bs[capture_bs.size() - 1] != max_generate_batch_size) {
        capture_bs.push_back(max_generate_batch_size);
    }
    return capture_bs;
}

void CudaGraphRunner::captureDecodeOneBatchSize(int bs) {
    captureOneGraphInstance(bs, "batch size");
}

void CudaGraphRunner::captureDecode() {
    RTP_LLM_LOG_INFO("Capture Decode Start");
    // Pre-initialize all graph instances with keep_graph based on debug mode
    for (int bs : capture_range_) {
        graph_instances_.try_emplace(bs, enable_cuda_graph_debug_mode_);
    }
    int capture_range_size = capture_range_.size();
    for (int i = capture_range_size - 1; i >= 0; i--) {
        int           bs = capture_range_[i];
        PyModelInputs inputs;
        // Prepare common inputs using shared function
        prepareCaptureInputs(inputs, bs, bs * num_tokens_per_bs_);

        // calculate context_total_kv_length
        int max_input_len  = inputs.attention_inputs.input_lengths.max().item<int>();
        int max_prefix_len = 0;
        if (inputs.attention_inputs.prefix_lengths.defined() && inputs.attention_inputs.prefix_lengths.numel() > 0) {
            max_prefix_len = inputs.attention_inputs.prefix_lengths.max().item<int>();
        }
        inputs.attention_inputs.context_total_kv_length = bs * (max_input_len + max_prefix_len);
        // capture-specific metadata above was written after prepareCaptureInputs synchronized the tag map.
        refreshTaggedAttentionInputs(inputs);

        graph_instances_[bs].mem_hold_ = createCaptureMemoryHold(inputs, bs * num_tokens_per_bs_);
        graph_instances_[bs].mem_hold_.attn_pyobj_ =
            py_attn_pyobj_method_(graph_instances_[bs].mem_hold_.py_model_inputs_, true);
        captureDecodeOneBatchSize(bs);
        cuda_graph::finish_capture_session();
        replayAndSyncCheck(bs, "batch size");
        RTP_LLM_LOG_INFO("capture success for batch size: %d", bs);
    }
    RTP_LLM_LOG_INFO("Capture Decode End");
}
}  // namespace rtp_llm
