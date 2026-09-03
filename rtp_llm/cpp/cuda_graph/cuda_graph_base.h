#pragma once
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtp_llm {

using namespace torch_ext;

enum class DecodeGraphRole : uint8_t {
    PREFILL,
    DECODE,
    MTP_DRAFT,
    TARGET_VERIFY,
};

enum class DecodeGraphFallbackReason : uint8_t {
    NONE,
    GRAPH_DISABLED,
    CAPTURE_EMPTY,
    BATCH_EXCEEDS,
    CAN_REPLAY_REJECTED,
};

// Host-only result of the most recent canRun decision. Consumers record it
// only at the final forward branch; prepare/update probes intentionally do not.
struct DecodeGraphDecision {
    DecodeGraphRole           role{DecodeGraphRole::DECODE};
    DecodeGraphFallbackReason fallback_reason{DecodeGraphFallbackReason::CAN_REPLAY_REJECTED};
    bool                      is_decode_graph{true};
    bool                      replay{false};
    int                       actual_batch{0};
    int                       graph_key{0};
    int                       padding_rows{0};
};

// Per-invocation CUDA graph state owned by the caller. Keeping the decision
// here prevents asynchronous prepare/update probes from overwriting another
// forward call's final replay/fallback facts.
struct CudaGraphState {
    int                 current_batch_size{1};
    int                 current_seq_len{1};
    int                 current_real_graph_bs{1};       // for decode
    int                 current_real_graph_seq_len{1};  // for prefill
    int                 seq_len_sum{0};
    DecodeGraphDecision decode_graph_decision;
};

struct GraphParams {
    bool             enable_cuda_graph            = false;
    bool             enable_cuda_graph_debug_mode = false;
    bool             is_prefill_cuda_graph_mode   = false;
    bool             is_target_verify             = false;
    DecodeGraphRole  decode_graph_role             = DecodeGraphRole::DECODE;
    int              max_seq_len                  = 0;
    int              tokens_per_block             = 0;  // physical kv block size
    int              kernel_tokens_per_block      = 0;  // must be explicitly configured
    int              num_tokens_per_bs      = 1;  // Number of tokens per batch (1 for decode, max_seq_len for prefill)
    int              sp_steps               = 0;
    // When true, this graph runner captures the entire N-1 step MTP draft
    // loop as a single CUDA graph (forward_draft_loop on the Python side).
    // The Python model's forward_draft_loop method is called once with
    // draft_loop_steps = sp_steps (== propose_step_ - 1) and produces all
    // draft tokens + hidden states in one kernel stream.
    bool             is_draft_loop           = false;
    size_t           max_context_batch_size = 128;
    std::size_t      hidden_size            = 0;
    c10::ScalarType  model_data_type        = c10::ScalarType::Float;
    std::vector<int> prefill_capture_seq_lens;
    std::vector<int> decode_capture_batch_sizes;
    int64_t          hc_mult = 1;
    // Golden cache-group identity for CUDA graph capture/replay. A one-group
    // topology keeps the direct AttentionInputs fast path; multiple groups
    // require an exact tag -> AttentionInputs mapping at replay time.
    std::vector<std::string> kv_cache_group_tags;
    // Per-token position-id factor for combo_position_ids capture buffer.
    // 0 = model does not use combo_position_ids (no buffer allocated, capture skips it).
    // >0 = factor (e.g. Mrope = rope_config.index_factor). Sourced from
    //     description_.attention_conf.rope_config in the model wrapper, not Python reflection.
    int position_id_len_factor = 0;
    // Width of one input_hiddens row. This is deliberately independent from
    // the model output hidden_size because auxiliary feature rows may be wider.
    std::size_t input_hidden_size = 0;
    NumericalStatusScope numerical_status_scope = NumericalStatusScope::NONE;
};

class GraphBase {
public:
    GraphBase(py::object py_instance): py_instance_(std::move(py_instance)) {}
    virtual ~GraphBase() {}
    virtual void           initCapture()                                                = 0;
    virtual PyModelOutputs forward(const PyModelInputs& inputs, CudaGraphState& state)  = 0;
    virtual void           setPositionEncoding(torch::Tensor position_encoding)         = 0;
    virtual void           setTokenTypeEmbedding(torch::Tensor token_type_embedding)    = 0;
    virtual void           setInputEmbeddingScalar(float input_embedding_scalar)        = 0;
    virtual bool           canRun(const PyModelInputs& inputs, CudaGraphState& state)   = 0;
    virtual void           prepareAttentionInputs(const PyModelInputs& inputs,
                                                  CudaGraphState&      state,
                                                  bool                 skip_forward_event_sync = false) = 0;

    // Refresh only captured kv_cache_kernel_block_id state and FlashInfer plan
    // buffers after page-table changes. Other captured fields stay untouched.
    virtual void updateKVCacheKernelBlockId(const PyModelInputs& inputs, CudaGraphState& state) {}

    py::object py_instance_;
};
}  // namespace rtp_llm
