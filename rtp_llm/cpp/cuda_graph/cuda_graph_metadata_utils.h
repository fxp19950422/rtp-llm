#pragma once

#include <cstdint>
#include <string_view>

namespace rtp_llm {

enum class MtpCudaGraphDiagnosticMode {
    OFF,
    LOG,
    EXACT_TARGET_VERIFY,
    INVALID,
};

inline MtpCudaGraphDiagnosticMode parseMtpCudaGraphDiagnosticMode(const char* value) {
    if (value == nullptr || value[0] == '\0' || std::string_view(value) == "0") {
        return MtpCudaGraphDiagnosticMode::OFF;
    }
    if (std::string_view(value) == "log") {
        return MtpCudaGraphDiagnosticMode::LOG;
    }
    if (std::string_view(value) == "exact") {
        return MtpCudaGraphDiagnosticMode::EXACT_TARGET_VERIFY;
    }
    return MtpCudaGraphDiagnosticMode::INVALID;
}

inline bool shouldRejectRoundedMtpTargetVerify(MtpCudaGraphDiagnosticMode mode,
                                               bool                       is_target_verify,
                                               int                        real_batch_size,
                                               int                        graph_batch_size) {
    return mode == MtpCudaGraphDiagnosticMode::EXACT_TARGET_VERIFY && is_target_verify
           && real_batch_size != graph_batch_size;
}

inline int64_t sumActiveKvLengths(const int32_t* input_lengths, const int32_t* prefix_lengths, int batch_size) {
    int64_t total = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        total += static_cast<int64_t>(input_lengths[batch_idx]) + static_cast<int64_t>(prefix_lengths[batch_idx]);
    }
    return total;
}

}  // namespace rtp_llm
