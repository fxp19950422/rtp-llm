#pragma once

#include <cstdint>

namespace rtp_llm {

inline int64_t sumActiveKvLengths(const int32_t* input_lengths, const int32_t* prefix_lengths, int batch_size) {
    int64_t total = 0;
    for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
        total += static_cast<int64_t>(input_lengths[batch_idx]) + static_cast<int64_t>(prefix_lengths[batch_idx]);
    }
    return total;
}

}  // namespace rtp_llm
