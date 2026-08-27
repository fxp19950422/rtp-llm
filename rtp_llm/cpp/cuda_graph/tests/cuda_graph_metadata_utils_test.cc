#include "rtp_llm/cpp/cuda_graph/cuda_graph_metadata_utils.h"

#include <cstdint>
#include <limits>

#include <gtest/gtest.h>

namespace rtp_llm {
namespace {

TEST(CudaGraphMetadataUtilsTest, SumsInputAndPrefixForActiveBatchOnly) {
    const int32_t input_lengths[]  = {4, 7, 1000};
    const int32_t prefix_lengths[] = {2, 3, 2000};

    EXPECT_EQ(sumActiveKvLengths(input_lengths, prefix_lengths, 2), 16);
}

TEST(CudaGraphMetadataUtilsTest, SupportsEmptyBatchAndInt64Accumulation) {
    const int32_t input_lengths[]  = {std::numeric_limits<int32_t>::max()};
    const int32_t prefix_lengths[] = {std::numeric_limits<int32_t>::max()};

    EXPECT_EQ(sumActiveKvLengths(input_lengths, prefix_lengths, 0), 0);
    EXPECT_EQ(sumActiveKvLengths(input_lengths, prefix_lengths, 1),
              2LL * std::numeric_limits<int32_t>::max());
}

}  // namespace
}  // namespace rtp_llm
