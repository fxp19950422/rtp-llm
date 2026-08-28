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

TEST(CudaGraphMetadataUtilsTest, ParsesMtpCudaGraphDiagnosticModeFailClosed) {
    EXPECT_EQ(parseMtpCudaGraphDiagnosticMode(nullptr), MtpCudaGraphDiagnosticMode::OFF);
    EXPECT_EQ(parseMtpCudaGraphDiagnosticMode(""), MtpCudaGraphDiagnosticMode::OFF);
    EXPECT_EQ(parseMtpCudaGraphDiagnosticMode("0"), MtpCudaGraphDiagnosticMode::OFF);
    EXPECT_EQ(parseMtpCudaGraphDiagnosticMode("log"), MtpCudaGraphDiagnosticMode::LOG);
    EXPECT_EQ(parseMtpCudaGraphDiagnosticMode("exact"), MtpCudaGraphDiagnosticMode::EXACT_TARGET_VERIFY);
    EXPECT_EQ(parseMtpCudaGraphDiagnosticMode("1"), MtpCudaGraphDiagnosticMode::INVALID);
    EXPECT_EQ(parseMtpCudaGraphDiagnosticMode("EXACT"), MtpCudaGraphDiagnosticMode::INVALID);
}

TEST(CudaGraphMetadataUtilsTest, ExactDiagnosticRejectsOnlyRoundedTargetVerify) {
    EXPECT_FALSE(shouldRejectRoundedMtpTargetVerify(MtpCudaGraphDiagnosticMode::OFF, true, 2, 4));
    EXPECT_FALSE(shouldRejectRoundedMtpTargetVerify(MtpCudaGraphDiagnosticMode::LOG, true, 2, 4));
    EXPECT_FALSE(shouldRejectRoundedMtpTargetVerify(MtpCudaGraphDiagnosticMode::INVALID, true, 2, 4));
    EXPECT_FALSE(shouldRejectRoundedMtpTargetVerify(MtpCudaGraphDiagnosticMode::EXACT_TARGET_VERIFY, false, 2, 4));
    EXPECT_FALSE(shouldRejectRoundedMtpTargetVerify(MtpCudaGraphDiagnosticMode::EXACT_TARGET_VERIFY, true, 4, 4));
    EXPECT_TRUE(shouldRejectRoundedMtpTargetVerify(MtpCudaGraphDiagnosticMode::EXACT_TARGET_VERIFY, true, 2, 4));
}

}  // namespace
}  // namespace rtp_llm
