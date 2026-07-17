#include <gtest/gtest.h>

#include "rtp_llm/cpp/cuda_graph/cuda_graph_utils.h"
#include "rtp_llm/cpp/cuda_graph/cuda_graph_runner.h"

namespace rtp_llm {
namespace test {

TEST(CaptureMemoryHoldTest, PreservesCpKvLayout) {
    torch_ext::PyModelInputs inputs;
    auto&                    layout = inputs.attention_inputs.cp_kv_layout;
    layout.kind                     = CpKvLayoutKind::PAGE_INTERLEAVED;
    layout.cp_size                  = 16;
    layout.cp_rank                  = 7;
    layout.page_size                = 64;
    layout.interleave_size          = 64;
    layout.layout_version           = 1;
    layout.max_global_length        = 200128;

    CaptureMemoryHold hold(torch::Tensor(), inputs, false);
    const auto&       copied = hold.py_model_inputs_.attention_inputs.cp_kv_layout;

    EXPECT_EQ(copied.kind, CpKvLayoutKind::PAGE_INTERLEAVED);
    EXPECT_EQ(copied.cp_size, 16);
    EXPECT_EQ(copied.cp_rank, 7);
    EXPECT_EQ(copied.page_size, 64);
    EXPECT_EQ(copied.interleave_size, 64);
    EXPECT_EQ(copied.layout_version, 1);
    EXPECT_EQ(copied.max_global_length, 200128);
}

TEST(CaptureMemoryHoldTest, PreservesTargetVerifyContextParallelMetadata) {
    torch_ext::PyModelInputs           inputs;
    torch_ext::PyContextParallelParams cp_info;
    cp_info.prefill_actual_input_lengths_cpu      = torch::tensor({6, 6}, torch::kInt32);
    inputs.attention_inputs.context_parallel_info = cp_info;

    CaptureMemoryHold hold(torch::Tensor(), inputs, false);

    ASSERT_TRUE(hold.py_model_inputs_.attention_inputs.context_parallel_info.has_value());
    EXPECT_TRUE(
        torch::equal(hold.py_model_inputs_.attention_inputs.context_parallel_info->prefill_actual_input_lengths_cpu,
                     torch::tensor({6, 6}, torch::kInt32)));
}

TEST(CudaGraphTargetVerifyTest, UsesCpPlannerLocalWidth) {
    EXPECT_EQ(targetVerifyGraphTokensPerRequest(6, 8), 2);
    EXPECT_EQ(targetVerifyGraphTokensPerRequest(16, 8), 2);
    EXPECT_EQ(targetVerifyGraphTokensPerRequest(17, 8), 4);
    EXPECT_EQ(targetVerifyGraphTokensPerRequest(6, 0), 6);
}

TEST(CudaGraphPaddingTest, ClearsPaddedTargetVerifyMetadata) {
    torch_ext::PyAttentionInputs inputs;
    inputs.input_lengths                  = torch::tensor({6, 6, 99, 99}, torch::kInt32);
    inputs.input_lengths_device           = inputs.input_lengths.clone();
    inputs.prefix_lengths                 = torch::tensor({100, 200, 99, 99}, torch::kInt32);
    inputs.prefix_lengths_device          = inputs.prefix_lengths.clone();
    inputs.sequence_lengths               = torch::tensor({0, 0, 99, 99}, torch::kInt32);
    inputs.sequence_lengths_plus_1_device = torch::tensor({101, 201, 99, 99}, torch::kInt32);
    inputs.cu_seqlens                     = torch::tensor({0, 6, 12, 99, 99}, torch::kInt32);
    inputs.cu_seqlens_device              = inputs.cu_seqlens.clone();
    inputs.cu_kv_seqlens_device           = torch::tensor({0, 106, 312, 99, 99}, torch::kInt32);

    resetPaddedPrefillMetadata(inputs, 2, 4, 12, 312);

    EXPECT_TRUE(torch::equal(inputs.input_lengths, torch::tensor({6, 6, 0, 0}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(inputs.input_lengths_device, torch::tensor({6, 6, 0, 0}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(inputs.prefix_lengths, torch::tensor({100, 200, 0, 0}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(inputs.prefix_lengths_device, torch::tensor({100, 200, 0, 0}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(inputs.sequence_lengths, torch::tensor({0, 0, 0, 0}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(inputs.sequence_lengths_plus_1_device, torch::tensor({101, 201, 0, 0}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(inputs.cu_seqlens, torch::tensor({0, 6, 12, 12, 12}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(inputs.cu_seqlens_device, torch::tensor({0, 6, 12, 12, 12}, torch::kInt32)));
    EXPECT_TRUE(torch::equal(inputs.cu_kv_seqlens_device, torch::tensor({0, 106, 312, 312, 312}, torch::kInt32)));
}

}  // namespace test
}  // namespace rtp_llm
