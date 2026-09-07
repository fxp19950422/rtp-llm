#include "rtp_llm/cpp/testing/RejectionSamplingOpTest.hpp"

class CudaRejectionSamplingOpTest: public RejectionSamplingOpTest {};

TEST_F(CudaRejectionSamplingOpTest, referenceCases) {
    runReferenceCases();
}

TEST_F(CudaRejectionSamplingOpTest, zeroAndOneSpeculativeTokenCases) {
    runZeroAndOneSpeculativeTokenCases();
}

TEST_F(CudaRejectionSamplingOpTest, rejectsInvalidTensorMetadata) {
    runRejectsInvalidTensorMetadata();
}

TEST_F(CudaRejectionSamplingOpTest, greedyPointMassMatchesDenseAtEveryAcceptanceLength) {
    // Reuse the existing optional-draft-probability interface. No new sampler:
    // both launches must match an independent token/acceptance oracle.
    for (int steps : {1, 3}) {
        for (int vocab : {5, 129280}) {
            const int batch = steps + 1;
            std::vector<int32_t> draft(batch * steps, 1);
            std::vector<int32_t> target(batch * (steps + 1) * 2, -99);
            std::vector<int32_t> expected(batch * (steps + 1), -1);
            std::vector<int32_t> accepted(batch);
            for (int row = 0; row < batch; ++row) {
                accepted[row] = row + 1;
                for (int col = 0; col <= steps; ++col) {
                    const int token = col < row ? 1 : 2;
                    target[(row * (steps + 1) + col) * 2 + 1] = token;
                    if (col <= row) {
                        expected[row * (steps + 1) + col] = token;
                    }
                }
            }
            auto ids = cudaIntTensor({batch, steps}, draft);
            auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
            auto dense = torch::zeros({batch, steps, vocab}, opts);
            dense.scatter_(-1, ids.to(torch::kLong).unsqueeze(-1), 1.0);
            RejectionSamplingParams params{
                dense, ids, torch::full({batch, steps + 1}, 0.5f, opts),
                torch::full({batch, steps + 1, vocab}, 1.0f / vocab, opts),
                cudaIntTensor({batch * (steps + 1), 2}, target),
                cudaIntTensor({batch, steps + 1}, std::vector<int32_t>(batch * (steps + 1), -7)),
                cudaIntTensor({batch}, std::vector<int32_t>(batch, 0)),
                cudaBoolTensor(std::vector<bool>(batch, false)),
            };
            for (bool point_mass : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "steps=" << steps << " vocab=" << vocab
                                                 << " point_mass=" << point_mass);
                params.draft_probs_point_mass = point_mass;
                params.draft_probs_d = point_mass ? torch::Tensor() : dense;
                params.output_token_ids_d.fill_(-7);
                params.output_accepted_token_num_d.zero_();
                rejectionSampling(params);
                ASSERT_TRUE(torch::equal(params.output_token_ids_d, cudaIntTensor({batch, steps + 1}, expected)));
                ASSERT_TRUE(torch::equal(params.output_accepted_token_num_d, cudaIntTensor({batch}, accepted)));
            }
        }
    }
}
