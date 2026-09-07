#include "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.h"

#include <gtest/gtest.h>
#include <torch/all.h>
#include <torch/cuda.h>
#include <chrono>
#include <cstdlib>
#include <iostream>

namespace rtp_llm {
namespace speculative {
namespace {

TEST(FastTopKSamplerTest, TopKOneReturnsArgmaxIndex) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});
    auto            out    = sampler.forward(logits, 1);

    ASSERT_EQ(out.token_ids.dim(), 2);
    ASSERT_EQ(out.token_ids.size(0), 1);
    ASSERT_EQ(out.token_ids.size(1), 1);
    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 2);
}

TEST(FastTopKSamplerTest, TopKGreaterThanOneReturnsTopKIndices) {
    FastTopKSampler sampler;
    auto            logits = torch::tensor({{1.0f, 2.0f, 5.0f, 3.0f}});
    auto            out    = sampler.forward(logits, 2);

    ASSERT_EQ(out.token_ids.dim(), 2);
    ASSERT_EQ(out.token_ids.size(0), 1);
    ASSERT_EQ(out.token_ids.size(1), 2);
    // softmax preserves ordering: indices 2 (5.0) then 3 (3.0).
    EXPECT_EQ(out.token_ids[0][0].item<int64_t>(), 2);
    EXPECT_EQ(out.token_ids[0][1].item<int64_t>(), 3);
}

TEST(FastTopKSamplerTest, TokenOnlyPreservesSoftmaxMaxIncludingTies) {
    FastTopKSampler sampler;
    auto logits = torch::tensor({{1.0f, 1.0f, -1000.0f, 0.0f},
                                 {1.0f, 1.0000001192092896f, 0.0f, -1.0f},
                                 {-1000.0f, 1000.0f, 1000.0f, 0.0f},
                                 {0.0f, 0.0f, 0.0f, 0.0f}});
    for (const auto device : {torch::kCPU, torch::kCUDA}) {
        auto input = logits.to(device);
        auto dense = sampler.forward(input, 1);
        auto ids   = sampler.forwardTokenIds(input);
        auto expected = std::get<1>(torch::max(torch::softmax(input, -1), -1, true));
        EXPECT_TRUE(torch::equal(ids, expected));
        EXPECT_TRUE(torch::equal(ids, dense.token_ids));
        EXPECT_TRUE(torch::equal(dense.all_probs, torch::zeros_like(input).scatter_(-1, ids, 1.0)));
    }
}

TEST(FastTopKSamplerTest, TokenOnlyHostLoopBenchmark) {
    if (std::getenv("RTP_LLM_TEST_TOKEN_ONLY_BENCH") == nullptr) {
        GTEST_SKIP() << "Opt-in isolated-device microbenchmark, not an end-to-end performance gate";
    }
    FastTopKSampler sampler;
    constexpr int64_t vocab = 129280;
    constexpr int iterations = 30;
    for (int64_t batch : {1, 16, 80}) {
        SCOPED_TRACE(batch);
        auto logits = torch::arange(vocab, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA))
                          .div(vocab).repeat({batch, 1});
        auto previous = sampler.forward(logits, 1);
        auto carried_probs = previous.all_probs.unsqueeze(1).unbind(0);
        auto run = [&](bool token_only) {
            FastTopKSamplerOutput result;
            std::vector<torch::Tensor> tokens{previous.token_ids};
            std::vector<torch::Tensor> probs;
            for (int step = 0; step < 2; ++step) {
                if (token_only) {
                    tokens.push_back(sampler.forwardTokenIds(logits));
                } else {
                    auto out = sampler.forward(logits, 1);
                    tokens.push_back(out.token_ids);
                    probs.push_back(out.all_probs.unsqueeze(1));
                }
            }
            if (!token_only) {
                probs.insert(probs.begin(), torch::stack(carried_probs, 0));
                result.all_probs = torch::cat(probs, 1);
            }
            result.token_ids = torch::cat(tokens, 1);
            return result;
        };
        auto dense = run(false);
        auto lean = run(true);
        ASSERT_TRUE(torch::equal(dense.token_ids, lean.token_ids));
        ASSERT_FALSE(lean.all_probs.defined());
        for (int i = 0; i < 10; ++i) {
            run(false);
            run(true);
        }
        // Alternate order to expose drift; timing includes host submission and GPU drain.
        for (int round = 0; round < 4; ++round) {
            for (int order = 0; order < 2; ++order) {
                bool token_only = (round + order) % 2 != 0;
                torch::cuda::synchronize();
                auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < iterations; ++i) {
                    run(token_only);
                }
                torch::cuda::synchronize();
                auto us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count();
                std::cout << "TOKEN_ONLY_BENCH batch=" << batch << " round=" << round
                          << " token_only=" << token_only << " us_per_cycle=" << us / iterations << std::endl;
            }
        }
    }
}

}  // namespace
}  // namespace speculative
}  // namespace rtp_llm
