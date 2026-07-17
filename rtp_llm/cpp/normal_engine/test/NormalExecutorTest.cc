#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"

#include "gtest/gtest.h"

namespace rtp_llm {

TEST(NormalExecutorTest, SkipSamplingForFakeStreams) {
    EXPECT_TRUE(detail::shouldSkipNormalSampling(/*tp_rank=*/0,
                                                 /*warm_up=*/false,
                                                 /*stream_count=*/1,
                                                 /*is_fake_stream=*/true));
    EXPECT_FALSE(detail::shouldSkipNormalSampling(/*tp_rank=*/0,
                                                  /*warm_up=*/false,
                                                  /*stream_count=*/1,
                                                  /*is_fake_stream=*/false));
}

TEST(NormalExecutorTest, PreserveExistingSkipConditions) {
    EXPECT_TRUE(detail::shouldSkipNormalSampling(/*tp_rank=*/1,
                                                 /*warm_up=*/false,
                                                 /*stream_count=*/1,
                                                 /*is_fake_stream=*/false));
    EXPECT_TRUE(detail::shouldSkipNormalSampling(/*tp_rank=*/0,
                                                 /*warm_up=*/true,
                                                 /*stream_count=*/1,
                                                 /*is_fake_stream=*/false));
    EXPECT_TRUE(detail::shouldSkipNormalSampling(/*tp_rank=*/0,
                                                 /*warm_up=*/false,
                                                 /*stream_count=*/0,
                                                 /*is_fake_stream=*/false));
}

}  // namespace rtp_llm
