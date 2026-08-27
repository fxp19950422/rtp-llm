#include <gtest/gtest.h>
#include <stdexcept>
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace test {

class CPSlotMapperTest: public ::testing::Test {};

TEST_F(CPSlotMapperTest, DefaultConstructorIsNotSharded) {
    CPSlotMapper mapper;
    EXPECT_FALSE(mapper.isSharded());  // cp_size=1 → not sharded
    EXPECT_EQ(mapper.cpRank(), 0);
    EXPECT_EQ(mapper.cpSize(), 1);
    EXPECT_EQ(mapper.blockSize(), 1);
    EXPECT_EQ(mapper.virtualBlockSize(), 1);
}

TEST_F(CPSlotMapperTest, SingleRankIsNotSharded) {
    CPSlotMapper mapper(0, 1, 32);
    EXPECT_FALSE(mapper.isSharded());  // cp_size=1 → not sharded
}

TEST_F(CPSlotMapperTest, MultiRankIsSharded) {
    CPSlotMapper mapper(0, 2, 32);
    EXPECT_TRUE(mapper.isSharded());           // cp_size=2 → sharded
    EXPECT_EQ(mapper.virtualBlockSize(), 64);  // block_size * cp_size
}

TEST_F(CPSlotMapperTest, RejectsInvalidGeometry) {
    EXPECT_THROW(CPSlotMapper(0, 0, 32), std::invalid_argument);
    EXPECT_THROW(CPSlotMapper(0, 2, 0), std::invalid_argument);
    EXPECT_THROW(CPSlotMapper(-1, 2, 32), std::invalid_argument);
    EXPECT_THROW(CPSlotMapper(2, 2, 32), std::invalid_argument);
}

TEST_F(CPSlotMapperTest, LocalBlockCount) {
    const int block_size = 4;

    // cp_size=2: localBlockCount = ceil(total_blocks / cp_size), same for all ranks
    CPSlotMapper rank0(0, 2, block_size);
    CPSlotMapper rank1(1, 2, block_size);

    // seq_len=0: 0 total blocks -> 0
    EXPECT_EQ(rank0.localBlockCount(0), 0);
    EXPECT_EQ(rank1.localBlockCount(0), 0);

    // seq_len=4: 1 total block -> ceil(1/2)=1
    EXPECT_EQ(rank0.localBlockCount(4), 1);
    EXPECT_EQ(rank1.localBlockCount(4), 1);

    // seq_len=8: 2 total blocks -> ceil(2/2)=1
    EXPECT_EQ(rank0.localBlockCount(8), 1);
    EXPECT_EQ(rank1.localBlockCount(8), 1);

    // seq_len=12: 3 total blocks -> ceil(3/2)=2
    EXPECT_EQ(rank0.localBlockCount(12), 2);
    EXPECT_EQ(rank1.localBlockCount(12), 2);

    // seq_len=16: 4 total blocks -> ceil(4/2)=2
    EXPECT_EQ(rank0.localBlockCount(16), 2);
    EXPECT_EQ(rank1.localBlockCount(16), 2);

    // seq_len=5: 2 total blocks -> ceil(2/2)=1
    EXPECT_EQ(rank0.localBlockCount(5), 1);
    EXPECT_EQ(rank1.localBlockCount(5), 1);
}

TEST_F(CPSlotMapperTest, LocalBlockCountFourRanks) {
    // seq_len=55, block_size=8, cp_size=4
    // total_blocks = ceil(55/8) = 7, localBlockCount = ceil(7/4) = 2
    // All ranks get 2 — rank3 has 1 unused trailing block
    const int block_size = 8;
    const int cp_size    = 4;

    for (int r = 0; r < cp_size; ++r) {
        CPSlotMapper mapper(r, cp_size, block_size);
        EXPECT_EQ(mapper.localBlockCount(55), 2) << "rank=" << r;
    }
}

TEST_F(CPSlotMapperTest, EffectiveSeqLenForAllocIsRankIndependent) {
    const int    block_size = 4;
    CPSlotMapper rank0(0, 2, block_size);
    CPSlotMapper rank1(1, 2, block_size);

    // effectiveSeqLenForAlloc = ceil(total_blocks / cp_size) * block_size
    // This is rank-independent — always allocates max across all ranks.
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(0), 0);
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(4), 4);   // ceil(1/2)=1 block * 4
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(8), 4);   // ceil(2/2)=1 block * 4
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(12), 8);  // ceil(3/2)=2 blocks * 4
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(16), 8);  // ceil(4/2)=2 blocks * 4

    // Same results for rank1 — rank-independent
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(0), 0);
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(4), 4);
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(8), 4);
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(12), 8);
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(16), 8);
}

TEST_F(CPSlotMapperTest, EffectiveSeqLenFourRanks) {
    // seq_len=55, block_size=8, cp_size=4
    // total_blocks=7, ceil(7/4)=2, effective=16
    // All ranks get the same value
    const int block_size = 8;
    const int cp_size    = 4;

    for (int r = 0; r < cp_size; ++r) {
        CPSlotMapper mapper(r, cp_size, block_size);
        EXPECT_EQ(mapper.effectiveSeqLenForAlloc(55), 16) << "rank=" << r;
    }
}

TEST_F(CPSlotMapperTest, NonShardedPassthrough) {
    CPSlotMapper mapper;  // cp_size=1, block_size=1

    EXPECT_EQ(mapper.localBlockCount(10), 10);
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(10), 10);
}

TEST_F(CPSlotMapperTest, BuildStorePlanUsesPolicyActiveTailBlocks) {
    CPSlotMapper mapper(0, 2, 4);

    auto default_swa = mapper.buildStorePlan(CacheGroupType::SWA,
                                             /*total_logical_blocks=*/5,
                                             /*reuse_block_size=*/0,
                                             /*use_hybrid=*/true);
    ASSERT_EQ(default_swa.size(), 2);
    EXPECT_EQ(default_swa[0].key_index, 3);
    EXPECT_EQ(default_swa[0].offset_index, 1);
    EXPECT_EQ(default_swa[1].key_index, 4);
    EXPECT_EQ(default_swa[1].offset_index, 2);

    CacheGroupPolicy policy   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.active_tail_blocks = 1;
    auto custom_swa           = mapper.buildStorePlan(policy,
                                            /*total_logical_blocks=*/5,
                                            /*reuse_block_size=*/0,
                                            /*use_hybrid=*/true);
    ASSERT_EQ(custom_swa.size(), 1);
    EXPECT_EQ(custom_swa[0].key_index, 4);
    EXPECT_EQ(custom_swa[0].offset_index, 2);
}

TEST_F(CPSlotMapperTest, CompactCpTailSlotsUseGlobalTailKeysForLongRequests) {
    constexpr size_t cache_key_count = 69;
    constexpr size_t slot_count      = 2;
    constexpr int    cp_size         = 8;

    size_t key_index = 0;
    ASSERT_TRUE(compactCpTailSlotKeyIndex(0, slot_count, cache_key_count, cp_size, key_index));
    EXPECT_EQ(key_index, 63u);
    ASSERT_TRUE(compactCpTailSlotKeyIndex(1, slot_count, cache_key_count, cp_size, key_index));
    EXPECT_EQ(key_index, 68u);

    CacheGroupPolicy policy = defaultCacheGroupPolicy(CacheGroupType::SWA);
    CPSlotMapper     mapper(/*cp_rank=*/0, cp_size, /*block_size=*/64);
    const auto       store_plan = mapper.buildStorePlan(
        policy, cache_key_count, /*reuse_block_size=*/0, /*use_hybrid=*/true);
    ASSERT_EQ(store_plan.size(), slot_count);
    EXPECT_EQ(store_plan[0].key_index, 63);
    EXPECT_EQ(store_plan[0].offset_index, 7);
    EXPECT_EQ(store_plan[1].key_index, 68);
    EXPECT_EQ(store_plan[1].offset_index, 8);

    // A one-block request still resolves to its only key, preserving the
    // short-request path that previously hid the missing tail offset.
    ASSERT_TRUE(compactCpTailSlotKeyIndex(0, slot_count, /*cache_key_count=*/1, cp_size, key_index));
    EXPECT_EQ(key_index, 0u);
    EXPECT_FALSE(compactCpTailSlotKeyIndex(1, slot_count, /*cache_key_count=*/1, cp_size, key_index));
}

TEST_F(CPSlotMapperTest, CompactCpDecodePhysicalBlocksUseGlobalTailKeysForLongRequests) {
    constexpr size_t cache_key_count = 69;
    constexpr size_t block_count     = 16;
    constexpr int    cp_size         = 8;

    size_t key_index = 0;
    ASSERT_TRUE(compactCpPhysicalBlockKeyIndex(7, block_count, cache_key_count, cp_size, key_index));
    EXPECT_EQ(key_index, 63u);
    ASSERT_TRUE(compactCpPhysicalBlockKeyIndex(15, block_count, cache_key_count, cp_size, key_index));
    EXPECT_EQ(key_index, 68u);

    EXPECT_FALSE(compactCpPhysicalBlockKeyIndex(6, block_count, cache_key_count, cp_size, key_index));
    EXPECT_FALSE(compactCpPhysicalBlockKeyIndex(16, block_count, cache_key_count, cp_size, key_index));
}

TEST_F(CPSlotMapperTest, CompactCpDecodeTableSlotsUseCanonicalKeysForShortAndLongRequests) {
    constexpr int cp_size = 8;
    size_t        key_index = 0;

    // Decode allocates one compact fixed-group block for a short request.
    // Slot zero must load its only canonical key instead of being rejected
    // for not looking like an uncompressed CP8 position (7, 15, ...).
    ASSERT_TRUE(compactCpDecodeTableSlotKeyIndex(
        /*slot_index=*/0, /*slot_count=*/1, /*base_cache_key_count=*/1, cp_size, key_index));
    EXPECT_EQ(key_index, 0u);

    // A partial base CP span is still fixed-group key zero. This is the
    // 516-token regression: the FULL namespace has two keys while the fixed
    // SWA/STATE namespace has only one.
    ASSERT_TRUE(compactCpDecodeTableSlotKeyIndex(
        /*slot_index=*/0, /*slot_count=*/1, /*base_cache_key_count=*/2, cp_size, key_index));
    EXPECT_EQ(key_index, 0u);
    ASSERT_TRUE(compactCpDecodeTableSlotKeyIndex(
        /*slot_index=*/0, /*slot_count=*/1, /*base_cache_key_count=*/4, cp_size, key_index));
    EXPECT_EQ(key_index, 0u);

    // 69 base keys coarsen to nine fixed-group keys; two retained table slots
    // map to the fixed-group tail keys seven and eight, matching Prefill's
    // buildCacheStorePlan input namespace.
    ASSERT_TRUE(compactCpDecodeTableSlotKeyIndex(
        /*slot_index=*/0, /*slot_count=*/2, /*base_cache_key_count=*/69, cp_size, key_index));
    EXPECT_EQ(key_index, 7u);
    ASSERT_TRUE(compactCpDecodeTableSlotKeyIndex(
        /*slot_index=*/1, /*slot_count=*/2, /*base_cache_key_count=*/69, cp_size, key_index));
    EXPECT_EQ(key_index, 8u);
}

TEST_F(CPSlotMapperTest, FullGroupIgnoresByteSlicePolicy) {
    CacheConfig config;
    config.seq_size_per_block = 8;
    config.layer_num          = 1;
    config.layer_all_num      = 1;

    auto full_spec = std::make_shared<MHAKVCacheSpec>();
    full_spec->tag = "full";
    GroupBase full_group;
    full_group.tag               = full_spec->tag;
    full_group.spec              = full_spec;
    full_group.layer_ids         = {0};
    full_group.policy            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_group.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    full_group.policy.cp_slice   = CpBlockSliceMode::EQUAL_BYTES;

    auto swa_spec = std::make_shared<MHAKVCacheSpec>();
    swa_spec->tag = "swa";
    GroupBase swa_group;
    swa_group.tag             = swa_spec->tag;
    swa_group.spec            = swa_spec;
    swa_group.layer_ids       = {0};
    swa_group.policy          = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_group.policy.cp_slice = CpBlockSliceMode::EQUAL_BYTES;
    config.setTopology({std::move(full_group), std::move(swa_group)}, {{0, {"full", "swa"}}});

    CPSlotMapper mapper(0, 2, 8);

    EXPECT_EQ(mapper.layoutForGroup(config, 0).mapping, CpBlockMappingMode::BLOCK_ROUND_ROBIN);
    EXPECT_EQ(mapper.layoutForGroup(config, 0).slice, CpBlockSliceMode::NONE);
    EXPECT_EQ(mapper.layoutForGroup(config, 1).slice, CpBlockSliceMode::EQUAL_BYTES);
}

}  // namespace test
}  // namespace rtp_llm
