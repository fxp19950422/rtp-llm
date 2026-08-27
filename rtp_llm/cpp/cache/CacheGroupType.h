#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtp_llm {

// Cache group type for hybrid KV-cache:
// - LINEAR: linear attention group (PD cache-store transfer keeps the last block)
// - FULL: full attention group (all blocks are needed for cache-store transfer)
// - SWA: sliding-window attention group (PD cache-store transfer keeps the last two blocks)
enum class CacheGroupType : int8_t {
    LINEAR = 0,
    FULL   = 1,
    SWA    = 2,
};

enum class CacheReusePolicy : int8_t {
    REUSABLE     = 0,
    NON_REUSABLE = 1,
};

enum class CacheEvictPolicy : int8_t {
    CHAIN       = 0,
    INDEPENDENT = 1,
    NONE        = 2,
};

enum class CacheMemoryPlacement : int8_t {
    DEVICE      = 0,
    HOST        = 1,
    HOST_PINNED = 2,
};

enum class CpBlockMappingMode : int8_t {
    NONE              = 0,
    BLOCK_ROUND_ROBIN = 1,
    COMPACT_LAST_RANK = 2,
};

enum class CpBlockSliceMode : int8_t {
    NONE          = 0,
    EQUAL_BYTES   = 1,
    PAYLOAD_BYTES = 2,
};

struct CacheGroupPolicy {
    CacheGroupType       group_type             = CacheGroupType::FULL;
    bool                 enable_prefix_reuse    = true;
    CacheEvictPolicy     evict_policy           = CacheEvictPolicy::CHAIN;
    bool                 reservable             = true;
    uint32_t             explicit_block_num     = 0;
    bool                 charge_to_paged_budget = false;
    CacheMemoryPlacement memory_placement       = CacheMemoryPlacement::DEVICE;
    uint32_t             active_tail_blocks     = 0;
    bool                 validate_tail_blocks   = true;
    CpBlockMappingMode   cp_mapping             = CpBlockMappingMode::NONE;
    CpBlockSliceMode     cp_slice               = CpBlockSliceMode::NONE;
};

// One cache-store registration step: pair a cache key from the full logical
// namespace with a slot in the tag-local physical block table. Under CP,
// both round-robin FULL groups and compact STATE/SWA groups use local slots.
struct CacheStoreBlockPair {
    int key_index;
    int offset_index;
};

// Map a slot in a compact CP block table back into the full canonical key
// namespace. Fixed STATE/SWA tables only retain tail slots, so slot zero is
// not necessarily compact block zero for a long request.
inline bool compactCpTailSlotKeyIndex(size_t slot_index,
                                      size_t slot_count,
                                      size_t cache_key_count,
                                      int    cp_size,
                                      size_t& cache_key_index) {
    if (cache_key_count == 0 || slot_count == 0 || cp_size <= 0 || slot_index >= slot_count) {
        return false;
    }
    const size_t cp_size_t          = static_cast<size_t>(cp_size);
    const size_t compact_key_count  = (cache_key_count + cp_size_t - 1) / cp_size_t;
    const size_t first_compact_slot = compact_key_count > slot_count ? compact_key_count - slot_count : 0;
    const size_t compact_index      = first_compact_slot + slot_index;
    if (compact_index >= compact_key_count) {
        return false;
    }
    cache_key_index = std::min((compact_index + 1) * cp_size_t - 1, cache_key_count - 1);
    return true;
}

// Decode receives cache_keys in the base FULL-group namespace, while a
// COMPACT_LAST_RANK table represents one additional CP-width coarsening of
// that namespace. Reconstruct the group-local logical key count before using
// the common compact-tail projection. This matters for partially filled CP
// spans: e.g. two or four base keys still correspond to fixed-group key zero,
// not base key one or three.
inline bool compactCpDecodeTableSlotKeyIndex(size_t slot_index,
                                             size_t slot_count,
                                             size_t base_cache_key_count,
                                             int    cp_size,
                                             size_t& cache_key_index) {
    if (base_cache_key_count == 0 || cp_size <= 0) {
        return false;
    }
    const size_t cp_size_t            = static_cast<size_t>(cp_size);
    const size_t group_cache_key_count = (base_cache_key_count + cp_size_t - 1) / cp_size_t;
    return compactCpTailSlotKeyIndex(slot_index, slot_count, group_cache_key_count, cp_size, cache_key_index);
}

// Decode keeps compact CP groups in a base-block table whose usable entries
// are the last block of each CP-width span (for CP8: 7, 15, ...). Convert that
// physical position to the compact slot ordinal before resolving the global
// canonical tail key.
inline bool compactCpPhysicalBlockKeyIndex(size_t block_pos,
                                           size_t block_count,
                                           size_t cache_key_count,
                                           int    cp_size,
                                           size_t& cache_key_index) {
    if (cp_size <= 0 || block_count == 0) {
        return false;
    }
    const size_t cp_size_t = static_cast<size_t>(cp_size);
    if (block_pos >= block_count || block_pos % cp_size_t != cp_size_t - 1) {
        return false;
    }
    const size_t slot_index = block_pos / cp_size_t;
    const size_t slot_count = (block_count + cp_size_t - 1) / cp_size_t;
    return compactCpTailSlotKeyIndex(slot_index, slot_count, cache_key_count, cp_size, cache_key_index);
}

// Keep cache-store projection header-only so bindings that consume ExecOps.cc
// as a source file do not need to link the full CPSlotMapper implementation.
inline std::vector<CacheStoreBlockPair> buildCacheStorePlan(const CacheGroupPolicy& policy,
                                                            size_t                  total_logical_blocks,
                                                            size_t                  reuse_block_size,
                                                            bool                    use_hybrid,
                                                            int                     cp_rank,
                                                            int                     cp_size) {
    std::vector<CacheStoreBlockPair> plan;
    if (total_logical_blocks == 0) {
        return plan;
    }

    const bool block_round_robin = policy.cp_mapping == CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    const bool compact_last_rank = policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK;
    const bool sharded_full      = cp_size > 1 && block_round_robin;
    const bool compact_swa_by_cp = cp_size > 1 && compact_last_rank;
    if (compact_swa_by_cp) {
        const size_t cp_size_t        = static_cast<size_t>(cp_size);
        const size_t canonical_blocks = (total_logical_blocks + cp_size_t - 1) / cp_size_t;
        const size_t tail_count       = std::max<size_t>(1, policy.active_tail_blocks);
        const size_t start = use_hybrid ? (canonical_blocks > tail_count ? canonical_blocks - tail_count : 0) :
                                          std::min(reuse_block_size, canonical_blocks);
        plan.reserve(canonical_blocks - start);
        for (size_t compact_idx = start; compact_idx < canonical_blocks; ++compact_idx) {
            size_t key_index = 0;
            const bool mapped = compactCpTailSlotKeyIndex(
                compact_idx, canonical_blocks, total_logical_blocks, cp_size, key_index);
            if (!mapped) {
                continue;
            }
            plan.push_back({static_cast<int>(key_index), static_cast<int>(compact_idx)});
        }
        return plan;
    }

    const bool   transfer_tail_blocks = policy.active_tail_blocks > 0;
    const size_t tail_count           = std::max<size_t>(1, policy.active_tail_blocks);
    size_t       start                = use_hybrid ? 0 : reuse_block_size;
    if (use_hybrid && transfer_tail_blocks) {
        start = total_logical_blocks > tail_count ? total_logical_blocks - tail_count : 0;
    }
    plan.reserve(total_logical_blocks - std::min(start, total_logical_blocks));
    for (size_t pos = start; pos < total_logical_blocks; ++pos) {
        const int block_pos = static_cast<int>(pos);
        if (sharded_full && block_pos % cp_size != cp_rank) {
            continue;
        }
        plan.push_back({block_pos, sharded_full ? block_pos / cp_size : block_pos});
    }
    return plan;
}

inline const char* cacheGroupTypeName(CacheGroupType group_type) {
    switch (group_type) {
        case CacheGroupType::LINEAR:
            return "LINEAR";
        case CacheGroupType::FULL:
            return "FULL";
        case CacheGroupType::SWA:
            return "SWA";
    }
    return "UNKNOWN";
}

inline const char* cacheEvictPolicyName(CacheEvictPolicy evict_policy) {
    switch (evict_policy) {
        case CacheEvictPolicy::CHAIN:
            return "chain";
        case CacheEvictPolicy::INDEPENDENT:
            return "independent";
        case CacheEvictPolicy::NONE:
            return "none";
    }
    return "unknown";
}

inline CacheGroupPolicy defaultCacheGroupPolicy(CacheGroupType group_type) {
    CacheGroupPolicy policy;
    policy.group_type          = group_type;
    policy.enable_prefix_reuse = group_type == CacheGroupType::FULL || group_type == CacheGroupType::LINEAR;
    policy.active_tail_blocks  = group_type == CacheGroupType::LINEAR ? 1 : (group_type == CacheGroupType::SWA ? 2 : 0);
    if (group_type == CacheGroupType::FULL) {
        policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    } else if (group_type == CacheGroupType::SWA) {
        policy.cp_mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
    }
    return policy;
}

}  // namespace rtp_llm
