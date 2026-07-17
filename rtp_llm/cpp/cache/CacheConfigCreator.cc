#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <cstdlib>
#include <numeric>
#include <string>

#include "rtp_llm/cpp/cache/HybridConfigCreator.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/SingleConfigCreator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

bool envEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return false;
    }
    const std::string normalized(value);
    return normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on";
}

void configureCpKvLayout(CacheConfig&             config,
                         const ModelConfig&       model_config,
                         const ParallelismConfig& parallelism_config) {
    const bool        shard_enabled = envEnabled("RTP_LLM_CP_KV_SHARD_DECODE");
    const char*       layout_env    = std::getenv("RTP_LLM_CP_KV_LAYOUT");
    const std::string requested     = layout_env == nullptr ? "" : std::string(layout_env);
    if (!shard_enabled && requested.empty()) {
        return;
    }

    const bool cp_enabled = parallelism_config.prefill_cp_config.is_enabled() && parallelism_config.tp_size > 1;
    RTP_LLM_CHECK_WITH_INFO(shard_enabled && cp_enabled,
                            "CP KV shard requires RTP_LLM_CP_KV_SHARD_DECODE=1 and enabled CP with tp_size>1");

    config.cp_shard_size                  = static_cast<int>(parallelism_config.tp_size);
    config.cp_kv_layout.kind              = CpKvLayoutKind::LEGACY_DYNAMIC_CONTIGUOUS;
    config.cp_kv_layout.cp_size           = static_cast<int>(parallelism_config.tp_size);
    config.cp_kv_layout.cp_rank           = static_cast<int>(parallelism_config.tp_rank);
    config.cp_kv_layout.page_size         = static_cast<int>(config.seq_size_per_block);
    config.cp_kv_layout.layout_version    = 0;
    config.cp_kv_layout.max_global_length = model_config.max_seq_len;

    if (requested.empty()) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(
        requested == "page_interleaved", "unsupported RTP_LLM_CP_KV_LAYOUT='%s'", requested.c_str());
    RTP_LLM_CHECK_WITH_INFO(CacheConfigCreator::supportsPageInterleavedCpKv(model_config, parallelism_config),
                            "page_interleaved CP KV is only supported for GLM4.7 PPU MHA CP-shard");
    RTP_LLM_CHECK_WITH_INFO(config.cp_kv_layout.page_size > 0, "page_interleaved CP KV requires a positive page size");

    config.cp_kv_layout.kind            = CpKvLayoutKind::PAGE_INTERLEAVED;
    config.cp_kv_layout.interleave_size = config.cp_kv_layout.page_size;
    config.cp_kv_layout.layout_version  = 1;
}

}  // namespace

bool CacheConfigCreator::supportsPageInterleavedCpKv(const ModelConfig&       model_config,
                                                     const ParallelismConfig& parallelism_config) {
#if USE_PPU
    const bool is_glm4_moe = model_config.model_type == "glm4_moe" || model_config.model_type == "glm4_moe-mtp"
                             || model_config.model_type == "glm4_moe_mtp";
    return is_glm4_moe && !model_config.attn_config.use_mla && parallelism_config.prefill_cp_config.is_enabled()
           && parallelism_config.tp_size > 1;
#else
    (void)model_config;
    (void)parallelism_config;
    return false;
#endif
}

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  bool                     is_mtp) {
    if (model_config.hybrid_attention_config.enable_hybrid_attention) {
        return HybridConfigCreator::createHybridConfig(model_config, parallelism_config, is_mtp);
    } else {
        return SingleConfigCreator::createSingleConfig(model_config, parallelism_config, is_mtp);
    }
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config    = CacheConfigCreator::createBasicConfig(model_config, parallelism_config);
    uint32_t    block_num = 0;

    config.linear_step = kv_cache_config.linear_step;
    if (kv_cache_config.kernel_seq_size_per_block > 0) {
        RTP_LLM_CHECK_WITH_INFO(kv_cache_config.seq_size_per_block % kv_cache_config.kernel_seq_size_per_block == 0,
                                "seq_size_per_block(%d) must be divisible by kernel_seq_size_per_block(%d)",
                                kv_cache_config.seq_size_per_block,
                                kv_cache_config.kernel_seq_size_per_block);
        config.kernel_seq_size_per_block = static_cast<size_t>(kv_cache_config.kernel_seq_size_per_block);
    } else {
        // Default: kernel block size == physical block size (no split).
        config.kernel_seq_size_per_block = config.seq_size_per_block;
    }

    configureCpKvLayout(config, model_config, parallelism_config);
    if (config.cp_shard_size > 1) {
        RTP_LLM_LOG_INFO("CP KV shard enabled: cp_size=%d cp_rank=%d layout=%d page_size=%d max_global_length=%ld",
                         config.cp_kv_layout.cp_size,
                         config.cp_kv_layout.cp_rank,
                         static_cast<int>(config.cp_kv_layout.kind),
                         config.cp_kv_layout.page_size,
                         config.cp_kv_layout.max_global_length);
    }

    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
        block_num = kv_cache_mem_size / config.block_size_bytes;
    }
    RTP_LLM_CHECK_WITH_INFO(block_num > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_num,
                            static_cast<long>(config.block_size_bytes / 1024 / 1024));

    const auto local_kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    const auto kv_cache_seq_len       = local_kv_cache_seq_len * static_cast<size_t>(config.cp_shard_size);
    config.block_num                  = static_cast<int>(block_num);
    RTP_LLM_LOG_INFO("kv cache block nums is %u, allows storing %ld tokens", block_num, kv_cache_seq_len);
    if (kv_cache_seq_len < model_config.max_seq_len) {
        RTP_LLM_LOG_WARNING("kv cache block nums %u can only store %ld tokens, less than max_seq_len %ld, "
                            "this is dangerous, consider decrease max_seq_len",
                            block_num,
                            kv_cache_seq_len,
                            model_config.max_seq_len);
    }
    return config;
}

CacheConfig CacheConfigCreator::createSpConfig(const ModelConfig&                 score_model_config,
                                               const ModelConfig&                 propose_model_config,
                                               const ParallelismConfig&           parallelism_config,
                                               const RuntimeConfig&               runtime_config,
                                               const KVCacheConfig&               kv_cache_config,
                                               const SpeculativeExecutionConfig&  sp_config,
                                               const std::optional<WarmUpResult>& warm_up_result,
                                               bool                               is_mtp,
                                               bool                               is_eagle) {
    CacheConfig score_config = CacheConfigCreator::createBasicConfig(score_model_config, parallelism_config, false);
    CacheConfig propose_config =
        CacheConfigCreator::createBasicConfig(propose_model_config, parallelism_config, is_mtp);

    if (kv_cache_config.kernel_seq_size_per_block > 0) {
        const size_t kernel_seq_size_per_block = static_cast<size_t>(kv_cache_config.kernel_seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(score_config.seq_size_per_block % kernel_seq_size_per_block == 0,
                                "score seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu)",
                                score_config.seq_size_per_block,
                                kernel_seq_size_per_block);
        RTP_LLM_CHECK_WITH_INFO(propose_config.seq_size_per_block % kernel_seq_size_per_block == 0,
                                "propose seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu)",
                                propose_config.seq_size_per_block,
                                kernel_seq_size_per_block);
        score_config.kernel_seq_size_per_block   = kernel_seq_size_per_block;
        propose_config.kernel_seq_size_per_block = kernel_seq_size_per_block;
    } else {
        // Default: kernel block size == physical block size (no split).
        score_config.kernel_seq_size_per_block   = score_config.seq_size_per_block;
        propose_config.kernel_seq_size_per_block = propose_config.seq_size_per_block;
    }

    configureCpKvLayout(score_config, score_model_config, parallelism_config);
    if (is_mtp) {
        configureCpKvLayout(propose_config, propose_model_config, parallelism_config);
    }

    int num_mtp_modules = 1;
    if (is_mtp) {
        // Decode steps reuse the same physical draft layers; they do not need separate KV caches.
        num_mtp_modules = static_cast<int>(propose_config.layer_num);
        if (is_eagle) {
            num_mtp_modules = 1;
        }
    }

    uint32_t total_layer_num = score_config.layer_num;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_layer_num += propose_config.layer_num;
    }

    size_t total_block_size_bytes = score_config.block_size_bytes;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_block_size_bytes += propose_config.block_size_bytes;
    }

    size_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);

        block_num = kv_cache_mem_size
                    / (static_cast<size_t>(score_config.block_size_bytes)
                       + static_cast<size_t>(propose_config.block_size_bytes) * static_cast<size_t>(num_mtp_modules));
    }

    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    CacheConfig config      = score_config;
    config.linear_step      = std::max(1, kv_cache_config.linear_step);
    config.layer_all_num    = total_layer_num;
    config.block_size_bytes = total_block_size_bytes;
    // config.block_size       = config.block_size_bytes / rtp_llm::getTypeSize(config.dtype);
    config.block_num = block_num;

    const uint32_t main_layer_num = score_config.layer_num;
    const uint32_t mtp_layer_num  = propose_config.layer_num;

    size_t full_gid = 0;
    if (config.group_types.size() > 1) {
        for (size_t gid = 0; gid < config.group_types.size(); ++gid) {
            if (config.group_types[gid] == CacheGroupType::FULL) {
                full_gid = gid;
                break;
            }
        }
    }

    // Each sub-model needs an independent CacheConfig because global_layer_ids differs per module.
    config.mtp_sub_configs.clear();
    config.mtp_sub_configs.reserve(num_mtp_modules);
    config.layer_to_group_id.resize(total_layer_num, 0);
    config.layer_attn_types.resize(total_layer_num, CacheGroupType::FULL);
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(total_layer_num), 0);

    // Main(score) model per-layer stride (kv + scale).
    // This is expected to be fully populated by createBasicConfig() (Single/Hybrid creators).
    const size_t score_layers = static_cast<size_t>(main_layer_num);
    RTP_LLM_CHECK_WITH_INFO(score_config.layer_to_block_stride_bytes.size() == score_layers,
                            "score_config.layer_to_block_stride_bytes size mismatch, got=%zu need=%zu",
                            score_config.layer_to_block_stride_bytes.size(),
                            score_layers);
    for (size_t l = 0; l < score_layers; ++l) {
        config.layer_to_block_stride_bytes[l] = score_config.layer_to_block_stride_bytes[l];
        if (l < score_config.layer_attn_types.size()) {
            config.layer_attn_types[l] = score_config.layer_attn_types[l];
        }
    }

    for (int m = 0; m < num_mtp_modules; ++m) {
        auto sub_cfg           = std::make_shared<CacheConfig>(propose_config);
        sub_cfg->block_num     = block_num;
        sub_cfg->layer_all_num = sub_cfg->layer_num;

        sub_cfg->global_layer_ids.clear();
        sub_cfg->global_layer_ids.resize(1);
        sub_cfg->global_layer_ids[0].resize(mtp_layer_num);
        RTP_LLM_CHECK_WITH_INFO(sub_cfg->layer_to_block_stride_bytes.size() == static_cast<size_t>(mtp_layer_num),
                                "sub_cfg.layer_to_block_stride_bytes size mismatch, got=%zu need=%u",
                                sub_cfg->layer_to_block_stride_bytes.size(),
                                mtp_layer_num);
        for (size_t l = 0; l < mtp_layer_num; ++l) {
            int global_layer_id                       = main_layer_num + m * mtp_layer_num + l;
            sub_cfg->global_layer_ids[0][l]           = global_layer_id;
            config.layer_to_group_id[global_layer_id] = static_cast<int>(full_gid);
            config.global_layer_ids[full_gid].push_back(global_layer_id);

            const int stride_bytes = sub_cfg->layer_to_block_stride_bytes[static_cast<size_t>(l)];
            config.layer_to_block_stride_bytes[static_cast<size_t>(global_layer_id)] = stride_bytes;
            if (l < sub_cfg->layer_attn_types.size()) {
                config.layer_attn_types[static_cast<size_t>(global_layer_id)] = sub_cfg->layer_attn_types[l];
            }
        }

        sub_cfg->layer_to_group_id.assign(static_cast<size_t>(sub_cfg->layer_num), static_cast<int>(full_gid));
        config.mtp_sub_configs.push_back(sub_cfg);
    }

    const auto local_kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    const auto kv_cache_seq_len       = local_kv_cache_seq_len * static_cast<size_t>(config.cp_shard_size);
    RTP_LLM_LOG_INFO("CacheConfig created: is_mtp=%d, total_layers=%u, num_mtp_modules=%d, block_num=%zu, "
                     "allows storing %zu tokens, total_block_size=%zu bytes (main=%zu + %d*propose=%zu)",
                     is_mtp,
                     total_layer_num,
                     num_mtp_modules,
                     block_num,
                     kv_cache_seq_len,
                     total_block_size_bytes,
                     score_config.block_size_bytes,
                     num_mtp_modules,
                     propose_config.block_size_bytes);

    RTP_LLM_LOG_INFO("CacheConfig debugString(main_score_model):\n%s", score_config.debugString().c_str());
    for (size_t i = 0; i < config.mtp_sub_configs.size(); ++i) {
        const auto& sub = config.mtp_sub_configs[i];
        RTP_LLM_LOG_INFO("CacheConfig debugString(sub_propose_model[%zu]):\n%s", i, sub->debugString().c_str());
    }

    return config;
}

}  // namespace rtp_llm
