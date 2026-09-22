#include "rtp_llm/cpp/engine_base/stream/StreamCacheResource.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/cache/AsyncContext.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"
#include "rtp_llm/cpp/config/RoleTypes.h"
#include "rtp_llm/cpp/engine_base/stream/CompleteTokenIds.h"
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"
#include <algorithm>
#include <limits>
#include <thread>

using namespace std;

namespace rtp_llm {

namespace {

std::shared_ptr<const CacheTopology> warmupCacheTopology() {
    static const auto topology = []() {
        constexpr auto kWarmupCacheTag = "__warmup__";
        auto           spec            = std::make_shared<MHAKVCacheSpec>();
        spec->tag                      = kWarmupCacheTag;

        GroupBase group;
        group.tag                       = kWarmupCacheTag;
        group.spec                      = std::move(spec);
        group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
        group.layer_ids                 = {0};
        group.seq_size_per_block        = 1;
        group.kernel_seq_size_per_block = 1;

        return CacheTopology::create({std::move(group)}, {{0, {kWarmupCacheTag}}});
    }();
    return topology;
}

}  // namespace

// ----------------------------- KVCacheConnectorReadWriteContextImpl -----------------------------

class KVCacheConnectorReadWriteContextImpl: public KVCacheConnectorReadWriteContext {
public:
    KVCacheConnectorReadWriteContextImpl(const std::shared_ptr<BatchKVCacheResource>& batch_resource,
                                         const std::shared_ptr<Meta>&                 meta):
        batch_resource_(batch_resource), meta_(meta) {}
    ~KVCacheConnectorReadWriteContextImpl() override = default;

public:
    const KVCacheResource& kvCacheResource() const override {
        return batch_resource_->cacheResource(0);
    }
    const std::shared_ptr<Meta>& meta() const override {
        return meta_;
    }

private:
    std::shared_ptr<BatchKVCacheResource> batch_resource_;
    std::shared_ptr<Meta>                 meta_;
};

class MetaImpl: public Meta {
public:
    MetaImpl(bool enable_memory_cache, bool enable_remote_cache, std::string trace_id):
        enable_memory_cache_(enable_memory_cache), enable_remote_cache_(enable_remote_cache), trace_id_(trace_id) {}
    virtual ~MetaImpl() = default;

public:
    bool enableMemoryCache() const override {
        return enable_memory_cache_;
    }
    bool enableRemoteCache() const override {
        return enable_remote_cache_;
    }
    const std::string& trace_id() const override {
        return trace_id_;
    }
    const std::string& unique_id() const override {
        return unique_id_;
    }
    const std::vector<int64_t>& tokens() const override {
        return tokens_;
    }

    // P2P read extension field
    GenerateStream* generateStream() const override {
        return generate_stream_;
    }

    // P2P routing context: cached at construction time, read-only access thereafter
    std::optional<P2PRoutingContext> p2pRouting() const override {
        if (!routing_ctx_.has_value()) {
            return std::nullopt;
        }
        return routing_ctx_;
    }

    // Fill routing context once from GenerateStream (called after construction)
    void fillRoutingContext(GenerateStream* stream) {
        if (stream && !routing_ctx_.has_value()) {
            auto& ctx           = routing_ctx_.emplace();
            ctx.request_id      = stream->streamId();
            ctx.unique_key      = stream->uniqueKey();
            ctx.deadline_ms     = stream->deadlineMs();
            ctx.prefill_addr    = stream->prefillAddr();
            ctx.prefill_tp_size = stream->getPrefillTpSize();
        }
    }

    GenerateStream*                  generate_stream_ = nullptr;
    std::optional<P2PRoutingContext> routing_ctx_;

private:
    bool                 enable_memory_cache_{false};
    bool                 enable_remote_cache_{false};
    std::string          trace_id_;
    std::string          unique_id_ = "";
    std::vector<int64_t> tokens_;  // TODO : get tokens (remote connector)
};

// ----------------------------- P2P Side-Channel Apply -----------------------------

// Extract P2P side-channel payload from FusedAsyncReadContext and apply to GenerateStream.
// Returns true if P2P payload was found and applied, false otherwise.
static bool tensorPbHasPayload(const TensorPB& tensor_pb) {
    return !tensor_pb.fp32_data().empty() || !tensor_pb.fp16_data().empty() || !tensor_pb.bf16_data().empty()
           || !tensor_pb.int32_data().empty();
}

static bool applyP2PSideChannelToStream(const std::shared_ptr<FusedAsyncReadContext>& read_context,
                                        GenerateStream*                               stream) {
    if (!read_context || !stream) {
        return false;
    }

    // Traverse fused read contexts to find P2PConnectorAsyncReadContext
    auto fused_read_ctx = read_context->fusedReadContext();
    if (!fused_read_ctx) {
        return false;
    }

    const P2PSideChannelPayload* payload = nullptr;
    for (const auto& ctx : fused_read_ctx->contexts()) {
        auto p2p_ctx = std::dynamic_pointer_cast<P2PConnectorAsyncReadContext>(ctx);
        if (p2p_ctx) {
            payload = p2p_ctx->sideChannelPayload();
            if (payload) {
                break;
            }
        }
    }
    if (!payload) {
        return false;
    }

    // Apply side-channel data to GenerateStream
    // 1. First token: append to stream
    if (payload->first_token_id >= 0) {
        stream->setIsContextStream(false);
        stream->step();
        auto new_tokens                   = torch::zeros({(int64_t)stream->nextBatchSize(), 1}, torch::kInt32);
        new_tokens.data_ptr<int32_t>()[0] = static_cast<int32_t>(payload->first_token_id);
        stream->incLastOutputPos();
        stream->update({.new_tokens        = new_tokens,
                        .num_new_tokens    = 1,
                        .hidden_states     = {},
                        .logits            = {},
                        .softmax_probs     = {},
                        .cum_log_probs     = {},
                        .all_probs         = {},
                        .loss              = {},
                        .src_batch_indices = {},
                        .all_hidden_states = {}});
        if (stream->nextBatchSize() == 1) {
            const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
            stream->setNormalAsyncDeviceState(GenerateStream::NormalAsyncDeviceState{
                .epoch                 = 0,
                .last_sample_token_gpu = new_tokens.reshape({1}).to(cuda_i32),
                .next_seq_len_gpu      = torch::full({1}, static_cast<int64_t>(stream->seqLength()), cuda_i32),
            });
        }
        RTP_LLM_LOG_DEBUG("applyP2PSideChannel: appended first_token_id=%ld, stream_id=%ld",
                          payload->first_token_id,
                          stream->streamId());
    }

    // 2. Reuse lengths
    if (payload->total_reuse_len > 0) {
        stream->setInitialReuseLength(payload->total_reuse_len);
        stream->setReuseLength(payload->total_reuse_len);
        stream->setLocalReuseLength(payload->local_reuse_len + payload->memory_reuse_len);
        stream->setMtpTokenIndex(payload->total_reuse_len);
        // The legacy P2P "memory" tier is the host tier in the current cache model.
        stream->setHostReuseLength(payload->memory_reuse_len);
        stream->setRemoteReuseLength(payload->remote_reuse_len);
        RTP_LLM_LOG_DEBUG("applyP2PSideChannel: reuse total=%d, local=%d, remote=%d, memory=%d",
                          payload->total_reuse_len,
                          payload->local_reuse_len,
                          payload->remote_reuse_len,
                          payload->memory_reuse_len);
    }

    // 3. Speculative proposal info. Unlike the gRPC handoff, this channel
    // carries real reuse accounting (block 2) and initKVBlock refreshes the
    // positions after allocation, so proposal-less streams keep them intact.
    if (!payload->propose_tokens.empty()) {
        stream->initSpeculativeHandoffPositions();
        stream->setContainProposeToken(true);
        stream->setProposeToken(payload->propose_tokens);

        auto sp_output_buffer          = std::make_shared<SpeculativeExecutorStreamOutput>();
        sp_output_buffer->propose_step = payload->propose_tokens.size() > 0 ? payload->propose_tokens.size() - 1 : 0;
        sp_output_buffer->tokens       = torch::zeros({1, (int64_t)payload->propose_tokens.size()}, torch::kInt32);
        memcpy(sp_output_buffer->tokens.data_ptr<int>(),
               payload->propose_tokens.data(),
               payload->propose_tokens.size() * sizeof(int));

        const auto cuda_i32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        sp_output_buffer->token_ids_are_point_mass = payload->proposal_is_point_mass;
        RTP_LLM_CHECK_WITH_INFO(!payload->proposal_is_point_mass || !tensorPbHasPayload(payload->propose_probs),
                                "point-mass P2P handoff must not carry dense probabilities");
        if (tensorPbHasPayload(payload->propose_probs)) {
            sp_output_buffer->all_probs = TensorPbConvert::pbToTorch(payload->propose_probs).to(torch::kCUDA);
        }
        if (tensorPbHasPayload(payload->propose_hidden)) {
            sp_output_buffer->hidden_states = TensorPbConvert::pbToTorch(payload->propose_hidden).to(torch::kCUDA);
        }

        stream->setSPOutputBuffer(sp_output_buffer);

        if (payload->propose_tokens.size() >= 2) {
            // Both the stream mirror and async state use the same draft-only
            // contract: target column 0 is excluded and every draft is kept.
            auto propose_tokens_gpu              = sp_output_buffer->draftTokens().to(cuda_i32, /*non_blocking=*/true);
            sp_output_buffer->propose_tokens_gpu = propose_tokens_gpu;
            auto accept_len                      = torch::ones({1}, cuda_i32);
            auto accept_tokens  = torch::zeros({1, static_cast<int64_t>(payload->propose_tokens.size())}, cuda_i32);
            accept_tokens[0][0] = sp_output_buffer->tokens[0][0];
            auto next_seq_len   = torch::full({1}, static_cast<int64_t>(stream->seqLength()), cuda_i32);

            stream->setMtpAsyncDeviceState(GenerateStream::MtpAsyncDeviceState{
                .epoch                  = 0,
                .accept_len_gpu         = std::move(accept_len),
                .accept_tokens_gpu      = std::move(accept_tokens),
                .next_seq_len_gpu       = std::move(next_seq_len),
                .propose_tokens_gpu     = std::move(propose_tokens_gpu),
                .last_hidden_states_gpu = sp_output_buffer->hidden_states,
                .draft_all_probs_gpu    = sp_output_buffer->all_probs,
                .previous_seq_len_upper_bound = stream->seqLength(),
                .next_seq_len_upper_bound     = stream->seqLength(),
                .draft_token_ids_are_point_mass = sp_output_buffer->token_ids_are_point_mass,
            });
        }
        RTP_LLM_LOG_DEBUG("applyP2PSideChannel: propose_tokens count=%zu", payload->propose_tokens.size());
    }

    // 4. Position IDs
    if (!payload->position_ids.empty()) {
        auto position_ids = torch::tensor(payload->position_ids, torch::dtype(torch::kInt32).device(torch::kCPU));
        stream->setContextPositionIds(std::move(position_ids));
        RTP_LLM_LOG_DEBUG("applyP2PSideChannel: position_ids count=%zu", payload->position_ids.size());
    }

    return true;
}

// ----------------------------- StreamCacheResource -----------------------------

void StreamCacheResource::init(int batch_size) {
    batch_kv_cache_resource_->resetBatchSize(batch_size);
    // cache manager is null when warmup
    const auto topology = resource_context_.cache_manager ?
                              resource_context_.cache_manager->cacheConfig().topologyPtr() :
                              warmupCacheTopology();
    batch_kv_cache_resource_->initGroups(topology);
    resource_released_ = false;
}

void StreamCacheResource::releaseResource() {
    RTP_LLM_PROFILE_FUNCTION();
    if (!resource_context_.cache_manager) {
        return;
    }
    // Check against double release
    if (resource_released_) {
        RTP_LLM_LOG_ERROR("=== DOUBLE RELEASE CACHE RESOURCE DETECTED ===");
        RTP_LLM_LOG_ERROR("  stream_ ptr:                   %p", static_cast<void*>(stream_));
        RTP_LLM_LOG_ERROR("  stream alive (magic check):    %s",
                          stream_->isStreamAlive() ? "YES" : "NO (stream already destroyed!)");
        if (stream_->isStreamAlive()) {
            RTP_LLM_LOG_ERROR("  stream id:                     %s", stream_->streamLogTag().c_str());
            RTP_LLM_LOG_ERROR("  stream state:                  %s",
                              StreamStateToString(stream_->generate_status_->status).c_str());
            RTP_LLM_LOG_ERROR("  stream hasError:                %d", stream_->hasErrorWithoutLock());
            RTP_LLM_LOG_ERROR("  stream hasNumBeams:            %d", stream_->hasNumBeams());
        }
        RTP_LLM_LOG_ERROR("  batch_kv_cache_resource_ use_count: %ld", batch_kv_cache_resource_.use_count());
        RTP_LLM_LOG_ERROR("  curBlocksNum:                  %d", curBlocksNum());
        RTP_LLM_LOG_ERROR("  need_release_resource:         %d", need_release_resource_);
        RTP_LLM_LOG_ERROR("  fake_inited:                   %d", fake_inited_);
        RTP_LLM_LOG_ERROR("  batch_kv_cache_resource:       %s", batch_kv_cache_resource_->debugString().c_str());
        RTP_LLM_LOG_ERROR("  thread id:                     %lu",
                          std::hash<std::thread::id>{}(std::this_thread::get_id()));
        abort();
    }
    if (allocator_load_context_) {
        if (!resource_context_.cache_manager->abortPendingLoad(allocator_load_context_)) {
            RTP_LLM_LOG_DEBUG("allocator load was already committed or completed before release");
        }
        allocator_load_context_.reset();
    }
    // do not reuse cache from stopped beam search streams, whose states are likely corrupted
    if (!need_release_resource_ && (!stream_->hasNumBeams() || !stream_->hasErrorWithoutLock())) {
        return;
    }
    RTP_LLM_LOG_DEBUG("releaseResource: stream=%ld, curBlocksNum=%d, pd_kvcache_ref=%p",
                      stream_->streamId(),
                      curBlocksNum(),
                      pd_kvcache_ref_.get());
    tryReleaseKVBlock(curBlocksNum());
    batch_kv_cache_resource_->clearBlocks();
    resource_released_ = true;
}

int StreamCacheResource::tryReleaseKVBlock(size_t nums) {
    RTP_LLM_PROFILE_FUNCTION();
    RTP_LLM_LOG_DEBUG("stream [%s] try release [%lu] blocks", stream_->streamLogTag().c_str(), nums);

    if (fake_inited_) {
        int max_blocks_num = curBlocksNum();
        int batch_size     = batch_kv_cache_resource_->batchSize();
        batch_kv_cache_resource_->clearBlocks();
        batch_kv_cache_resource_->resetBatchSize(batch_size);
        fake_inited_ = false;
        return max_blocks_num;
    }

    // NOTE: Currently only support releasing all blocks
    // Partial release (shrink) is not supported yet
    int total_blocks = curBlocksNum();
    RTP_LLM_CHECK(nums == total_blocks);

    if (total_blocks > 0) {
        if (reuseCache() && !stream_->hasErrorWithoutLock() && stream_->getStatus() == StreamState::FINISHED) {
            const Tier target_tier = storeTarget();
            RTP_LLM_LOG_DEBUG("tryReleaseKVBlock: stream=%ld, storing cache, curBlocksNum=%d, target_tier=%s",
                              stream_->streamId(),
                              total_blocks,
                              tierName(target_tier));
            if (target_tier != Tier::NONE) {
                InsertInfo insert_info{batch_kv_cache_resource_, stream_->completeTokenIdsPtr(), false, target_tier};
                size_t     resident_prefix_length = 0;
                resource_context_.cache_manager->insertIntoCache(insert_info, resident_prefix_length);
            }
        } else {
            RTP_LLM_LOG_DEBUG("tryReleaseKVBlock: stream=%ld, NOT storing cache, reuseCache=%d, hasError=%d, status=%s",
                              stream_->streamId(),
                              reuseCache(),
                              stream_->hasErrorWithoutLock(),
                              StreamStateToString(stream_->getStatus()).c_str());
        }

        FreeInfo free_info{batch_kv_cache_resource_, stream_->completeTokenIdsPtr()};
        free_info.request_id = stream_->streamId();

        resource_context_.cache_manager->free(free_info);
    }

    return total_blocks;
}

// TODO, 等待删除。
int StreamCacheResource::singleBatchNeedBlocks(int seq_len, int reserve_step) const {
    return resource_context_.cache_manager->singleBatchNeedBlocks(batch_kv_cache_resource_, seq_len, reserve_step);
}

int StreamCacheResource::estimatePeakNeedBlocks(
    int seq_len, int common_seq_len, int remaining_tokens, int reserve_step, int target_batch_size) const {
    return resource_context_.cache_manager->estimatePeakNeedBlocks(batch_kv_cache_resource_,
                                                                   seq_len,
                                                                   common_seq_len,
                                                                   remaining_tokens,
                                                                   reserve_step,
                                                                   reuseCache(),
                                                                   target_batch_size);
}

void StreamCacheResource::publishReuseLengths(int total, int host, int disk, int backend) {
    stream_->setReuseLength(total);
    stream_->setMtpTokenIndex(total);
    stream_->setInitialReuseLength(total);
    stream_->setLocalReuseLength(total - backend);
    stream_->setRemoteReuseLength(backend);
    stream_->setHostReuseLength(host);
    stream_->setDiskReuseLength(disk);
}

// TODO(xinfei.sxf) 保证这个函数的原子性
absl::Status StreamCacheResource::initKVBlock() {
    RTP_LLM_PROFILE_FUNCTION();
    // Decode side: first malloc should NOT use device cache, regardless of runtime config.
    // Follow-up allocations (incrKVBlock) use enableCacheLookup().
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource = batch_kv_cache_resource_;
    malloc_info.complete_token_ids      = stream_->completeTokenIdsPtr();
    malloc_info.request_id              = stream_->streamId();
    malloc_info.verbose                 = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;

    const bool disable_first_malloc_reuse =
        resource_context_.cache_manager->cacheConfig().disable_decode_first_malloc_device_reuse;
    const bool is_decode_role  = (resource_context_.role_type == RoleType::DECODE);
    const bool is_first_malloc = (batch_kv_cache_resource_->curBlocksNum() == 0);

    if (disable_first_malloc_reuse && is_decode_role && is_first_malloc) {
        malloc_info.reuse_cache         = false;
        malloc_info.enable_cache_lookup = false;
    } else {
        malloc_info.reuse_cache         = reuseCache();
        malloc_info.enable_cache_lookup = enableCacheLookup();
    }
    malloc_info.enable_remove_skipped_blocks = false;

    MallocResult result = resource_context_.cache_manager->malloc(malloc_info);
    recordCacheReuseMallocResult(result);
    if (!result.success) {
        malloc_failed_times_++;
        switch (result.status) {
            case MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED:
                return absl::UnavailableError("kv cache is temporarily unavailable");
            case MallocStatus::PERMANENT_RESOURCE_EXHAUSTED:
                return absl::ResourceExhaustedError("request exceeds usable kv cache capacity");
            case MallocStatus::INTERNAL_ERROR:
                return absl::InternalError("malloc failed");
            case MallocStatus::NONE:
                RTP_LLM_LOG_ERROR("malloc returned failure without an error status, request_id=%ld",
                                  malloc_info.request_id);
                return absl::InternalError("malloc failed without an error status");
        }
        RTP_LLM_LOG_ERROR("malloc returned failure with unknown status=%d, request_id=%ld",
                          static_cast<int>(result.status),
                          malloc_info.request_id);
        return absl::InternalError("malloc failed with unknown status");
    }

    // HOST/DISK reuse is published after the asynchronous load completes.
    publishReuseLengths(result.reuse_len, 0, 0, 0);
    allocator_load_context_ = std::move(result.async_context);
    return absl::OkStatus();
}

void StreamCacheResource::clearCacheReuseState() {
    stream_->setReuseLength(0);
    stream_->setMtpTokenIndex(0);
    stream_->setInitialReuseLength(0);
    stream_->setHostReuseLength(0);
    stream_->setDiskReuseLength(0);
    stream_->setLocalReuseLength(0);
    stream_->setRemoteReuseLength(0);
}

void StreamCacheResource::recordCacheReuseMallocResult(const MallocResult& result) {
    cache_reuse_metrics_.block_aligned_input_length = result.block_aligned_input_length;
    cache_reuse_metrics_.match_latency_us           = result.match_cost_time_us;
    cache_reuse_metrics_.report_match_latency       = result.match_end_time_us > 0;
    malloc_begin_time_us_                           = result.malloc_begin_time_us;
    cache_reuse_metrics_.load_attempted             = result.load_attempted;
    if (!result.load_attempted) {
        if (result.match_end_time_us > 0) {
            cache_reuse_metrics_.match_to_ready_latency_us =
                std::max<int64_t>(currentTimeUs() - malloc_begin_time_us_, 0);
            cache_reuse_metrics_.report_match_to_ready_latency = true;
        }
        return;
    }

    // Metrics describe the current load attempt. A retry must not inherit the
    // previous attempt's terminal result or latency.
    cache_reuse_metrics_.load_success                  = false;
    cache_reuse_metrics_.report_load_metrics           = false;
    cache_reuse_metrics_.report_load_wait_latency      = false;
    cache_reuse_metrics_.load_wait_latency_us          = 0;
    cache_reuse_metrics_.report_match_to_ready_latency = false;
    cache_reuse_metrics_.match_to_ready_latency_us     = 0;
    cache_reuse_metrics_.load_prepare_latency_us       = result.load_prepare_latency_us;
    load_wait_begin_time_us_                           = currentTimeUs();
    if (!result.success) {
        clearCacheReuseState();
        cache_reuse_metrics_.load_success              = false;
        cache_reuse_metrics_.report_load_metrics       = true;
        cache_reuse_metrics_.match_to_ready_latency_us = std::max<int64_t>(currentTimeUs() - malloc_begin_time_us_, 0);
        cache_reuse_metrics_.report_match_to_ready_latency = true;
    }
}

absl::Status StreamCacheResource::finalizeAllocatorLoad() {
    const bool load_success  = allocator_load_context_->success();
    const auto error         = allocator_load_context_->errorInfo();
    const auto load_context  = std::dynamic_pointer_cast<LoadAsyncContext>(allocator_load_context_);
    const auto malloc_status = load_context == nullptr ? MallocStatus::NONE : load_context->mallocStatus();
    if (load_success) {
        RTP_LLM_CHECK(load_context != nullptr);
        const size_t total = load_context->matchedBlocks();
        const size_t local = load_context->localMatchedBlocks();
        const size_t host  = load_context->matchedBlocks(Tier::HOST);
        const size_t disk  = load_context->matchedBlocks(Tier::DISK);
        RTP_LLM_CHECK_WITH_INFO(total >= local && host <= local && disk <= local - host,
                                "invalid allocator reuse counts: total=%zu local=%zu host=%zu disk=%zu",
                                total,
                                local,
                                host,
                                disk);
        const int tokens = reuseBlockTokens();
        RTP_LLM_CHECK_WITH_INFO(tokens > 0 && total <= static_cast<size_t>(std::numeric_limits<int>::max()) / tokens,
                                "allocator reuse token count exceeds int range: blocks=%zu tokens=%d",
                                total,
                                tokens);
        const size_t backend  = total - local;
        auto&        resource = batch_kv_cache_resource_->cacheResource(0);
        resource.setDeviceReuseBlockNum(local - host - disk);
        resource.setMemoryReuseBlockNum(host);
        resource.setDiskReuseBlockNum(disk);
        resource.setStorageBackendReuseBlockNum(backend);
        publishReuseLengths(total * tokens, host * tokens, disk * tokens, backend * tokens);
    } else if (resource_context_.role_type == RoleType::PREFILL && malloc_status == MallocStatus::NONE) {
        stream_->setHostReuseLength(0);
        stream_->setDiskReuseLength(0);
    } else {
        clearCacheReuseState();
    }
    if (!cache_reuse_metrics_.report_load_metrics) {
        const int64_t load_end_time_us                = currentTimeUs();
        cache_reuse_metrics_.load_success             = load_success;
        cache_reuse_metrics_.report_load_metrics      = true;
        cache_reuse_metrics_.report_load_wait_latency = true;
        cache_reuse_metrics_.load_wait_latency_us = std::max<int64_t>(load_end_time_us - load_wait_begin_time_us_, 0);
        cache_reuse_metrics_.match_to_ready_latency_us = std::max<int64_t>(load_end_time_us - malloc_begin_time_us_, 0);
        cache_reuse_metrics_.report_match_to_ready_latency = true;
    }
    allocator_load_context_.reset();

    if (load_success) {
        return absl::OkStatus();
    }
    if (malloc_status == MallocStatus::RETRYABLE_RESOURCE_EXHAUSTED) {
        return absl::UnavailableError("allocator load materialization is temporarily out of KV blocks");
    }
    if (malloc_status == MallocStatus::PERMANENT_RESOURCE_EXHAUSTED) {
        return absl::ResourceExhaustedError("allocator load materialization exceeds KV cache capacity");
    }
    if (malloc_status == MallocStatus::INTERNAL_ERROR) {
        return absl::InternalError("allocator load materialization failed");
    }
    if (resource_context_.role_type == RoleType::PREFILL) {
        return absl::OkStatus();
    }
    const std::string error_text = error.ToString();
    return absl::InternalError(error_text.empty() ? "allocator load failed" : "allocator load failed: " + error_text);
}

void StreamCacheResource::reportCacheReuseMetrics() {
    if (resource_context_.cache_manager == nullptr || stream_->metrics_reporter_ == nullptr || !reuseCache()) {
        return;
    }
    const int64_t input_length                 = stream_->inputLength();
    const int64_t total_reuse_length           = stream_->initialReuseLength();
    cache_reuse_metrics_.kv_cache_reuse_length = total_reuse_length;
    cache_reuse_metrics_.device_reuse_length   = stream_->deviceReuseLength();
    cache_reuse_metrics_.host_reuse_length     = stream_->hostReuseLength();
    cache_reuse_metrics_.disk_reuse_length     = stream_->diskReuseLength();
    cache_reuse_metrics_.remote_reuse_length   = stream_->remoteReuseLength();
    resource_context_.cache_manager->recordCacheHitTokens(input_length, cache_reuse_metrics_);
    cache_reuse_metrics_.report_reuse_metrics = true;
    kmonitor::MetricsTags tags;
    stream_->metrics_reporter_->report<RtpLLMCacheReuseMetrics, RtpLLMCacheReuseMetricsCollector>(
        &tags, &cache_reuse_metrics_);
}

std::optional<absl::Status> StreamCacheResource::pollAllocatorLoad() {
    if (!allocator_load_context_) {
        return absl::OkStatus();
    }
    if (!allocator_load_context_->done()) {
        return std::nullopt;
    }
    return finalizeAllocatorLoad();
}

absl::Status StreamCacheResource::waitForAllocatorLoad() {
    if (!allocator_load_context_) {
        return absl::OkStatus();
    }
    allocator_load_context_->waitDone();
    if (!allocator_load_context_->done()) {
        return absl::InternalError("allocator load context is non-terminal after waitDone");
    }
    return finalizeAllocatorLoad();
}

absl::Status StreamCacheResource::incrKVBlock(int seq_len_override, int prefill_chunk_start) {
    RTP_LLM_PROFILE_FUNCTION();
    // TODO(xinfei.sxf) add reserver_blocks
    if (fake_inited_) {
        return absl::InternalError("fake inited not allow to incr block");
    }

    MallocInfo malloc_info;
    malloc_info.batch_kv_cache_resource      = batch_kv_cache_resource_;
    malloc_info.complete_token_ids           = stream_->completeTokenIdsPtr();
    malloc_info.request_id                   = stream_->streamId();
    malloc_info.verbose                      = malloc_failed_times_ >= 10 ? malloc_failed_times_ % 100 == 0 : true;
    malloc_info.reuse_cache                  = reuseCache();
    malloc_info.enable_cache_lookup          = enableCacheLookup();
    malloc_info.enable_remove_skipped_blocks = true;
    malloc_info.incr_seq_len_override        = seq_len_override;
    malloc_info.prefill_chunk_start           = prefill_chunk_start;

    auto result = resource_context_.cache_manager->malloc(malloc_info);
    if (!result.success) {
        malloc_failed_times_++;
        return absl::InternalError("malloc failed");
    }

    if (result.reuse_len > 0) {
        publishReuseLengths(result.reuse_len, 0, 0, 0);
    }
    if (result.async_context) {
        const bool aborted = resource_context_.cache_manager->abortPendingLoad(result.async_context);
        if (!aborted) {
            RTP_LLM_LOG_DEBUG("incremental allocator load was already committed before rejection");
        }
        return absl::FailedPreconditionError("async incremental KV block allocation is unsupported");
    }

    return absl::OkStatus();
}

bool StreamCacheResource::asyncLoadCache() {
    RTP_LLM_PROFILE_FUNCTION();
    return allocator_load_context_ != nullptr;
}

bool StreamCacheResource::loadCacheDone() {
    if (allocator_load_context_) {
        if (!allocator_load_context_->done()) {
            return false;
        }
        const ErrorInfo error   = allocator_load_context_->errorInfo();
        const bool      success = allocator_load_context_->success();
        const auto      status  = finalizeAllocatorLoad();
        if (!success && !absl::IsUnavailable(status)) {
            RTP_LLM_LOG_WARNING(
                "block tree load failed, stream=%ld error=%s", stream_->streamId(), error.ToString().c_str());
        }
        if (absl::IsUnavailable(status)) {
            ++malloc_failed_times_;
            stream_->generate_status_->clearLoadInitiated();
            reportMallocRetry();
        } else if (!status.ok()) {
            stream_->reportEventWithoutLock(
                StreamEvents::Error, ErrorCode::MALLOC_FAILED, std::string(status.message()));
        }
    }
    return true;
}

// TODO, delete it soon
int StreamCacheResource::curBlocksNum() const {
    return batch_kv_cache_resource_->curBlocksNum();
}

bool StreamCacheResource::isContextStream() const {
    RTP_LLM_CHECK_WITH_INFO(stream_ != nullptr, "StreamCacheResource::isContextStream called with null stream");
    return stream_->isContextStream();
}

const BatchKVCacheResource& StreamCacheResource::kvCache() const {
    batch_kv_cache_resource_->check();
    return *batch_kv_cache_resource_;
}

BatchKVCacheResource& StreamCacheResource::kvCacheMutable() {
    batch_kv_cache_resource_->check();
    return *batch_kv_cache_resource_;
}

void StreamCacheResource::setKVCache(const BatchKVCacheResource& kv_cache_resource) {
    *batch_kv_cache_resource_ = kv_cache_resource;
}

bool StreamCacheResource::updateKVBlock(const std::vector<int>& block_src_batch, bool copy_last_block) {
    return resource_context_.cache_manager->updateKVBlock(
        batch_kv_cache_resource_, block_src_batch, copy_last_block, block_update_mapping_);
}

bool StreamCacheResource::hasCacheKeys() const {
    return batch_kv_cache_resource_->hasCacheKeys();
}

const CacheKeysType& StreamCacheResource::cacheKeys(int32_t batch_id) const {
    return batch_kv_cache_resource_->cacheKeys(batch_id);
}

void StreamCacheResource::fakeInitKVBlock(size_t reserved_blocks) {
    fake_inited_ = true;
    batch_kv_cache_resource_->resetBatchSize(stream_->maxBatchSize());
    const auto topology = resource_context_.cache_manager ?
                              resource_context_.cache_manager->cacheConfig().topologyPtr() :
                              warmupCacheTopology();
    batch_kv_cache_resource_->initGroups(topology);

    reserved_blocks = std::max(1ul, reserved_blocks);
    batch_kv_cache_resource_->resizeBlocks(reserved_blocks, 0);
}

int StreamCacheResource::mallocFailedTimes() const {
    return malloc_failed_times_;
}

void StreamCacheResource::reportMallocRetry() const {
    if (!stream_->metrics_reporter_) {
        return;
    }
    RtpLLMCacheOperationMetricsCollector collector;
    collector.operation_type = RtpLLMCacheOperationMetricsCollector::OpType::MALLOC_RETRY;
    stream_->metrics_reporter_->report<RtpLLMCacheOperationMetrics, RtpLLMCacheOperationMetricsCollector>(nullptr,
                                                                                                          &collector);
}

bool StreamCacheResource::reuseCache() const {
    // AND logic: global REUSE_CACHE=1 AND per-request reuse_cache both must be true.
    // Per-request field flows frontend → FlexLB → engine via protobuf.
    return resource_context_.reuse_cache && (resource_context_.ignore_request_cache_switches || stream_->reuseCache());
}

bool StreamCacheResource::enableMemoryCache() const {
    // Local tiers are deployment policy; request fields remain wire-compatible only.
    return resource_context_.enable_memory_cache;
}

bool StreamCacheResource::enableDeviceCache() const {
    return resource_context_.enable_device_cache;
}

bool StreamCacheResource::enableDiskCache() const {
    return resource_context_.enable_disk_cache;
}

bool StreamCacheResource::enableCacheLookup() const {
    const bool any_global_tier = resource_context_.enable_device_cache || resource_context_.enable_memory_cache
                                 || resource_context_.enable_disk_cache || resource_context_.enable_remote_cache;
    return reuseCache() && any_global_tier;
}

Tier StreamCacheResource::storeTarget() const {
    if (!reuseCache()) {
        return Tier::NONE;
    }
    if (enableDeviceCache()) {
        return Tier::DEVICE;
    }
    if (enableMemoryCache()) {
        return Tier::HOST;
    }
    if (enableDiskCache()) {
        return Tier::DISK;
    }
    return Tier::NONE;
}

void StreamCacheResource::swapLinearBlocks(int32_t batch_id, size_t rhs, size_t lhs) {
    if (rhs == lhs) {
        return;
    }

    auto type_list = resource_context_.cache_manager->cacheConfig().groupTypesSnapshot();

    for (size_t i = 0; i < type_list.size(); i++) {
        if (type_list[i] == CacheGroupType::LINEAR) {
            batch_kv_cache_resource_->swapBlocks(batch_id, i, rhs, lhs);
        }
    }
}

void StreamCacheResource::holdKVCacheForPDSep() {
    auto&       resource   = batch_kv_cache_resource_->cacheResource(0);
    const auto& cache_keys = resource.cacheKeys();
    auto        ref = resource_context_.cache_manager->incrKVCacheRef(resource, cache_keys, /*is_connector=*/true);
    if (ref) {
        pd_kvcache_ref_ = std::move(ref);
    }
}

void StreamCacheResource::releaseKVCacheForPDSep() {
    pd_kvcache_ref_.reset();
}
}  // namespace rtp_llm
