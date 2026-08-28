#include "rtp_llm/cpp/normal_engine/DecodeCycleObserver.h"

#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <utility>
#include <unistd.h>

namespace rtp_llm {
namespace {

std::string observerPrefix(const char* setting) {
    const char* explicit_path = std::getenv("RTP_LLM_DECODE_OBSERVE_PATH");
    if (explicit_path != nullptr && *explicit_path != '\0') {
        return explicit_path;
    }
    if (setting != nullptr && std::string(setting) != "1") {
        return setting;
    }
    return "/tmp/rtp_llm_decode_observe";
}

}  // namespace

DecodeCycleObserver& DecodeCycleObserver::instance() {
    static DecodeCycleObserver observer = []() {
        const char* setting = std::getenv("RTP_LLM_DECODE_OBSERVE");
        const bool enabled = setting != nullptr && *setting != '\0' && std::string(setting) != "0";
        return DecodeCycleObserver(enabled, enabled ? observerPrefix(setting) : std::string());
    }();
    return observer;
}

DecodeCycleObserver::DecodeCycleObserver(bool enabled, std::string output_path, size_t max_records):
    enabled_(enabled), max_records_(max_records), output_prefix_(std::move(output_path)) {}

DecodeCycleObserver::DecodeCycleObserver(bool enabled, std::string output_path, RankInfo ranks, size_t max_records):
    enabled_(enabled), max_records_(max_records), ranks_(ranks) {
    if (!enabled_) {
        return;
    }
    openLocked(output_path);
}

void DecodeCycleObserver::openLocked(const std::string& path) {
    output_.rdbuf()->pubsetbuf(output_buffer_.data(), output_buffer_.size());
    output_.open(path, std::ios::out | std::ios::app);
    enabled_ = output_.is_open();
}

std::string DecodeCycleObserver::rankedPath(const std::string& prefix, int64_t world_rank) {
    const std::string effective_prefix = prefix.empty() ? "/tmp/rtp_llm_decode_observe" : prefix;
    return effective_prefix + ".pid" + std::to_string(::getpid()) + ".rank" + std::to_string(world_rank) + ".jsonl";
}

DecodeCycleObserver::~DecodeCycleObserver() {
    if (output_.is_open()) {
        output_.flush();
    }
}

int64_t DecodeCycleObserver::monotonicNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

bool DecodeCycleObserver::shouldSample(uint64_t cycle_seq) const {
    return cycle_seq <= 64 || cycle_seq % 8 == 0;
}

uint64_t DecodeCycleObserver::currentCycleSeq() const {
    if (!enabled_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    return cycle_open_ ? cycle_seq_ + 1 : cycle_seq_;
}

void DecodeCycleObserver::configureRanks(RankInfo ranks) {
    if (!enabled_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    ranks_ = ranks;
    if (!output_.is_open()) {
        openLocked(rankedPath(output_prefix_, ranks.world_rank));
    }
}

void DecodeCycleObserver::beginCycle() {
    if (!enabled_ || !output_.is_open()) {
        return;
    }
    const int64_t now = monotonicNs();
    std::lock_guard<std::mutex> lock(mutex_);
    if (cycle_open_) {
        return;
    }
    cycle_open_ = true;
    cycle_start_ns_ = now;
    cycle_period_us_ = previous_cycle_start_ns_ == 0 ? 0 : (now - previous_cycle_start_ns_) / 1000;
    active_ = records_written_ < max_records_ && shouldSample(cycle_seq_ + 1);
    if (!active_) {
        return;
    }
    process_submit_us_      = 0;
    scheduled_real_batch_   = 0;
    effective_batch_        = 0;
    fake_batch_             = 0;
    ep_pad_enabled_         = false;
    ep_pad_reason_          = "disabled";
    graph_calls_.clear();
    acceptance_ = Acceptance{};
}

void DecodeCycleObserver::recordPadding(size_t      scheduled_real_batch,
                                        size_t      effective_batch,
                                        size_t      fake_batch,
                                        bool        ep_pad_enabled,
                                        std::string ep_pad_reason) {
    if (!enabled_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    scheduled_real_batch_ = scheduled_real_batch;
    effective_batch_      = effective_batch;
    fake_batch_           = fake_batch;
    ep_pad_enabled_       = ep_pad_enabled;
    ep_pad_reason_        = std::move(ep_pad_reason);
}

void DecodeCycleObserver::recordProcessSubmit() {
    if (!enabled_) {
        return;
    }
    const int64_t now = monotonicNs();
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_) {
        process_submit_us_ = (now - cycle_start_ns_) / 1000;
    }
}

void DecodeCycleObserver::recordGraphCall(std::string role,
                                          int64_t     actual_batch,
                                          int64_t     graph_key,
                                          int64_t     padding_rows,
                                          bool        replay,
                                          std::string fallback_reason) {
    if (!enabled_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    for (auto& call : graph_calls_) {
        if (call.role == role && call.actual_batch == actual_batch && call.graph_key == graph_key
            && call.padding_rows == padding_rows) {
            if (replay) {
                ++call.replay_count;
            } else {
                ++call.fallback_count;
                if (call.fallback_reason.empty()) {
                    call.fallback_reason = std::move(fallback_reason);
                }
            }
            return;
        }
    }
    GraphCall call;
    call.role            = std::move(role);
    call.actual_batch    = actual_batch;
    call.graph_key       = graph_key;
    call.padding_rows    = padding_rows;
    call.replay_count    = replay ? 1 : 0;
    call.fallback_count  = replay ? 0 : 1;
    call.fallback_reason = std::move(fallback_reason);
    graph_calls_.push_back(std::move(call));
}

void DecodeCycleObserver::recordAcceptance(uint64_t                       source_cycle_seq,
                                           int64_t                        stream_count,
                                           int64_t                        accepted_output_tokens,
                                           const std::array<int64_t, 3>& accepted_draft_per_pos,
                                           int64_t                        proposed_draft_tokens) {
    if (!enabled_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!active_) {
        return;
    }
    acceptance_.source_cycle_seq          = source_cycle_seq;
    acceptance_.stream_count              = stream_count;
    acceptance_.accepted_output_tokens    = accepted_output_tokens;
    acceptance_.accepted_draft_per_pos    = accepted_draft_per_pos;
    acceptance_.proposed_draft_tokens     = proposed_draft_tokens;
}

std::string DecodeCycleObserver::escapeJson(const std::string& value) {
    std::ostringstream escaped;
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': escaped << "\\\""; break;
            case '\\': escaped << "\\\\"; break;
            case '\n': escaped << "\\n"; break;
            case '\r': escaped << "\\r"; break;
            case '\t': escaped << "\\t"; break;
            default:
                if (ch < 0x20) {
                    escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch)
                            << std::dec;
                } else {
                    escaped << ch;
                }
        }
    }
    return escaped.str();
}

void DecodeCycleObserver::writeCycleLocked(int64_t) {
    output_ << "{\"schema_version\":1,\"pid\":" << ::getpid() << ",\"tp\":" << ranks_.tp_rank
            << ",\"ep\":" << ranks_.ep_rank << ",\"dp\":" << ranks_.dp_rank << ",\"world_rank\":"
            << ranks_.world_rank << ",\"cycle_seq\":" << cycle_seq_ << ",\"cycle_start_mono_ns\":"
            << cycle_start_ns_ << ",\"cycle_period_us\":" << cycle_period_us_ << ",\"scheduled_real_batch\":"
            << scheduled_real_batch_ << ",\"effective_batch\":" << effective_batch_ << ",\"fake_batch\":"
            << fake_batch_ << ",\"ep_pad_enabled\":" << (ep_pad_enabled_ ? "true" : "false")
            << ",\"ep_pad_reason\":\"" << escapeJson(ep_pad_reason_) << "\",\"process_submit_us\":"
            << process_submit_us_ << ",\"graph_calls\":[";
    for (size_t i = 0; i < graph_calls_.size(); ++i) {
        const auto& call = graph_calls_[i];
        if (i != 0) output_ << ',';
        output_ << "{\"role\":\"" << escapeJson(call.role) << "\",\"actual_batch\":" << call.actual_batch
                << ",\"graph_key\":" << call.graph_key << ",\"padding_rows\":" << call.padding_rows
                << ",\"replay_count\":" << call.replay_count << ",\"fallback_count\":" << call.fallback_count
                << ",\"fallback_reason\":\"" << escapeJson(call.fallback_reason) << "\"}";
    }
    const double average = acceptance_.stream_count > 0
                               ? static_cast<double>(acceptance_.accepted_output_tokens) / acceptance_.stream_count
                               : 0.0;
    output_ << "],\"accept_record\":{\"source_cycle_seq\":" << acceptance_.source_cycle_seq
            << ",\"stream_count\":" << acceptance_.stream_count << ",\"accepted_output_tokens\":"
            << acceptance_.accepted_output_tokens << ",\"accepted_draft_per_pos\":["
            << acceptance_.accepted_draft_per_pos[0] << ',' << acceptance_.accepted_draft_per_pos[1] << ','
            << acceptance_.accepted_draft_per_pos[2] << "],\"proposed_draft_tokens\":"
            << acceptance_.proposed_draft_tokens << ",\"avg_output_tokens_per_stream\":" << average << "}}\n";
}

void DecodeCycleObserver::finishCycle() {
    if (!enabled_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cycle_open_) {
        return;
    }
    ++cycle_seq_;
    previous_cycle_start_ns_ = cycle_start_ns_;
    if (active_) {
        writeCycleLocked(monotonicNs());
        ++records_written_;
    }
    cycle_open_ = false;
    active_     = false;
}

void DecodeCycleObserver::cancelCycle() {
    if (!enabled_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    cycle_open_ = false;
    active_     = false;
}

void DecodeCycleObserver::flushForTest() {
    std::lock_guard<std::mutex> lock(mutex_);
    output_.flush();
}

}  // namespace rtp_llm
