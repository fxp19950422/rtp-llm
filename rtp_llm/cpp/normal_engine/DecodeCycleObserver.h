#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

namespace rtp_llm {

// Process-wide, host-only decode-cycle trace. The production singleton is a
// no-op unless RTP_LLM_DECODE_OBSERVE is set to a non-zero value.
class DecodeCycleObserver {
public:
    struct RankInfo {
        int64_t tp_rank{0};
        int64_t ep_rank{0};
        int64_t dp_rank{0};
        int64_t world_rank{0};
    };

    static DecodeCycleObserver& instance();

    // Public injection constructor keeps the JSON writer independently testable
    // without initializing Torch, Python, or a device.
    DecodeCycleObserver(bool enabled, std::string output_path, size_t max_records = 2048);
    DecodeCycleObserver(bool enabled, std::string output_path, RankInfo ranks, size_t max_records = 2048);
    ~DecodeCycleObserver();

    DecodeCycleObserver(const DecodeCycleObserver&)            = delete;
    DecodeCycleObserver& operator=(const DecodeCycleObserver&) = delete;

    bool     enabled() const noexcept { return enabled_; }
    uint64_t currentCycleSeq() const;
    void     configureRanks(RankInfo ranks);
    void     beginCycle();
    void     recordPadding(size_t      scheduled_real_batch,
                           size_t      effective_batch,
                           size_t      fake_batch,
                           bool        ep_pad_enabled,
                           std::string ep_pad_reason);
    void     recordProcessSubmit();
    void     recordGraphCall(std::string role,
                             int64_t     actual_batch,
                             int64_t     graph_key,
                             int64_t     padding_rows,
                             bool        replay,
                             std::string fallback_reason = {});
    void     recordAcceptance(uint64_t                       source_cycle_seq,
                              int64_t                        stream_count,
                              int64_t                        accepted_output_tokens,
                              const std::array<int64_t, 3>& accepted_draft_per_pos,
                              int64_t                        proposed_draft_tokens);
    void     finishCycle();
    void     flushForTest();

private:
    struct GraphCall {
        std::string role;
        int64_t     actual_batch{0};
        int64_t     graph_key{0};
        int64_t     padding_rows{0};
        uint64_t    replay_count{0};
        uint64_t    fallback_count{0};
        std::string fallback_reason;
    };
    struct Acceptance {
        uint64_t               source_cycle_seq{0};
        int64_t                stream_count{0};
        int64_t                accepted_output_tokens{0};
        std::array<int64_t, 3> accepted_draft_per_pos{{0, 0, 0}};
        int64_t                proposed_draft_tokens{0};
    };

    static int64_t monotonicNs();
    static std::string escapeJson(const std::string& value);
    static std::string rankedPath(const std::string& prefix, int64_t world_rank);
    bool               shouldSample(uint64_t cycle_seq) const;
    void               openLocked(const std::string& path);
    void               writeCycleLocked(int64_t end_ns);

    bool                    enabled_{false};
    size_t                  max_records_{2048};
    mutable std::mutex      mutex_;
    RankInfo                ranks_;
    std::string             output_prefix_;
    std::ofstream           output_;
    std::array<char, 65536> output_buffer_{};
    uint64_t                cycle_seq_{0};
    uint64_t                records_written_{0};
    int64_t                 previous_cycle_start_ns_{0};
    int64_t                 cycle_start_ns_{0};
    int64_t                 cycle_period_us_{0};
    int64_t                 process_submit_us_{0};
    size_t                  scheduled_real_batch_{0};
    size_t                  effective_batch_{0};
    size_t                  fake_batch_{0};
    bool                    ep_pad_enabled_{false};
    std::string             ep_pad_reason_{"disabled"};
    std::vector<GraphCall>  graph_calls_;
    Acceptance              acceptance_;
    bool                    active_{false};
};

}  // namespace rtp_llm
