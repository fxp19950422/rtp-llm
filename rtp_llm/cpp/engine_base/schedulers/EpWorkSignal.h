#pragma once

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <string>

namespace rtp_llm {

// Same-host, process-shared work notification for DP ranks participating in an
// EP collective. This is deliberately CPU-only: idle ranks sleep in a futex and
// only enter a fake model step when a peer publishes real scheduler work.
class EpWorkSignal {
public:
    static constexpr size_t kMaxDpRanks = 256;

    EpWorkSignal(std::string signal_id, size_t dp_rank, size_t dp_size);
    ~EpWorkSignal();

    EpWorkSignal(const EpWorkSignal&)            = delete;
    EpWorkSignal& operator=(const EpWorkSignal&) = delete;

    // Publish demand for this rank's next collective epoch. Repeated enqueues
    // before that epoch are intentionally coalesced.
    void publishWork();
    bool hasWorkForNextEpoch() const;
    void waitForWorkForNextEpoch() const;
    void completeStep();
    void wakeAll();

    const std::string& shmNameForTest() const {
        return shm_name_;
    }

private:
    struct alignas(64) SharedState {
        uint64_t magic;
        uint32_t version;
        uint32_t dp_size;
        alignas(4) uint32_t wake_sequence;
        uint32_t reserved[11];
        alignas(64) uint64_t rank_work_epoch[kMaxDpRanks];
    };

    static std::string makeShmName(const std::string& signal_id);
    static uint64_t    loadEpoch(const uint64_t* value);
    uint64_t           nextEpoch() const;

    std::string           shm_name_;
    size_t                dp_rank_ = 0;
    size_t                dp_size_ = 0;
    int                   fd_      = -1;
    bool                  owns_shm_name_ = false;
    SharedState*          state_   = nullptr;
    std::atomic<uint64_t> completed_epoch_{0};
};

}  // namespace rtp_llm
