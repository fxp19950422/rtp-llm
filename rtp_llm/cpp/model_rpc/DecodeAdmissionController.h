#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>

namespace rtp_llm {

class DecodeAdmissionController {
public:
    enum class AcquireResult {
        ACQUIRED,
        CANCELLED,
        TIMED_OUT,
        OVERSIZED,
    };

    explicit DecodeAdmissionController(size_t limit = 1, size_t block_limit = std::numeric_limits<size_t>::max()):
        limit_(std::max<size_t>(limit, 1)), block_limit_(block_limit) {}

    void setLimit(size_t limit) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            limit_ = std::max<size_t>(limit, 1);
        }
        condition_.notify_all();
    }

    void setBlockLimit(size_t block_limit) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            block_limit_ = block_limit;
        }
        condition_.notify_all();
    }

    AcquireResult acquire(size_t slots, size_t blocks, const std::function<bool()>& cancelled, int64_t timeout_ms) {
        slots = std::max<size_t>(slots, 1);
        std::unique_lock<std::mutex> lock(mutex_);
        if (slots > limit_ || blocks > block_limit_) {
            return AcquireResult::OVERSIZED;
        }

        const bool has_deadline = timeout_ms >= 0;
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::milliseconds(std::max<int64_t>(0, timeout_ms));
        while (true) {
            if (cancelled && cancelled()) {
                return AcquireResult::CANCELLED;
            }
            if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
                return AcquireResult::TIMED_OUT;
            }
            const bool slots_available  = slots <= limit_ - std::min(active_slots_, limit_);
            const bool blocks_available = blocks <= block_limit_ - std::min(active_blocks_, block_limit_);
            if (slots_available && blocks_available) {
                active_slots_ += slots;
                active_blocks_ += blocks;
                return AcquireResult::ACQUIRED;
            }

            if (has_deadline) {
                condition_.wait_until(lock, std::min(deadline, std::chrono::steady_clock::now() + kCancelPollInterval));
            } else {
                condition_.wait_for(lock, kCancelPollInterval);
            }
        }
    }

    AcquireResult acquire(size_t slots, const std::function<bool()>& cancelled, int64_t timeout_ms) {
        return acquire(slots, /*blocks=*/0, cancelled, timeout_ms);
    }

    size_t activeSlots() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_slots_;
    }

    size_t limit() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return limit_;
    }

    size_t activeBlocks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return active_blocks_;
    }

    size_t blockLimit() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return block_limit_;
    }

private:
    friend class DecodeAdmissionGuard;

    void release(size_t slots, size_t blocks) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_slots_ -= std::min(active_slots_, std::max<size_t>(slots, 1));
            active_blocks_ -= std::min(active_blocks_, blocks);
        }
        condition_.notify_all();
    }

private:
    static constexpr std::chrono::milliseconds kCancelPollInterval{50};

    mutable std::mutex      mutex_;
    std::condition_variable condition_;
    size_t                  limit_         = 1;
    size_t                  active_slots_  = 0;
    size_t                  block_limit_   = std::numeric_limits<size_t>::max();
    size_t                  active_blocks_ = 0;
};

class DecodeAdmissionGuard {
public:
    DecodeAdmissionGuard(DecodeAdmissionController& controller, size_t slots, size_t blocks = 0):
        controller_(&controller), slots_(std::max<size_t>(slots, 1)), blocks_(blocks) {}

    ~DecodeAdmissionGuard() {
        if (controller_ != nullptr) {
            controller_->release(slots_, blocks_);
        }
    }

    DecodeAdmissionGuard(const DecodeAdmissionGuard&) = delete;
    DecodeAdmissionGuard& operator=(const DecodeAdmissionGuard&) = delete;

private:
    DecodeAdmissionController* controller_;
    size_t                     slots_;
    size_t                     blocks_;
};

}  // namespace rtp_llm
