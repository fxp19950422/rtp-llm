#include "rtp_llm/cpp/engine_base/schedulers/EpWorkSignal.h"

#include <cerrno>
#include <chrono>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <thread>

#include <linux/futex.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rtp_llm {
namespace {

constexpr uint64_t kMagic   = 0x5254504550574B32ULL;  // "RTPEPWK2"
constexpr uint32_t kVersion = 2;

[[noreturn]] void throwSystemError(const std::string& operation) {
    throw std::runtime_error(operation + ": " + std::strerror(errno));
}

int futexWait(uint32_t* address, uint32_t expected) {
    return static_cast<int>(::syscall(SYS_futex, address, FUTEX_WAIT, expected, nullptr, nullptr, 0));
}

void futexWakeAll(uint32_t* address) {
    (void)::syscall(SYS_futex, address, FUTEX_WAKE, INT_MAX, nullptr, nullptr, 0);
}

}  // namespace

std::string EpWorkSignal::makeShmName(const std::string& signal_id) {
    if (signal_id.empty() || signal_id.size() > 64) {
        throw std::invalid_argument("dp_adaptive_fake_wakeup_id must contain 1..64 characters");
    }
    std::string sanitized;
    sanitized.reserve(signal_id.size());
    for (const unsigned char ch : signal_id) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-'
            || ch == '_') {
            sanitized.push_back(static_cast<char>(ch));
        } else {
            throw std::invalid_argument("dp_adaptive_fake_wakeup_id accepts only [A-Za-z0-9_-]");
        }
    }
    return "/rtp_llm_ep_work_" + std::to_string(::getuid()) + "_" + sanitized;
}

uint64_t EpWorkSignal::loadEpoch(const uint64_t* value) {
    return __atomic_load_n(value, __ATOMIC_ACQUIRE);
}

EpWorkSignal::EpWorkSignal(std::string signal_id, size_t dp_rank, size_t dp_size):
    shm_name_(makeShmName(signal_id)), dp_rank_(dp_rank), dp_size_(dp_size) {
    if (dp_size_ < 2 || dp_size_ > kMaxDpRanks || dp_rank_ >= dp_size_) {
        throw std::invalid_argument("invalid DP geometry for adaptive fake wakeup");
    }

    try {
        if (dp_rank_ == 0) {
            fd_ = ::shm_open(shm_name_.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (fd_ < 0) {
                if (errno == EEXIST) {
                    throw std::runtime_error("adaptive fake wakeup id is already in use: " + shm_name_);
                }
                throwSystemError("shm_open(create " + shm_name_ + ")");
            }
            // O_EXCL succeeded: only this instance may unlink the name.
            // In particular, a duplicate creator rejected above owns nothing.
            owns_shm_name_ = true;
            if (::ftruncate(fd_, sizeof(SharedState)) != 0) {
                throwSystemError("ftruncate(" + shm_name_ + ")");
            }
        } else {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            do {
                fd_ = ::shm_open(shm_name_.c_str(), O_RDWR | O_CLOEXEC, 0600);
                if (fd_ >= 0) {
                    break;
                }
                if (errno != ENOENT) {
                    throwSystemError("shm_open(attach " + shm_name_ + ")");
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } while (std::chrono::steady_clock::now() < deadline);
            if (fd_ < 0) {
                throw std::runtime_error("timed out waiting for adaptive fake wakeup owner: " + shm_name_);
            }
            // The name is visible before the creator's ftruncate completes.
            // mmap of an empty object can succeed, but reading magic then
            // raises SIGBUS. Wait for backing storage before any mapped read.
            struct stat info {};
            for (;;) {
                if (::fstat(fd_, &info) != 0) {
                    throwSystemError("fstat(" + shm_name_ + ")");
                }
                if (info.st_size == static_cast<off_t>(sizeof(SharedState))) {
                    break;
                }
                if (info.st_size != 0) {
                    throw std::runtime_error("adaptive fake wakeup shared-state size mismatch: " + shm_name_);
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    throw std::runtime_error("timed out waiting for adaptive fake wakeup storage: " + shm_name_);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }

        void* mapping = ::mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (mapping == MAP_FAILED) {
            state_ = nullptr;
            throwSystemError("mmap(" + shm_name_ + ")");
        }
        state_ = static_cast<SharedState*>(mapping);

        if (dp_rank_ == 0) {
            std::memset(state_, 0, sizeof(SharedState));
            state_->version = kVersion;
            state_->dp_size = static_cast<uint32_t>(dp_size_);
            __atomic_store_n(&state_->magic, kMagic, __ATOMIC_RELEASE);
            futexWakeAll(&state_->wake_sequence);
        } else {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (__atomic_load_n(&state_->magic, __ATOMIC_ACQUIRE) != kMagic
                   && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (__atomic_load_n(&state_->magic, __ATOMIC_ACQUIRE) != kMagic) {
                throw std::runtime_error("adaptive fake wakeup shared state was not initialized: " + shm_name_);
            }
            if (state_->version != kVersion || state_->dp_size != dp_size_) {
                throw std::runtime_error("adaptive fake wakeup shared-state geometry/version mismatch");
            }
        }
    } catch (...) {
        if (state_ != nullptr) {
            ::munmap(state_, sizeof(SharedState));
            state_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        if (owns_shm_name_) {
            ::shm_unlink(shm_name_.c_str());
        }
        throw;
    }
}

EpWorkSignal::~EpWorkSignal() {
    if (state_ != nullptr) {
        ::munmap(state_, sizeof(SharedState));
    }
    if (fd_ >= 0) {
        ::close(fd_);
    }
    if (owns_shm_name_) {
        ::shm_unlink(shm_name_.c_str());
    }
}

void EpWorkSignal::publishWork() {
    const uint64_t epoch = nextEpoch();
    uint64_t       seen  = loadEpoch(&state_->rank_work_epoch[dp_rank_]);
    while (seen < epoch
           && !__atomic_compare_exchange_n(&state_->rank_work_epoch[dp_rank_],
                                           &seen,
                                           epoch,
                                           false,
                                           __ATOMIC_RELEASE,
                                           __ATOMIC_ACQUIRE)) {
    }
    if (seen >= epoch) {
        return;
    }
    __atomic_add_fetch(&state_->wake_sequence, uint32_t{1}, __ATOMIC_RELEASE);
    futexWakeAll(&state_->wake_sequence);
}

uint64_t EpWorkSignal::nextEpoch() const {
    return completed_epoch_.load(std::memory_order_acquire) + 1;
}

bool EpWorkSignal::hasWorkForNextEpoch() const {
    const uint64_t epoch = nextEpoch();
    for (size_t rank = 0; rank < dp_size_; ++rank) {
        if (loadEpoch(&state_->rank_work_epoch[rank]) >= epoch) {
            return true;
        }
    }
    return false;
}

void EpWorkSignal::waitForWorkForNextEpoch() const {
    if (hasWorkForNextEpoch()) {
        return;
    }
    const uint32_t sequence = __atomic_load_n(&state_->wake_sequence, __ATOMIC_ACQUIRE);
    if (hasWorkForNextEpoch()) {
        return;
    }
    if (futexWait(&state_->wake_sequence, sequence) == 0 || errno == EAGAIN || errno == EINTR) {
        // Return after any wake so the scheduler can also re-check its local
        // stop/enqueue predicate under the scheduler mutex.
        return;
    }
    throwSystemError("futex wait for adaptive fake wakeup");
}

void EpWorkSignal::completeStep() {
    completed_epoch_.fetch_add(1, std::memory_order_release);
}

void EpWorkSignal::wakeAll() {
    __atomic_add_fetch(&state_->wake_sequence, uint32_t{1}, __ATOMIC_RELEASE);
    futexWakeAll(&state_->wake_sequence);
}

}  // namespace rtp_llm
