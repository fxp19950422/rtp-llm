#pragma once

#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace rtp_llm {

// Benchmark-only, one round per engine process. The controller supplies a new,
// empty shared directory for every round, and restarts every Decode rank.
class DecodeBenchmarkGate {
public:
    DecodeBenchmarkGate(std::string directory, int target, int rank, int ranks):
        directory_(std::move(directory)), target_(target), rank_(rank), ranks_(ranks),
        started_(std::chrono::steady_clock::now()) {}

    bool hold(size_t ready) {
        if (target_ <= 0 || released_) {
            return false;
        }
        if (directory_.empty() || rank_ < 0 || rank_ >= ranks_) {
            throw std::runtime_error("invalid Decode benchmark gate configuration");
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - started_ > std::chrono::minutes(20)) {
            throw std::runtime_error("Decode benchmark gate timed out before release");
        }
        if (now - published_ >= std::chrono::milliseconds(20)) {
            publish(directory_ + "/rank" + std::to_string(rank_), std::to_string(ready));
            published_ = now;
        }
        const std::string release = directory_ + "/release";
        if (rank_ == 0 && !std::ifstream(release).good()) {
            size_t total = 0;
            bool all_ranks = true;
            for (int rank = 0; rank < ranks_; ++rank) {
                size_t count = 0;
                std::ifstream input(directory_ + "/rank" + std::to_string(rank));
                if (!(input >> count)) {
                    all_ranks = false;
                    break;
                }
                total += count;
            }
            if (all_ranks && total >= static_cast<size_t>(target_)) {
                publish(release, std::to_string(total));
            }
        }
        size_t total = 0;
        std::ifstream input(release);
        if (input >> total && total >= static_cast<size_t>(target_)) {
            released_ = true;
            publish(directory_ + "/released-rank" + std::to_string(rank_), std::to_string(ready));
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return true;
    }

private:
    static void publish(const std::string& path, const std::string& value) {
        const std::string temporary = path + ".tmp";
        {
            std::ofstream output(temporary);
            output << value << '\n';
            output.close();
            if (!output) {
                throw std::runtime_error("cannot write Decode benchmark gate snapshot: " + path);
            }
        }
        if (std::rename(temporary.c_str(), path.c_str()) != 0) {
            throw std::runtime_error("cannot publish Decode benchmark gate snapshot: " + path);
        }
    }

    std::string directory_;
    int target_, rank_, ranks_;
    bool released_ = false;
    std::chrono::steady_clock::time_point started_, published_{};
};

}  // namespace rtp_llm
