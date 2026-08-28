#include "rtp_llm/cpp/normal_engine/DecodeCycleObserver.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using rtp_llm::DecodeCycleObserver;
int failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::string tempPath(const char* suffix) {
    return "/tmp/decode_cycle_observer_test." + std::to_string(::getpid()) + "." + suffix + ".jsonl";
}

std::string readAll(const std::string& path) {
    std::ifstream input(path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return contents.str();
}

size_t lineCount(const std::string& text) {
    size_t count = 0;
    for (char ch : text) {
        count += ch == '\n';
    }
    return count;
}

int64_t jsonIntField(const std::string& text, const std::string& field, size_t occurrence = 0) {
    const std::string marker = "\"" + field + "\":";
    size_t            pos    = 0;
    for (size_t i = 0; i <= occurrence; ++i) {
        pos = text.find(marker, pos);
        if (pos == std::string::npos) {
            return -1;
        }
        pos += marker.size();
    }
    return std::strtoll(text.c_str() + pos, nullptr, 10);
}

void runCycles(DecodeCycleObserver& observer, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        observer.beginCycle();
        observer.finishCycle();
    }
}

void testDefaultDisabled() {
    ::unsetenv("RTP_LLM_DECODE_OBSERVE");
    expect(!DecodeCycleObserver::instance().enabled(), "RTP_LLM_DECODE_OBSERVE must default to disabled");
    const auto path = tempPath("disabled");
    std::remove(path.c_str());
    DecodeCycleObserver observer(false, path);
    observer.beginCycle();
    observer.finishCycle();
    observer.flushForTest();
    expect(::access(path.c_str(), F_OK) != 0, "disabled observer must not create an output file");
}

void testSampling() {
    const auto path = tempPath("sampling");
    std::remove(path.c_str());
    DecodeCycleObserver observer(true, path, DecodeCycleObserver::RankInfo{});
    runCycles(observer, 80);
    observer.flushForTest();
    const auto output = readAll(path);
    expect(lineCount(output) == 66, "cycles 1..64 plus 72 and 80 must be sampled");
    expect(output.find("\"cycle_seq\":64") != std::string::npos, "cycle 64 must be present");
    expect(output.find("\"cycle_seq\":65") == std::string::npos, "cycle 65 must not be present");
    expect(output.find("\"cycle_seq\":72") != std::string::npos, "cycle 72 must be present");
    expect(output.find("\"cycle_seq\":80") != std::string::npos, "cycle 80 must be present");
    std::remove(path.c_str());
}

void testCancelledIdleDoesNotConsumeCycle() {
    const auto path = tempPath("cancel-idle");
    std::remove(path.c_str());
    DecodeCycleObserver observer(true, path, DecodeCycleObserver::RankInfo{});
    for (size_t i = 0; i < 10000; ++i) {
        observer.beginCycle();
        expect(observer.currentCycleSeq() == 1, "cancelled idle cycle must expose but not consume candidate seq 1");
        observer.cancelCycle();
    }
    observer.beginCycle();
    expect(observer.currentCycleSeq() == 1, "first work after idle must still use cycle seq 1");
    observer.finishCycle();
    observer.flushForTest();
    const auto output = readAll(path);
    expect(lineCount(output) == 1, "cancelled idle cycles must not emit records");
    expect(jsonIntField(output, "cycle_seq") == 1, "first work after idle must serialize cycle seq 1");
    expect(jsonIntField(output, "cycle_period_us") == 0, "first valid work cycle period must be zero");
    std::remove(path.c_str());
}

void testCancelledCycleDoesNotResetPeriod() {
    const auto path = tempPath("cancel-period");
    std::remove(path.c_str());
    DecodeCycleObserver observer(true, path, DecodeCycleObserver::RankInfo{});
    observer.beginCycle();
    observer.finishCycle();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    observer.beginCycle();
    observer.cancelCycle();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    observer.beginCycle();
    observer.finishCycle();
    observer.flushForTest();
    const auto output = readAll(path);
    expect(jsonIntField(output, "cycle_period_us", 0) == 0, "first valid cycle period must be zero");
    expect(jsonIntField(output, "cycle_period_us", 1) >= 8000,
           "cancelled cycle must not replace the previous valid cycle start");
    std::remove(path.c_str());
}

void testUnsampledWorkAdvancesPeriod() {
    const auto path = tempPath("unsampled-period");
    std::remove(path.c_str());
    DecodeCycleObserver observer(true, path, DecodeCycleObserver::RankInfo{});
    runCycles(observer, 70);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    observer.beginCycle();  // valid but unsampled cycle 71
    expect(observer.currentCycleSeq() == 71, "unsampled valid work must expose candidate seq 71");
    observer.finishCycle();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    observer.beginCycle();  // sampled cycle 72
    observer.finishCycle();
    observer.flushForTest();
    const auto output = readAll(path);
    expect(output.find("\"cycle_seq\":71") == std::string::npos, "valid cycle 71 must remain unsampled");
    expect(output.find("\"cycle_seq\":72") != std::string::npos, "valid cycle 72 must be sampled");
    expect(jsonIntField(output, "cycle_period_us", 64) < 10000,
           "sampled cycle 72 period must use unsampled valid cycle 71 as its previous start");
    std::remove(path.c_str());
}

void testDeferredOpenAndRankPath() {
    const auto prefix = tempPath("deferred-prefix");
    const auto path   = prefix + ".pid" + std::to_string(::getpid()) + ".rank7.jsonl";
    std::remove(prefix.c_str());
    std::remove(path.c_str());
    DecodeCycleObserver observer(true, prefix);
    expect(observer.enabled(), "deferred observer must retain the enabled setting");
    expect(::access(prefix.c_str(), F_OK) != 0, "singleton-style constructor must not open the prefix directly");
    expect(::access(path.c_str(), F_OK) != 0, "ranked output must not open before configureRanks");
    observer.configureRanks({1, 2, 3, 7});
    expect(::access(path.c_str(), F_OK) == 0, "configureRanks must open a pid-and-rank-specific JSONL path");
    observer.beginCycle();
    observer.finishCycle();
    observer.flushForTest();
    expect(readAll(path).find("\"tp\":1,\"ep\":2,\"dp\":3,\"world_rank\":7") != std::string::npos,
           "deferred output must use configured direct ranks");
    std::remove(path.c_str());
}

void testDefaultPrefixRankPath() {
    const auto path = "/tmp/rtp_llm_decode_observe.pid" + std::to_string(::getpid()) + ".rank9.jsonl";
    std::remove(path.c_str());
    DecodeCycleObserver observer(true, std::string());
    observer.configureRanks({4, 5, 6, 9});
    observer.beginCycle();
    observer.finishCycle();
    observer.flushForTest();
    expect(::access(path.c_str(), F_OK) == 0, "empty/default prefix must still create a pid-and-rank path");
    std::remove(path.c_str());
}

void testMax2048() {
    const auto path = tempPath("max");
    std::remove(path.c_str());
    DecodeCycleObserver observer(true, path, DecodeCycleObserver::RankInfo{});
    runCycles(observer, 20000);
    observer.flushForTest();
    expect(lineCount(readAll(path)) == 2048, "observer must cap output at 2048 records per rank");
    std::remove(path.c_str());
}

void testRecordAndBufferedOutput() {
    const auto path = tempPath("record");
    std::remove(path.c_str());
    DecodeCycleObserver observer(true, path, {1, 2, 3, 7});
    observer.beginCycle();
    observer.recordPadding(5, 8, 3, true, std::string("pad\"\\\n") + char{1});
    observer.recordProcessSubmit();
    observer.recordGraphCall("mtp_target", 5, 8, 3, true);
    observer.recordGraphCall("mtp_target", 5, 8, 3, true);
    observer.recordGraphCall("mtp_target", 5, 8, 3, false, "shape\"mismatch");
    observer.recordAcceptance(41, 2, 5, {{2, 1, 0}}, 6);
    observer.finishCycle();

    struct stat stat_buffer {};
    expect(::stat(path.c_str(), &stat_buffer) == 0, "enabled observer must open output file");
    expect(stat_buffer.st_size == 0, "finishCycle must not flush each JSONL record");

    observer.flushForTest();
    const auto output = readAll(path);
    expect(lineCount(output) == 1, "one sampled cycle must produce one JSONL record");
    expect(output.find("\"tp\":1,\"ep\":2,\"dp\":3,\"world_rank\":7") != std::string::npos,
           "rank fields must be serialized");
    expect(output.find("\"scheduled_real_batch\":5,\"effective_batch\":8,\"fake_batch\":3")
               != std::string::npos,
           "padding fields must be serialized");
    expect(output.find("\"replay_count\":2,\"fallback_count\":1") != std::string::npos,
           "replay and fallback calls for the same graph must aggregate");
    expect(output.find("\"fallback_reason\":\"shape\\\"mismatch\"") != std::string::npos,
           "fallback reason must be JSON escaped");
    expect(output.find("\"ep_pad_reason\":\"pad\\\"\\\\\\n\\u0001\"") != std::string::npos,
           "control characters must be JSON escaped");
    expect(output.find("\"source_cycle_seq\":41,\"stream_count\":2,\"accepted_output_tokens\":5")
               != std::string::npos,
           "acceptance source cycle and totals must be serialized");
    expect(output.find("\"accepted_draft_per_pos\":[2,1,0],\"proposed_draft_tokens\":6")
               != std::string::npos,
           "acceptance per-position and proposed token fields must be serialized");
    expect(output.find("\"avg_output_tokens_per_stream\":2.5") != std::string::npos,
           "acceptance average must be derived from host counters");
    std::remove(path.c_str());
}

}  // namespace

int main() {
    testDefaultDisabled();
    testSampling();
    testCancelledIdleDoesNotConsumeCycle();
    testCancelledCycleDoesNotResetPeriod();
    testUnsampledWorkAdvancesPeriod();
    testDeferredOpenAndRankPath();
    testDefaultPrefixRankPath();
    testMax2048();
    testRecordAndBufferedOutput();
    if (failures != 0) {
        std::cerr << failures << " observer test assertion(s) failed\n";
        return 1;
    }
    std::cout << "DecodeCycleObserver tests passed\n";
    return 0;
}
