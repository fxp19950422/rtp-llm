#include <chrono>
#include <cstdlib>
#include <thread>

#include "gtest/gtest.h"
#include "autil/NetUtil.h"
#include "aios/network/arpc/arpc/ANetRPCController.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/proto/cache_store_service.pb.h"

namespace rtp_llm {
namespace {

class DelayedLoadService: public KvCacheStoreService {
public:
    void load(google::protobuf::RpcController*,
              const CacheLoadRequest*,
              CacheLoadResponse* response,
              google::protobuf::Closure* done) override {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        response->set_error_code(EC_SUCCESS);
        done->Run();
    }
};

class TcpServerTest: public ::testing::Test {
protected:
    void SetUp() override {
        const char* old = std::getenv("CACHE_STORE_TCP_MAX_IDLE_TIME_MS");
        had_old_value_ = old != nullptr;
        if (old) {
            old_value_ = old;
        }
    }
    void TearDown() override {
        if (had_old_value_) {
            setenv("CACHE_STORE_TCP_MAX_IDLE_TIME_MS", old_value_.c_str(), 1);
        } else {
            unsetenv("CACHE_STORE_TCP_MAX_IDLE_TIME_MS");
        }
    }
    void checkDelayedReply(const char* max_idle_ms, bool expect_success) {
        if (max_idle_ms) {
            ASSERT_EQ(0, setenv("CACHE_STORE_TCP_MAX_IDLE_TIME_MS", max_idle_ms, 1));
        } else {
            ASSERT_EQ(0, unsetenv("CACHE_STORE_TCP_MAX_IDLE_TIME_MS"));
        }
        DelayedLoadService service;
        TcpServer server;
        ASSERT_TRUE(server.init(1, 1, false));
        ASSERT_TRUE(server.registerService(&service));
        const auto port = autil::NetUtil::randomPort();
        ASSERT_TRUE(server.start(port));
        TcpClient client;
        ASSERT_TRUE(client.init(1));
        auto channel = client.getChannel("127.0.0.1", port);
        ASSERT_NE(nullptr, channel);
        KvCacheStoreService_Stub stub(channel.get(), google::protobuf::Service::STUB_DOESNT_OWN_CHANNEL);
        arpc::ANetRPCController controller;
        controller.SetExpireTime(6000);
        CacheLoadRequest request;
        CacheLoadResponse response;
        stub.load(&controller, &request, &response, nullptr);
        EXPECT_EQ(expect_success, !controller.Failed());
        if (expect_success) {
            EXPECT_TRUE(response.has_error_code());
            EXPECT_EQ(EC_SUCCESS, response.error_code());
        }
    }

private:
    bool had_old_value_ = false;
    std::string old_value_;
};

TEST_F(TcpServerTest, DefaultAllowsDelayedReply) {
    checkDelayedReply(nullptr, true);
}

TEST_F(TcpServerTest, IdleCloseLosesPendingReply) {
    checkDelayedReply("500", false);
}

TEST_F(TcpServerTest, ConfiguredIdleTimePreservesPendingReply) {
    checkDelayedReply("5000", true);
}

TEST_F(TcpServerTest, RejectsNonPositiveIdleTime) {
    for (const char* value : {"0", "-1"}) {
        ASSERT_EQ(0, setenv("CACHE_STORE_TCP_MAX_IDLE_TIME_MS", value, 1));
        TcpServer server;
        ASSERT_TRUE(server.init(1, 1, false));
        EXPECT_FALSE(server.start(autil::NetUtil::randomPort()));
    }
}

}  // namespace
}  // namespace rtp_llm
