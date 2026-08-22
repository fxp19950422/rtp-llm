#include <gtest/gtest.h>

#include <functional>
#include <string>

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"

namespace rtp_llm {

TEST(P2PConnectorConfigTest, KeepsLegacyDescriptorTransportDisabledByDefault) {
    RuntimeConfig runtime_config;
    CacheStoreConfig cache_store_config;
    ParallelismConfig parallelism_config;
    PDSepConfig pd_sep_config;

    const auto config = P2PConnectorConfig::create(
        runtime_config, cache_store_config, parallelism_config, pd_sep_config, /*layer_all_num=*/1);

    EXPECT_FALSE(config.worker_config.transfer_backend_config.require_descriptor_handshake);
    EXPECT_FALSE(static_cast<bool>(config.worker_config.transfer_backend_config.descriptor_wire_provider));
}

TEST(P2PConnectorConfigTest, ExplicitDescriptorProviderReachesWorkerBackendConfig) {
    RuntimeConfig runtime_config;
    CacheStoreConfig cache_store_config;
    ParallelismConfig parallelism_config;
    PDSepConfig pd_sep_config;
    std::string observed_wire;

    auto provider = [&observed_wire](std::string& wire) {
        wire = "explicit-descriptor-wire";
        observed_wire = wire;
        return true;
    };
    const auto config = P2PConnectorConfig::create(runtime_config,
                                                   cache_store_config,
                                                   parallelism_config,
                                                   pd_sep_config,
                                                   /*layer_all_num=*/1,
                                                   /*require_descriptor_handshake=*/true,
                                                   provider);

    EXPECT_TRUE(config.worker_config.transfer_backend_config.require_descriptor_handshake);
    ASSERT_TRUE(static_cast<bool>(config.worker_config.transfer_backend_config.descriptor_wire_provider));
    std::string wire;
    EXPECT_TRUE(config.worker_config.transfer_backend_config.descriptor_wire_provider(wire));
    EXPECT_EQ(wire, "explicit-descriptor-wire");
    EXPECT_EQ(observed_wire, wire);
}

TEST(P2PConnectorConfigTest, RejectsEnabledHandshakeWithoutProviderAtTopLevel) {
    RuntimeConfig runtime_config;
    CacheStoreConfig cache_store_config;
    ParallelismConfig parallelism_config;
    PDSepConfig pd_sep_config;

    EXPECT_THROW(P2PConnectorConfig::create(runtime_config,
                                             cache_store_config,
                                             parallelism_config,
                                             pd_sep_config,
                                             /*layer_all_num=*/1,
                                             /*require_descriptor_handshake=*/true),
                 std::invalid_argument);
}

TEST(P2PConnectorConfigTest, RejectsProviderWithoutExplicitHandshakeAtWorkerBoundary) {
    CacheStoreConfig cache_store_config;
    ParallelismConfig parallelism_config;
    PDSepConfig pd_sep_config;
    std::function<bool(std::string&)> provider = [](std::string&) { return true; };

    EXPECT_THROW(P2PConnectorWorkerConfig::create(cache_store_config,
                                                  pd_sep_config,
                                                  parallelism_config,
                                                  /*layer_all_num=*/1,
                                                  /*require_descriptor_handshake=*/false,
                                                  provider),
                 std::invalid_argument);
}

}  // namespace rtp_llm
