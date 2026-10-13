#include "rtp_llm/cpp/disaggregate/cache_store/TcpServer.h"

#include "aios/network/arpc/arpc/metric/KMonitorANetServerMetricReporter.h"
#include "autil/LockFreeThreadPool.h"
#include "autil/EnvUtil.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TcpServer::~TcpServer() {
    stop();
}

bool TcpServer::init(uint32_t io_thread_count,
                     uint32_t worker_thread_count,
                     bool     enable_metric,
                     uint32_t worker_queue_size) {
    if (rpc_server_transport_ == nullptr) {
        rpc_server_transport_.reset(new anet::Transport(io_thread_count));
        if (!rpc_server_transport_ || !rpc_server_transport_->start()) {
            return false;
        }
        rpc_server_transport_->setName("TcpServer");
    }

    rpc_server_.reset(new arpc::ANetRPCServer(rpc_server_transport_.get(), 3, 100));
    if (enable_metric) {
        arpc::KMonitorANetMetricReporterConfig metricConfig;
        metricConfig.metricLevel                 = kmonitor::FATAL;
        metricConfig.anetConfig.enableANetMetric = true;
        metricConfig.arpcConfig.enableArpcMetric = true;
        auto metricReporter = std::make_shared<arpc::KMonitorANetServerMetricReporter>(metricConfig);
        if (!metricReporter->init(rpc_server_transport_.get())) {
            RTP_LLM_LOG_WARNING("init anet metric reporter failed");
            return false;
        }
        rpc_server_->SetMetricReporter(metricReporter);
    }

    rpc_worker_threadpool_.reset(new autil::LockFreeThreadPool(
        worker_thread_count, worker_queue_size, nullptr, "tcp_server_rpc_threadpool", false));
    if (!rpc_worker_threadpool_->start()) {
        RTP_LLM_LOG_WARNING("tcp server init failed, start rpc worker threadpool failed");
        return false;
    }

    RTP_LLM_LOG_INFO("tcp server init success, io thread count %d, worker thread count %d, worker queue size %d",
                     io_thread_count,
                     worker_thread_count,
                     worker_queue_size);
    return true;
}

bool TcpServer::start(uint32_t listen_port) {
    const int max_idle_time_ms = autil::EnvUtil::getEnv("CACHE_STORE_TCP_MAX_IDLE_TIME_MS", arpc::MAX_IDLE_TIME);
    if (max_idle_time_ms <= 0) {
        RTP_LLM_LOG_WARNING("cache store TCP max idle time must be positive, got %d", max_idle_time_ms);
        return false;
    }
    std::string listen_spec = "tcp:0.0.0.0:" + std::to_string(listen_port);
    // A pending load can wait for Prefill without any TCP traffic. Its control
    // connection must survive that wait, independently of the RDMA data path.
    if (!rpc_server_->Listen(listen_spec, 5000, max_idle_time_ms)) {
        RTP_LLM_LOG_WARNING("tcp server init failed, listen %s failed", listen_spec.c_str());
        return false;
    }
    RTP_LLM_LOG_INFO("tcp server start success, listen %s, max idle time %d ms", listen_spec.c_str(), max_idle_time_ms);
    return true;
}

bool TcpServer::registerService(RPCService* rpc_service) {
    if (rpc_server_ == nullptr) {
        RTP_LLM_LOG_INFO("tcp server not init, register service failed");
        return false;
    }
    return rpc_server_->RegisterService(rpc_service, rpc_worker_threadpool_);
}

void TcpServer::stop() {
    if (rpc_server_transport_) {
        rpc_server_transport_->stop();
        rpc_server_transport_->wait();
    }

    if (rpc_server_) {
        rpc_server_->Close();
        rpc_server_.reset();
    }

    if (rpc_worker_threadpool_) {
        rpc_worker_threadpool_->stop();
        rpc_worker_threadpool_.reset();
    }

    rpc_server_transport_.reset();
}

}  // namespace rtp_llm
