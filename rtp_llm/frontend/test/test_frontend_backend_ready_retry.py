import asyncio
import unittest
from types import SimpleNamespace

from rtp_llm.frontend.frontend_app import FrontendApp


class _HangingThenReadyGrpcClient:
    def __init__(self):
        self.calls = 0

    async def post_request(self, method, payload):
        self.calls += 1
        if self.calls == 1:
            await asyncio.Future()
        return {"status": "ok"}


class FrontendBackendReadyRetryTest(unittest.TestCase):
    def test_hung_health_attempt_times_out_and_retries(self):
        app = FrontendApp.__new__(FrontendApp)
        app.frontend_server = SimpleNamespace(is_embedding=False)
        app.server_config = SimpleNamespace(rank_id=0, frontend_server_id=0)
        app.grpc_client = _HangingThenReadyGrpcClient()
        app._BACKEND_HEALTH_READY_TIMEOUT_S = 1.0
        app._BACKEND_HEALTH_REQUEST_TIMEOUT_S = 0.01
        app._BACKEND_HEALTH_RETRY_INTERVAL_S = 0.0

        asyncio.run(app._wait_backend_health_ready_impl())

        self.assertEqual(app.grpc_client.calls, 2)


if __name__ == "__main__":
    unittest.main()
