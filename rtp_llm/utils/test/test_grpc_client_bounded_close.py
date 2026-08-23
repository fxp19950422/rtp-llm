import asyncio
import unittest

from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper


class _HangingChannel:
    def __init__(self):
        self.close_started = False

    async def close(self):
        self.close_started = True
        await asyncio.Future()


class GrpcClientBoundedCloseTest(unittest.TestCase):
    def test_model_rpc_channels_force_http_proxy_off(self):
        wrapper = GrpcClientWrapper(
            12345,
            client_config={"grpc.keepalive_time_ms": 1000, "grpc.enable_http_proxy": 1},
        )
        self.assertEqual(
            wrapper._channel_options(),
            [("grpc.keepalive_time_ms", 1000), ("grpc.enable_http_proxy", 0)],
        )

    def test_hanging_channel_close_is_bounded_and_detached(self):
        wrapper = GrpcClientWrapper(12345)
        wrapper._CHANNEL_CLOSE_TIMEOUT_S = 0.01
        channel = _HangingChannel()
        wrapper.channel = channel
        wrapper.stub = object()

        asyncio.run(wrapper.close())

        self.assertTrue(channel.close_started)
        self.assertIsNone(wrapper.channel)
        self.assertIsNone(wrapper.stub)


if __name__ == "__main__":
    unittest.main()
