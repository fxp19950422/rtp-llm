import asyncio
import unittest
from types import SimpleNamespace

from rtp_llm.utils.concurrency_controller import (
    ConcurrencyController,
    ConcurrencyException,
    init_controller,
)


class ConcurrencyControllerTest(unittest.IsolatedAsyncioTestCase):
    async def test_init_controller_preserves_blocking_configuration(self):
        config = SimpleNamespace(concurrency_limit=3, concurrency_with_block=True)

        controller = init_controller(config, dp_size=2)

        self.assertEqual(controller.max_concurrency, 6)
        self.assertTrue(controller.block)

    async def test_nonblocking_controller_rejects_when_full(self):
        controller = ConcurrencyController(max_concurrency=1)
        self.assertEqual(await controller.increment_async(), 1)

        with self.assertRaises(ConcurrencyException):
            await controller.increment_async()

        controller.decrement()

    async def test_blocking_controller_waits_without_blocking_event_loop(self):
        controller = ConcurrencyController(max_concurrency=1, block=True)
        self.assertEqual(await controller.increment_async(), 1)
        waiter = asyncio.create_task(controller.increment_async())

        await asyncio.sleep(0.02)
        self.assertFalse(waiter.done())
        controller.decrement()

        self.assertEqual(await asyncio.wait_for(waiter, timeout=0.2), 2)
        controller.decrement()


if __name__ == "__main__":
    unittest.main()
