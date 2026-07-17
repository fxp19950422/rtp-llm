import asyncio
import logging
import os
from multiprocessing import Lock, Value
from typing import Optional


class ConcurrencyException(Exception):
    pass


class ConcurrencyController:
    def __init__(self, max_concurrency: int = 1, block: bool = False) -> None:
        self.max_concurrency = max_concurrency
        self.block = block
        self.lock = Lock()
        self.current_concurrency = Value("i", 0)
        self.request_counter = Value("i", 0)

    def get_available_concurrency(self) -> int:
        with self.lock:
            return self.max_concurrency - self.current_concurrency.value

    def _try_increment(self) -> Optional[int]:
        with self.lock:
            if self.current_concurrency.value >= self.max_concurrency:
                return None
            self.current_concurrency.value += 1
            self.request_counter.value += 1
            return self.request_counter.value

    def increment(self) -> int:
        sequence = self._try_increment()
        if sequence is None:
            raise ConcurrencyException(
                f"Concurrency limit {self.max_concurrency} reached"
            )
        return sequence

    async def increment_async(self) -> int:
        while True:
            sequence = self._try_increment()
            if sequence is not None:
                return sequence
            if not self.block:
                raise ConcurrencyException(
                    f"Concurrency limit {self.max_concurrency} reached"
                )
            # Do not block the Uvicorn event loop while an active request releases a slot.
            await asyncio.sleep(0.01)

    def decrement(self) -> None:
        with self.lock:
            self.current_concurrency.value -= 1

    def get_request_counter(self) -> int:
        with self.lock:
            return self.request_counter.value

    def __enter__(self) -> None:
        self.increment()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.decrement()


global_controller: Optional[ConcurrencyController] = None


def init_controller(concurrency_config, dp_size=1):
    """Initialize concurrency controller.

    Args:
        concurrency_config: ConcurrencyConfig object.
        dp_size: Data parallel size. If None,
    """

    concurrency_limit = concurrency_config.concurrency_limit
    global_concurrency_limit = concurrency_limit * dp_size
    logging.info(
        f"concurrency_limit : {concurrency_limit}, global_concurrency_limit : {global_concurrency_limit}, "
        f"concurrency_with_block : {concurrency_config.concurrency_with_block}"
    )
    controller = ConcurrencyController(
        global_concurrency_limit, concurrency_config.concurrency_with_block
    )
    return controller


def set_global_controller(_global_controller: ConcurrencyController):
    global global_controller
    global_controller = _global_controller


def get_global_controller() -> Optional[ConcurrencyController]:
    return global_controller
