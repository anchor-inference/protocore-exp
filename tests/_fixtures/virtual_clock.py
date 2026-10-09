"""An event loop whose clock jumps to the next timer instead of waiting for it.

For scenarios measured in minutes of silence: the loop's own timers decide
where each jump stops, so ``asyncio.wait_for``, ``asyncio.sleep`` and every
deadline built on ``loop.time()`` behave exactly as on a real loop, only
without the wait. Nothing that reads the wall clock directly moves with it.
"""
from __future__ import annotations

import asyncio
import selectors
from collections.abc import Awaitable, Callable
from typing import Any


class _JumpingSelector(selectors.DefaultSelector):
    """With nothing ready, move the clock by the timeout instead of blocking for it."""

    def __init__(self, clock: list[float]) -> None:
        super().__init__()
        self._clock = clock

    def select(self, timeout: float | None = None) -> list[Any]:
        ready = super().select(0)
        if not ready and timeout:
            self._clock[0] += timeout
        return ready


class VirtualClockLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        self._virtual = [0.0]
        super().__init__(_JumpingSelector(self._virtual))

    def time(self) -> float:
        return self._virtual[0]


def run_on_virtual_clock[T](factory: Callable[[], Awaitable[T]]) -> tuple[T, float]:
    """Run a coroutine on a fresh virtual-clock loop; return its result and the virtual seconds spent."""
    loop = VirtualClockLoop()
    try:
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(factory())
        return result, loop.time()
    finally:
        asyncio.set_event_loop(None)
        loop.close()
