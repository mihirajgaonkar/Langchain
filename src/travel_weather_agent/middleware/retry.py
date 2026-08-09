"""Small bounded retry helper for transient operations."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar


T = TypeVar("T")


def retry_call(
    func: Callable[[], T],
    *,
    attempts: int = 2,
    delay_seconds: float = 0.2,
    retry_exceptions: tuple[type[BaseException], ...] = (RuntimeError,),
) -> T:
    """Run a callable with a small bounded retry loop."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except retry_exceptions as exc:
            last_error = exc
            if attempt == attempts:
                break
            time.sleep(delay_seconds)

    assert last_error is not None
    raise last_error

