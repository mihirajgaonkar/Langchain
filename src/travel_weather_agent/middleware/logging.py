"""Simple logging helpers for requests and tool calls."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator


logger = logging.getLogger("travel_weather_agent")


@dataclass(frozen=True)
class RequestLogContext:
    """Metadata for one agent request."""

    request_id: str
    thread_id: str
    user_query: str


def new_request_context(thread_id: str, user_query: str) -> RequestLogContext:
    """Create a request log context with a non-secret request id."""
    return RequestLogContext(
        request_id=str(uuid.uuid4()),
        thread_id=thread_id,
        user_query=user_query,
    )


@contextmanager
def log_request(context: RequestLogContext) -> Iterator[None]:
    """Log request start/end, duration, and success/failure."""
    start = time.perf_counter()
    logger.info(
        "request start",
        extra={
            "request_id": context.request_id,
            "thread_id": context.thread_id,
            "user_query": context.user_query,
        },
    )
    try:
        yield
    except Exception:
        duration = time.perf_counter() - start
        logger.exception(
            "request failed",
            extra={
                "request_id": context.request_id,
                "thread_id": context.thread_id,
                "duration": round(duration, 3),
            },
        )
        raise
    else:
        duration = time.perf_counter() - start
        logger.info(
            "request finished",
            extra={
                "request_id": context.request_id,
                "thread_id": context.thread_id,
                "duration": round(duration, 3),
                "success": True,
            },
        )


def sanitize_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Limit logged tool args and avoid secret-looking fields."""
    sanitized: dict[str, Any] = {}
    for key, value in args.items():
        lowered = key.lower()
        if "key" in lowered or "token" in lowered or "secret" in lowered:
            sanitized[key] = "<redacted>"
        else:
            text = str(value)
            sanitized[key] = text if len(text) <= 120 else f"{text[:117]}..."
    return sanitized


def log_tool_result(
    *,
    tool_name: str,
    args: dict[str, Any],
    duration: float,
    success: bool,
) -> None:
    """Log one tool execution without dumping large result payloads."""
    logger.info(
        "tool call",
        extra={
            "tool_name": tool_name,
            "tool_args": sanitize_tool_args(args),
            "duration": round(duration, 3),
            "success": success,
        },
    )

