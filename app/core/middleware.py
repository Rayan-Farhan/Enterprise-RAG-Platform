"""Request context and tracking middleware."""

import time
import uuid
from collections.abc import Callable
from typing import Any

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware attaching request_id, trace_id, and calculating request latency."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Any]) -> Response:  # type: ignore[override]
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))

        # Bind to structlog context
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            trace_id=trace_id,
            path=request.url.path,
            method=request.method,
        )

        request.state.request_id = request_id
        request.state.trace_id = trace_id
        request.state.start_time = time.perf_counter()

        response: Response = await call_next(request)

        duration_ms = (time.perf_counter() - request.state.start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Response-Time-MS"] = f"{duration_ms:.2f}"

        return response
