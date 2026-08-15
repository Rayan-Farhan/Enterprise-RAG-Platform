"""Structured JSON logging configuration using structlog."""

import logging
import sys
from typing import Any

import structlog


def setup_logging(debug: bool = False, app_env: str = "development") -> None:
    """Configure structured JSON logging for the application and standard libraries."""
    log_level = logging.DEBUG if debug else logging.INFO

    # Shared structlog processors
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if app_env == "development" and debug:
        # Development console formatting
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        # Production JSON formatting
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Suppress overly chatty libraries
    for quiet_logger in ("uvicorn.access", "urllib3", "httpcore", "httpx"):
        logging.getLogger(quiet_logger).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> Any:
    """Return a configured structlog logger."""
    return structlog.get_logger(name)
