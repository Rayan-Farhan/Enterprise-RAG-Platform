"""The Celery application (Task 7.1, ADR-017, ADR-018).

RabbitMQ carries the work. Job state does not live in Celery: Task 7.2's `jobs`
table is the durable record, so task results are ignored by default. The Redis
result backend exists only for the tasks that opt in, such as the diagnostics
ping that proves a queue is being consumed.
"""

from __future__ import annotations

from typing import Any

from celery import Celery
from celery.signals import setup_logging as celery_setup_logging

from app.core.config import AppSettings, get_settings
from app.core.logging import setup_logging
from app.workers.queues import MAX_PRIORITY, QueueDomain, build_queues

TASK_MODULES = ("app.workers.tasks.diagnostics", "app.workers.tasks.ingestion")


def create_celery_app(settings: AppSettings | None = None) -> Celery:
    settings = settings or get_settings()
    app = Celery(
        "enterprise_rag",
        broker=settings.rabbitmq_url,
        backend=settings.redis_url,
        include=list(TASK_MODULES),
    )
    app.conf.update(
        task_queues=build_queues(),
        # Every real task names its queue (tests enforce it); this only catches
        # an ad-hoc send, and a queue no pool reads would hide it.
        task_default_queue=QueueDomain.INGESTION.value,
        task_default_exchange=QueueDomain.INGESTION.value,
        task_default_routing_key=QueueDomain.INGESTION.value,
        task_create_missing_queues=False,
        task_queue_max_priority=MAX_PRIORITY,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        # A message is acknowledged after the task finishes, not when it is
        # received, and a worker that dies mid-task hands the message back. A
        # crash therefore re-runs the task rather than losing it, which is why
        # every Stage 7 task must be idempotent (Task 7.4).
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # Ingestion tasks run for seconds to minutes. Prefetching more than one
        # would park work on a busy worker while an idle one waits.
        worker_prefetch_multiplier=1,
        task_ignore_result=True,
        result_expires=3600,
        broker_connection_retry_on_startup=True,
        worker_hijack_root_logger=False,
        timezone="UTC",
        enable_utc=True,
    )
    return app


@celery_setup_logging.connect
def _use_application_logging(**_: Any) -> None:
    """Route worker logs through the same structured JSON logging as the API."""
    settings = get_settings()
    setup_logging(debug=settings.DEBUG, app_env=settings.APP_ENV)


celery_app = create_celery_app()
