"""Workers consume the locked queues through a real RabbitMQ (Task 7.1).

An in-process worker bound to every queue answers a ping sent to each one, with
results carried back through Redis. Locally the test skips when the broker is
not reachable; CI sets ``RAG_REQUIRE_SERVICES=1`` and starts both services.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from celery.contrib.testing.worker import start_worker

from app.workers.celery_app import celery_app
from app.workers.queues import QueueDomain


def broker_reachable() -> bool:
    try:
        with celery_app.connection_for_write() as connection:
            connection.ensure_connection(max_retries=1, timeout=3)
        celery_app.backend.client.ping()
    except Exception:  # noqa: BLE001 - any failure means "not available"
        return False
    return True


@pytest.fixture(scope="module")
def worker() -> Iterator[None]:
    if not broker_reachable():
        if os.getenv("RAG_REQUIRE_SERVICES") == "1":
            pytest.fail("RabbitMQ and Redis are required (RAG_REQUIRE_SERVICES=1)")
        pytest.skip("RabbitMQ or Redis not reachable; start them with `make up`")
    celery_app.loader.import_default_modules()
    with start_worker(
        celery_app,
        pool="threads",
        concurrency=2,
        queues=[d.value for d in QueueDomain],
        perform_ping_check=False,
        shutdown_timeout=10,
    ):
        yield


@pytest.mark.parametrize("domain", list(QueueDomain), ids=lambda d: d.value)
def test_every_queue_is_consumed(worker: None, domain: QueueDomain) -> None:
    reply = celery_app.send_task("diagnostics.ping", queue=domain.value).get(timeout=15)

    assert reply["queue"] == domain.value
