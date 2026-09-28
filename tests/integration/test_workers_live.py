"""Workers consume the locked queues through a real RabbitMQ (Tasks 7.1, 7.2).

An in-process worker bound to every queue answers a ping sent to each one, with
results carried back through Redis, and runs job-tracked tasks whose state the
test reads while they run. Locally the test skips when the broker is
not reachable; CI sets ``RAG_REQUIRE_SERVICES=1`` and starts both services.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
from celery.contrib.testing.worker import start_worker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models.base import Base
from app.db.models.job import Job, JobStatus, JobType
from app.jobs.service import JobService
from app.workers import job_task
from app.workers.celery_app import celery_app
from app.workers.job_task import enqueue
from app.workers.queues import QueueDomain
from app.workers.tasks.diagnostics import exercise_job


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


@pytest.fixture
async def job_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[JobService]:
    url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    monkeypatch.setattr(job_task, "worker_engine_factory", lambda: create_async_engine(url))
    yield JobService(async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False))
    await engine.dispose()


async def wait_for(jobs: JobService, job_id: uuid.UUID, *statuses: JobStatus) -> Job:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        job = await jobs.get(job_id)
        if job.status in statuses:
            return job
        await asyncio.sleep(0.05)
    raise AssertionError(f"job {job_id} never reached {statuses}; last {job.status}")


async def test_a_job_runs_through_the_broker_with_live_progress(
    worker: None, job_db: JobService
) -> None:
    job = await enqueue(
        exercise_job, task_type=JobType.DIAGNOSTIC, jobs=job_db, steps=4, step_seconds=0.3
    )

    running = await wait_for(job_db, job.id, JobStatus.RUNNING)
    while (current := await job_db.get(job.id)).progress == 0.0:
        await asyncio.sleep(0.05)
    assert current.status == JobStatus.RUNNING
    assert 0.0 < current.progress < 1.0

    done = await wait_for(job_db, job.id, JobStatus.SUCCEEDED)
    assert running.worker and done.attempt == 1
    assert done.progress == 1.0


async def test_a_running_job_stops_after_a_cancel_request(worker: None, job_db: JobService) -> None:
    job = await enqueue(
        exercise_job, task_type=JobType.DIAGNOSTIC, jobs=job_db, steps=50, step_seconds=0.1
    )
    await wait_for(job_db, job.id, JobStatus.RUNNING)

    await job_db.request_cancel(job.id)

    stopped = await wait_for(job_db, job.id, JobStatus.CANCELLED)
    assert stopped.progress < 1.0
