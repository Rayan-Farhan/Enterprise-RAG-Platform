"""Bind Celery task executions to durable job records (Task 7.2).

    @job_task(name="ingestion.parse_document", queue=QueueDomain.PARSING)
    async def parse_document(ctx: JobContext, *, storage_key: str) -> None:
        ...
        await ctx.progress(0.5, "parsed 120 of 240 pages")

    job = await enqueue(parse_document, task_type=JobType.PARSE_DOCUMENT,
                        document_id=doc_id, storage_key=key)

`enqueue` writes the job row before publishing the message, so a worker can
never pick up a job it cannot find. The Celery task id is the job id, which
makes a broker message, a worker log line and an API response one identifier.

The body is async because the pipeline services it wraps are. Each execution
runs on a fresh event loop with its own NullPool engine: a pooled asyncpg
connection belongs to the loop that opened it, and worker threads each run
their own loop, so a shared pool would hand connections across loops.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from celery import Task
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models.job import Job
from app.jobs.service import JobAlreadyFinished, JobCancelled, JobService
from app.workers.celery_app import celery_app
from app.workers.queues import QueueDomain

logger = get_logger("app.workers.jobs")

JobBody = Callable[..., Awaitable[None]]


class JobContext:
    """What a job body gets: its identity, progress reporting, and a database."""

    def __init__(
        self, job_id: uuid.UUID, jobs: JobService, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        self.job_id = job_id
        self.sessions = sessions
        self._jobs = jobs

    async def progress(self, fraction: float, message: str | None = None) -> None:
        """Report progress. Raises JobCancelled if the job has been asked to stop."""
        await self._jobs.report_progress(self.job_id, fraction, message)

    async def checkpoint(self) -> None:
        """Stop here if the job has been cancelled; a safe point to abandon work."""
        await self._jobs.checkpoint(self.job_id)


def create_worker_engine() -> AsyncEngine:
    return create_async_engine(get_settings().async_database_url, poolclass=NullPool)


# Indirection so tests can point workers at a throwaway database.
worker_engine_factory: Callable[[], AsyncEngine] = create_worker_engine


async def execute_job(body: JobBody, job_id: uuid.UUID, worker: str, kwargs: dict[str, Any]) -> str:
    """Run one delivery of a job and record how it ended. Returns the final status."""
    engine = worker_engine_factory()
    try:
        sessions = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        jobs = JobService(sessions)
        try:
            job = await jobs.start(job_id, worker=worker)
        except JobCancelled:
            logger.info("job cancelled before it started", job_id=str(job_id))
            return "cancelled"
        except JobAlreadyFinished as exc:
            # A redelivery of work that already completed - acknowledge and move on.
            logger.info(
                "job already finished; skipping delivery", job_id=str(job_id), reason=str(exc)
            )
            return "skipped"

        logger.info("job started", job_id=str(job_id), task_type=job.task_type, attempt=job.attempt)
        try:
            await body(JobContext(job_id, jobs, sessions), **kwargs)
        except JobCancelled:
            await jobs.mark_cancelled(job_id)
            logger.info("job cancelled", job_id=str(job_id))
            return "cancelled"
        except Exception as exc:
            await jobs.fail(job_id, f"{type(exc).__name__}: {exc}")
            logger.exception("job failed", job_id=str(job_id))
            raise
        await jobs.succeed(job_id)
        logger.info("job succeeded", job_id=str(job_id))
        return "succeeded"
    finally:
        await engine.dispose()


def job_task(*, name: str, queue: QueueDomain, **options: Any) -> Callable[[JobBody], Task]:
    """Register an async job body as a Celery task on its queue."""

    def decorator(body: JobBody) -> Task:
        def run(self: Task, job_id: str, **kwargs: Any) -> str:
            worker = self.request.hostname or "unknown"
            return asyncio.run(execute_job(body, uuid.UUID(job_id), worker, kwargs))

        run.__name__ = body.__name__
        run.__doc__ = body.__doc__
        return celery_app.task(name=name, queue=queue.value, bind=True, **options)(run)

    return decorator


async def enqueue(
    task: Task,
    *,
    task_type: str,
    document_id: uuid.UUID | None = None,
    version_id: uuid.UUID | None = None,
    jobs: JobService | None = None,
    **kwargs: Any,
) -> Job:
    """Record a job, then publish it to its task's queue."""
    jobs = jobs or JobService()
    queue = task.queue
    job = await jobs.create(
        task_type=task_type, queue=queue, document_id=document_id, version_id=version_id
    )
    try:
        await asyncio.to_thread(
            task.apply_async, kwargs={"job_id": str(job.id), **kwargs}, task_id=str(job.id)
        )
    except Exception as exc:
        # Nothing will ever run this job; say so rather than leave it queued.
        await jobs.fail_unpublished(job.id, f"could not publish to queue {queue!r}: {exc}")
        raise
    return job
