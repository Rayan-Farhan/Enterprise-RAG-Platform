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

A task registered with ``then`` is a link in a chain (Task 7.3): when its job
succeeds, the next task is enqueued with the same arguments. The next job's id
is derived from this one's, so a redelivered step that already succeeded - or
two deliveries racing - enqueue the follow-up at most once.

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
from app.db.models.job import Job, JobStatus
from app.jobs.service import JobAlreadyExists, JobAlreadyFinished, JobCancelled, JobService
from app.workers.celery_app import celery_app
from app.workers.queues import QueueDomain

logger = get_logger("app.workers.jobs")

JobBody = Callable[..., Awaitable[None]]

# Celery task name -> the job type its rows carry, and the task that follows it.
_TASK_TYPES: dict[str, str] = {}
_FOLLOW_UPS: dict[str, str] = {}


class JobContext:
    """What a job body gets: its identity, progress reporting, and a database."""

    def __init__(
        self,
        job_id: uuid.UUID,
        jobs: JobService,
        sessions: async_sessionmaker[AsyncSession],
        document_id: uuid.UUID | None = None,
        version_id: uuid.UUID | None = None,
    ) -> None:
        self.job_id = job_id
        self.document_id = document_id
        self.version_id = version_id
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


async def execute_job(
    body: JobBody,
    job_id: uuid.UUID,
    worker: str,
    kwargs: dict[str, Any],
    then: str | None = None,
) -> str:
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
            # The previous delivery may have died between recording success and
            # enqueueing the next step; enqueueing is idempotent, so do it again.
            if then is not None:
                finished = await jobs.get(job_id)
                if finished.status == JobStatus.SUCCEEDED:
                    await enqueue_follow_up(jobs, finished, then, kwargs)
            return "skipped"

        logger.info("job started", job_id=str(job_id), task_type=job.task_type, attempt=job.attempt)
        try:
            await body(
                JobContext(job_id, jobs, sessions, job.document_id, job.version_id), **kwargs
            )
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
        if then is not None:
            await enqueue_follow_up(jobs, job, then, kwargs)
        return "succeeded"
    finally:
        await engine.dispose()


def follow_up_job_id(job_id: uuid.UUID, next_task: str) -> uuid.UUID:
    return uuid.uuid5(job_id, next_task)


async def enqueue_follow_up(
    jobs: JobService, job: Job, next_task: str, kwargs: dict[str, Any]
) -> Job | None:
    """Enqueue the chain step after ``job``, unless an earlier delivery already did."""
    task = celery_app.tasks[next_task]
    try:
        return await enqueue(
            task,
            document_id=job.document_id,
            version_id=job.version_id,
            job_id=follow_up_job_id(job.id, next_task),
            jobs=jobs,
            **kwargs,
        )
    except JobAlreadyExists:
        logger.info("follow-up already enqueued", job_id=str(job.id), next_task=next_task)
        return None


def job_task(
    *,
    name: str,
    queue: QueueDomain,
    task_type: str | None = None,
    then: str | None = None,
    **options: Any,
) -> Callable[[JobBody], Task]:
    """Register an async job body as a Celery task on its queue.

    ``task_type`` is what the task's job rows record; ``then`` names the task
    enqueued after this one succeeds.
    """

    def decorator(body: JobBody) -> Task:
        def run(self: Task, job_id: str, **kwargs: Any) -> str:
            worker = self.request.hostname or "unknown"
            return asyncio.run(execute_job(body, uuid.UUID(job_id), worker, kwargs, then=then))

        run.__name__ = body.__name__
        run.__doc__ = body.__doc__
        if task_type is not None:
            _TASK_TYPES[name] = task_type
        if then is not None:
            _FOLLOW_UPS[name] = then
        return celery_app.task(name=name, queue=queue.value, bind=True, **options)(run)

    return decorator


def follow_up_of(task_name: str) -> str | None:
    """The task a chain step hands over to, or None at the end of a chain."""
    return _FOLLOW_UPS.get(task_name)


async def enqueue(
    task: Task,
    *,
    task_type: str | None = None,
    document_id: uuid.UUID | None = None,
    version_id: uuid.UUID | None = None,
    job_id: uuid.UUID | None = None,
    jobs: JobService | None = None,
    **kwargs: Any,
) -> Job:
    """Record a job, then publish it to its task's queue."""
    jobs = jobs or JobService()
    task_type = task_type or _TASK_TYPES.get(task.name)
    if task_type is None:
        raise ValueError(f"Task {task.name!r} declares no job type; pass task_type")
    queue = task.queue
    job = await jobs.create(
        task_type=task_type,
        queue=queue,
        document_id=document_id,
        version_id=version_id,
        job_id=job_id,
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
