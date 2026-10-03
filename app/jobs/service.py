"""Job lifecycle: creation, status transitions, progress, cancellation (Task 7.2).

    queued ──start──> running ──succeed──> succeeded
      ▲ │                 │ │ └───fail─────> failed (a dead letter, until replayed)
      │ │                 │ └─cancel seen──> cancelled
      │ └────cancel───────┼─────────────────> cancelled
      └──schedule_retry───┘  (a transient failure, Task 7.5)

A running job cannot be stopped from outside without risking half-written
state, so cancelling one only sets `cancel_requested_at`; the task sees it at
its next progress checkpoint and stops there. A queued job is cancelled at
once, and the task that later picks it up finds it cancelled and does nothing.

`start` also accepts a job that is already running. That is a redelivery: with
late acks, a worker lost mid-task hands the message back, and the next worker
starts the same job again with `attempt` incremented.

Every transition is a single conditional UPDATE, so two actors racing - a
cancel against a start, two redeliveries - resolve in the database, not in
whichever process read the row last. Each call commits on its own, which is
what makes progress visible to readers while the job is still running.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ConflictException, NotFoundException
from app.db.models.job import TERMINAL_STATUSES, FailureKind, Job, JobStatus
from app.db.session import get_session_factory

# Progress messages longer than the column are truncated, never rejected: a
# verbose message must not fail the job it describes.
_MESSAGE_LIMIT = 255
_ERROR_LIMIT = 4000


class JobCancelled(Exception):
    """The job was cancelled; the task must stop without doing more work."""


class JobAlreadyExists(Exception):
    """A job with the requested id was already recorded."""


class JobAlreadyActive(Exception):
    """The same step of the same version is already queued or running."""

    def __init__(self, job: Job) -> None:
        super().__init__(f"{job.task_type} for version {job.version_id} is already {job.status}")
        self.job = job


class JobAlreadyFinished(Exception):
    """The job reached a terminal state before this delivery of it started."""


def _now() -> datetime:
    return datetime.now(UTC)


class JobService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None) -> None:
        self._sessions = session_factory or get_session_factory()

    async def create(
        self,
        *,
        task_type: str,
        queue: str,
        document_id: uuid.UUID | None = None,
        version_id: uuid.UUID | None = None,
        job_id: uuid.UUID | None = None,
        task_name: str | None = None,
        payload: dict[str, Any] | None = None,
        replay_of_id: uuid.UUID | None = None,
    ) -> Job:
        """Record a queued job.

        Raises JobAlreadyExists when a caller-chosen ``job_id`` is taken - how a
        chain step is enqueued at most once - and JobAlreadyActive when the same
        step of the same version is already queued or running.
        """
        job = Job(
            id=job_id or uuid.uuid4(),
            document_id=document_id,
            version_id=version_id,
            task_type=task_type,
            queue=queue,
            status=JobStatus.QUEUED.value,
            attempt=0,
            progress=0.0,
            task_name=task_name,
            payload=payload,
            replay_of_id=replay_of_id,
        )
        async with self._sessions() as session:
            session.add(job)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                if job_id is not None and await session.get(Job, job_id) is not None:
                    raise JobAlreadyExists(str(job_id)) from None
                active = await self._active_step(session, version_id, task_type)
                if active is None:
                    raise
                raise JobAlreadyActive(active) from None
        return job

    @staticmethod
    async def _active_step(
        session: AsyncSession, version_id: uuid.UUID | None, task_type: str
    ) -> Job | None:
        if version_id is None:
            return None
        result = await session.execute(
            select(Job).where(
                Job.version_id == version_id,
                Job.task_type == task_type,
                Job.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
            )
        )
        return result.scalars().first()

    async def get(self, job_id: uuid.UUID) -> Job:
        async with self._sessions() as session:
            job = await session.get(Job, job_id)
        if job is None:
            raise NotFoundException(f"Job '{job_id}' was not found")
        return job

    async def list_for_document(self, document_id: uuid.UUID) -> Sequence[Job]:
        async with self._sessions() as session:
            result = await session.execute(
                select(Job).where(Job.document_id == document_id).order_by(Job.created_at, Job.id)
            )
            return result.scalars().all()

    async def list_for_version(self, version_id: uuid.UUID) -> Sequence[Job]:
        async with self._sessions() as session:
            result = await session.execute(
                select(Job).where(Job.version_id == version_id).order_by(Job.created_at, Job.id)
            )
            return result.scalars().all()

    async def start(self, job_id: uuid.UUID, *, worker: str) -> Job:
        """Claim the job for this worker. Raises if it must not run."""
        now = _now()
        claimed = await self._transition(
            job_id,
            allowed_from=(JobStatus.QUEUED, JobStatus.RUNNING),
            values={
                "status": JobStatus.RUNNING.value,
                "attempt": Job.attempt + 1,
                "worker": worker[:255],
                "started_at": now,
                "error": None,
            },
            require_no_cancel_request=True,
        )
        if claimed:
            return await self.get(job_id)

        job = await self.get(job_id)
        if job.status == JobStatus.RUNNING and job.cancel_requested_at is not None:
            # Cancelled while its previous delivery was running; that worker
            # was lost before it could see the request, so this one records it.
            await self.mark_cancelled(job_id)
            raise JobCancelled(str(job_id))
        if job.status == JobStatus.CANCELLED:
            raise JobCancelled(str(job_id))
        raise JobAlreadyFinished(f"job {job_id} is already {job.status}")

    async def report_progress(
        self, job_id: uuid.UUID, fraction: float, message: str | None = None
    ) -> None:
        """Record progress; raises JobCancelled if a cancel has been requested."""
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"progress must be within [0, 1], got {fraction}")
        values: dict[str, Any] = {"progress": fraction}
        if message is not None:
            values["progress_message"] = message[:_MESSAGE_LIMIT]
        updated = await self._transition(
            job_id,
            allowed_from=(JobStatus.RUNNING,),
            values=values,
            require_no_cancel_request=True,
        )
        if not updated:
            await self._raise_for_stopped(job_id)

    async def checkpoint(self, job_id: uuid.UUID) -> None:
        """Raise JobCancelled if a cancel has been requested; otherwise do nothing."""
        job = await self.get(job_id)
        if job.cancel_requested_at is not None or job.status == JobStatus.CANCELLED:
            raise JobCancelled(str(job_id))

    async def succeed(self, job_id: uuid.UUID, message: str | None = None) -> None:
        values: dict[str, Any] = {
            "status": JobStatus.SUCCEEDED.value,
            "progress": 1.0,
            "completed_at": _now(),
        }
        if message is not None:
            values["progress_message"] = message[:_MESSAGE_LIMIT]
        await self._finish(job_id, values)

    async def fail(
        self, job_id: uuid.UUID, error: str, kind: FailureKind = FailureKind.PERMANENT
    ) -> None:
        """End the job as failed; it is now a dead letter until replayed."""
        await self._finish(
            job_id,
            {
                "status": JobStatus.FAILED.value,
                "error": error[:_ERROR_LIMIT],
                "failure_kind": kind.value,
                "completed_at": _now(),
            },
        )

    async def schedule_retry(self, job_id: uuid.UUID, error: str, message: str) -> None:
        """Put a running job back in the queue after a transient failure.

        The row returns to `queued` with the error kept, so the API shows why
        it is waiting; the next start counts another attempt. A cancel request
        that arrived meanwhile wins: the job is cancelled instead.
        """
        values = {
            "status": JobStatus.QUEUED.value,
            "error": error[:_ERROR_LIMIT],
            "progress_message": message[:_MESSAGE_LIMIT],
            "worker": None,
        }
        if not await self._transition(
            job_id,
            allowed_from=(JobStatus.RUNNING,),
            values=values,
            require_no_cancel_request=True,
        ):
            await self._raise_for_stopped(job_id)

    async def fail_unpublished(self, job_id: uuid.UUID, error: str) -> None:
        """Fail a job whose message never reached the broker, so it never ran."""
        await self._transition(
            job_id,
            allowed_from=(JobStatus.QUEUED,),
            values={
                "status": JobStatus.FAILED.value,
                "error": error[:_ERROR_LIMIT],
                "failure_kind": FailureKind.UNPUBLISHED.value,
                "completed_at": _now(),
            },
        )

    async def dead_letters(self, limit: int = 50, offset: int = 0) -> Sequence[Job]:
        """Failed jobs that no replay has picked up yet, newest first."""
        replayed = select(Job.replay_of_id).where(Job.replay_of_id.is_not(None))
        async with self._sessions() as session:
            result = await session.execute(
                select(Job)
                .where(Job.status == JobStatus.FAILED.value, Job.id.not_in(replayed))
                .order_by(Job.completed_at.desc(), Job.id)
                .limit(limit)
                .offset(offset)
            )
            return result.scalars().all()

    async def replay_of(self, job_id: uuid.UUID) -> Job | None:
        async with self._sessions() as session:
            result = await session.execute(select(Job).where(Job.replay_of_id == job_id))
            return result.scalars().first()

    async def mark_cancelled(self, job_id: uuid.UUID) -> None:
        await self._finish(job_id, {"status": JobStatus.CANCELLED.value, "completed_at": _now()})

    async def request_cancel(self, job_id: uuid.UUID) -> Job:
        """Cancel a queued job now, or ask a running one to stop."""
        now = _now()
        if await self._transition(
            job_id,
            allowed_from=(JobStatus.QUEUED,),
            values={
                "status": JobStatus.CANCELLED.value,
                "completed_at": now,
                "cancel_requested_at": now,
            },
        ):
            return await self.get(job_id)
        if await self._transition(
            job_id,
            allowed_from=(JobStatus.RUNNING,),
            values={"cancel_requested_at": now},
            require_no_cancel_request=True,
        ):
            return await self.get(job_id)

        job = await self.get(job_id)
        if job.status in TERMINAL_STATUSES:
            raise ConflictException(f"Job '{job_id}' is already {job.status}")
        return job  # running, with a cancel already requested: idempotent

    async def _finish(self, job_id: uuid.UUID, values: dict[str, Any]) -> None:
        if not await self._transition(job_id, allowed_from=(JobStatus.RUNNING,), values=values):
            job = await self.get(job_id)
            raise ConflictException(
                f"Job '{job_id}' cannot become {values['status']} from {job.status}"
            )

    async def _raise_for_stopped(self, job_id: uuid.UUID) -> None:
        job = await self.get(job_id)
        if job.cancel_requested_at is not None or job.status == JobStatus.CANCELLED:
            raise JobCancelled(str(job_id))
        raise ConflictException(f"Job '{job_id}' is {job.status}, not running")

    async def _transition(
        self,
        job_id: uuid.UUID,
        *,
        allowed_from: tuple[JobStatus, ...],
        values: dict[str, Any],
        require_no_cancel_request: bool = False,
    ) -> bool:
        statement = (
            update(Job)
            .where(Job.id == job_id, Job.status.in_([s.value for s in allowed_from]))
            .values(**values, updated_at=_now())
        )
        if require_no_cancel_request:
            statement = statement.where(Job.cancel_requested_at.is_(None))
        async with self._sessions() as session:
            result = await session.execute(statement)
            await session.commit()
        if result.rowcount == 0:  # type: ignore[attr-defined]
            # Distinguish "no such job" from "not in an allowed state".
            await self.get(job_id)
            return False
        return True


def get_job_service() -> JobService:
    return JobService()
