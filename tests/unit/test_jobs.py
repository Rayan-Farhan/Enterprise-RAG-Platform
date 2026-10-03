"""Job lifecycle over a real ORM, the worker binding, and the jobs API (Task 7.2)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from starlette.testclient import TestClient

from app.core.exceptions import ConflictException, NotFoundException
from app.db.models.base import Base
from app.db.models.document import Document
from app.db.models.job import Job, JobStatus, JobType
from app.db.session import get_db_session
from app.jobs.service import JobAlreadyFinished, JobCancelled, JobService, get_job_service
from app.main import app
from app.workers import job_task as job_task_module
from app.workers.job_task import JobContext, enqueue, execute_job


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine, None]:
    # A file, not :memory:, so execute_job's own engine sees the same database.
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def jobs(sessions: async_sessionmaker[AsyncSession]) -> JobService:
    return JobService(sessions)


@pytest.fixture
def worker_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine: AsyncEngine) -> None:
    url = f"sqlite+aiosqlite:///{tmp_path / 'jobs.db'}"
    monkeypatch.setattr(job_task_module, "worker_engine_factory", lambda: create_async_engine(url))


async def queued(jobs: JobService, **kwargs: Any) -> uuid.UUID:
    job = await jobs.create(task_type=JobType.DIAGNOSTIC, queue="ingestion", **kwargs)
    return job.id


class TestTransitions:
    async def test_a_new_job_is_queued_with_no_attempts(self, jobs: JobService) -> None:
        job = await jobs.get(await queued(jobs))

        assert job.status == JobStatus.QUEUED
        assert job.attempt == 0
        assert job.progress == 0.0

    async def test_start_claims_the_job_for_a_worker(self, jobs: JobService) -> None:
        job_id = await queued(jobs)

        job = await jobs.start(job_id, worker="document@host")

        assert job.status == JobStatus.RUNNING
        assert job.attempt == 1
        assert job.worker == "document@host"
        assert job.started_at is not None

    async def test_redelivery_of_a_running_job_counts_a_new_attempt(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="a@host")

        job = await jobs.start(job_id, worker="b@host")

        assert job.attempt == 2
        assert job.worker == "b@host"

    async def test_progress_is_visible_while_running(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")

        await jobs.report_progress(job_id, 0.4, "page 40 of 100")

        job = await jobs.get(job_id)
        assert job.progress == pytest.approx(0.4)
        assert job.progress_message == "page 40 of 100"

    @pytest.mark.parametrize("fraction", [-0.1, 1.5])
    async def test_progress_outside_zero_to_one_is_rejected(
        self, jobs: JobService, fraction: float
    ) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")

        with pytest.raises(ValueError):
            await jobs.report_progress(job_id, fraction)

    async def test_long_progress_message_is_truncated_not_rejected(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")

        await jobs.report_progress(job_id, 0.1, "x" * 1000)

        assert len((await jobs.get(job_id)).progress_message or "") == 255

    async def test_progress_on_a_queued_job_is_a_conflict(self, jobs: JobService) -> None:
        job_id = await queued(jobs)

        with pytest.raises(ConflictException):
            await jobs.report_progress(job_id, 0.5)

    async def test_success_completes_progress(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")

        await jobs.succeed(job_id)

        job = await jobs.get(job_id)
        assert job.status == JobStatus.SUCCEEDED
        assert job.progress == 1.0
        assert job.completed_at is not None

    async def test_failure_records_the_error(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")

        await jobs.fail(job_id, "ParserError: page 12 is encrypted")

        job = await jobs.get(job_id)
        assert job.status == JobStatus.FAILED
        assert job.error == "ParserError: page 12 is encrypted"

    async def test_a_finished_job_cannot_finish_again(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")
        await jobs.succeed(job_id)

        with pytest.raises(ConflictException):
            await jobs.fail(job_id, "late failure")

    async def test_a_finished_job_is_not_restarted_by_a_redelivery(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")
        await jobs.succeed(job_id)

        with pytest.raises(JobAlreadyFinished):
            await jobs.start(job_id, worker="w")

    async def test_unknown_job_is_not_found(self, jobs: JobService) -> None:
        with pytest.raises(NotFoundException):
            await jobs.start(uuid.uuid4(), worker="w")

    async def test_unpublished_job_fails_without_ever_running(self, jobs: JobService) -> None:
        job_id = await queued(jobs)

        await jobs.fail_unpublished(job_id, "broker down")

        job = await jobs.get(job_id)
        assert job.status == JobStatus.FAILED
        assert job.attempt == 0


class TestCancellation:
    async def test_a_queued_job_is_cancelled_at_once(self, jobs: JobService) -> None:
        job_id = await queued(jobs)

        job = await jobs.request_cancel(job_id)

        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None

    async def test_a_cancelled_job_never_starts(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.request_cancel(job_id)

        with pytest.raises(JobCancelled):
            await jobs.start(job_id, worker="w")

    async def test_a_running_job_is_asked_to_stop_and_keeps_running(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")

        job = await jobs.request_cancel(job_id)

        assert job.status == JobStatus.RUNNING
        assert job.cancel_requested_at is not None

    async def test_the_request_stops_the_job_at_its_next_progress_report(
        self, jobs: JobService
    ) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")
        await jobs.request_cancel(job_id)

        with pytest.raises(JobCancelled):
            await jobs.report_progress(job_id, 0.6)
        with pytest.raises(JobCancelled):
            await jobs.checkpoint(job_id)

    async def test_repeating_a_cancel_request_is_harmless(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")
        first = await jobs.request_cancel(job_id)

        second = await jobs.request_cancel(job_id)

        assert second.cancel_requested_at == first.cancel_requested_at

    async def test_cancelling_a_finished_job_is_a_conflict(self, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")
        await jobs.succeed(job_id)

        with pytest.raises(ConflictException, match="succeeded"):
            await jobs.request_cancel(job_id)

    async def test_redelivery_after_a_cancel_request_records_the_cancel(
        self, jobs: JobService
    ) -> None:
        # The worker running it died before seeing the request.
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="lost")
        await jobs.request_cancel(job_id)

        with pytest.raises(JobCancelled):
            await jobs.start(job_id, worker="next")
        assert (await jobs.get(job_id)).status == JobStatus.CANCELLED


class TestExecuteJob:
    async def test_body_runs_and_the_job_succeeds(self, jobs: JobService, worker_db: None) -> None:
        job_id = await queued(jobs)
        seen: list[float] = []

        async def body(ctx: JobContext, *, pages: int) -> None:
            for page in range(1, pages + 1):
                await ctx.progress(page / pages)
                seen.append((await jobs.get(ctx.job_id)).progress)

        outcome = await execute_job(body, job_id, "w@host", {"pages": 4})

        assert outcome == "succeeded"
        assert seen == [0.25, 0.5, 0.75, 1.0]
        assert (await jobs.get(job_id)).status == JobStatus.SUCCEEDED

    async def test_body_exception_fails_the_job_and_propagates(
        self, jobs: JobService, worker_db: None
    ) -> None:
        job_id = await queued(jobs)

        async def body(ctx: JobContext) -> None:
            raise RuntimeError("OCR engine crashed")

        with pytest.raises(RuntimeError):
            await execute_job(body, job_id, "w", {})

        job = await jobs.get(job_id)
        assert job.status == JobStatus.FAILED
        assert job.error == "RuntimeError: OCR engine crashed"

    async def test_cancel_mid_run_stops_at_the_next_checkpoint(
        self, jobs: JobService, worker_db: None
    ) -> None:
        job_id = await queued(jobs)
        steps_done = 0

        async def body(ctx: JobContext) -> None:
            nonlocal steps_done
            for step in range(1, 6):
                await ctx.progress(step / 5)
                steps_done = step
                if step == 2:
                    await jobs.request_cancel(ctx.job_id)

        assert await execute_job(body, job_id, "w", {}) == "cancelled"
        assert steps_done == 2
        assert (await jobs.get(job_id)).status == JobStatus.CANCELLED

    async def test_a_job_cancelled_while_queued_never_runs_its_body(
        self, jobs: JobService, worker_db: None
    ) -> None:
        job_id = await queued(jobs)
        await jobs.request_cancel(job_id)
        ran = False

        async def body(ctx: JobContext) -> None:
            nonlocal ran
            ran = True

        assert await execute_job(body, job_id, "w", {}) == "cancelled"
        assert not ran

    async def test_a_redelivered_finished_job_is_skipped(
        self, jobs: JobService, worker_db: None
    ) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")
        await jobs.succeed(job_id)

        async def body(ctx: JobContext) -> None:
            raise AssertionError("must not run")

        assert await execute_job(body, job_id, "w", {}) == "skipped"


class FakeTask:
    name = "tests.fake_task"
    queue = "parsing"

    def __init__(self, jobs: JobService, fail: bool = False) -> None:
        self.jobs = jobs
        self.fail = fail
        self.published: list[dict[str, Any]] = []

    def apply_async(self, kwargs: dict[str, Any], task_id: str) -> None:
        if self.fail:
            raise ConnectionError("broker unreachable")
        self.published.append({"kwargs": kwargs, "task_id": task_id})


class TestEnqueue:
    async def test_the_row_exists_before_the_message_and_shares_its_id(
        self, jobs: JobService
    ) -> None:
        task = FakeTask(jobs)

        job = await enqueue(
            task,  # type: ignore[arg-type]
            task_type=JobType.PARSE_DOCUMENT,
            jobs=jobs,
            storage_key="original/x.pdf",
        )

        assert task.published == [
            {
                "kwargs": {"job_id": str(job.id), "storage_key": "original/x.pdf"},
                "task_id": str(job.id),
            }
        ]
        stored = await jobs.get(job.id)
        assert stored.queue == "parsing"
        assert stored.status == JobStatus.QUEUED

    async def test_a_publish_failure_fails_the_job_instead_of_leaving_it_queued(
        self, jobs: JobService, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        task = FakeTask(jobs, fail=True)

        with pytest.raises(ConnectionError):
            await enqueue(task, task_type=JobType.PARSE_DOCUMENT, jobs=jobs)  # type: ignore[arg-type]

        async with sessions() as session:
            (job,) = (await session.execute(select(Job))).scalars().all()
        assert job.status == JobStatus.FAILED
        assert "broker unreachable" in (job.error or "")


class TestJobsApi:
    @pytest.fixture
    def client(
        self, jobs: JobService, sessions: async_sessionmaker[AsyncSession]
    ) -> Generator[TestClient, None, None]:
        async def session_override() -> AsyncGenerator[AsyncSession, None]:
            async with sessions() as session:
                yield session

        app.dependency_overrides[get_job_service] = lambda: jobs
        app.dependency_overrides[get_db_session] = session_override
        with TestClient(app) as client:
            yield client
        app.dependency_overrides.clear()

    @pytest.fixture
    async def document_id(self, sessions: async_sessionmaker[AsyncSession]) -> uuid.UUID:
        document = Document(
            id=uuid.uuid4(),
            title="Handbook",
            mime_type="application/pdf",
            file_size_bytes=1,
            file_hash="f" * 64,
            storage_key="original/f/handbook.pdf",
        )
        async with sessions() as session:
            session.add(document)
            await session.commit()
        return document.id

    async def test_lists_a_documents_jobs_oldest_first(
        self, client: TestClient, jobs: JobService, document_id: uuid.UUID
    ) -> None:
        first = await queued(jobs, document_id=document_id)
        second = await queued(jobs, document_id=document_id)
        await queued(jobs)  # another document's job must not appear
        await jobs.start(first, worker="w")
        await jobs.report_progress(first, 0.5, "halfway")

        response = client.get(f"/api/v1/documents/{document_id}/jobs")

        assert response.status_code == 200
        items = response.json()["items"]
        assert [i["id"] for i in items] == [str(first), str(second)]
        assert items[0]["status"] == "running"
        assert items[0]["progress"] == 0.5
        assert items[0]["progress_message"] == "halfway"

    def test_jobs_of_an_unknown_document_is_404(self, client: TestClient) -> None:
        assert client.get(f"/api/v1/documents/{uuid.uuid4()}/jobs").status_code == 404

    async def test_get_one_job(self, client: TestClient, jobs: JobService) -> None:
        job_id = await queued(jobs)

        response = client.get(f"/api/v1/jobs/{job_id}")

        assert response.status_code == 200
        assert response.json()["status"] == "queued"

    def test_unknown_job_is_404(self, client: TestClient) -> None:
        assert client.get(f"/api/v1/jobs/{uuid.uuid4()}").status_code == 404

    async def test_cancel_a_queued_job(self, client: TestClient, jobs: JobService) -> None:
        job_id = await queued(jobs)

        response = client.post(f"/api/v1/jobs/{job_id}/cancel")

        assert response.status_code == 202
        assert response.json()["status"] == "cancelled"

    async def test_cancel_a_finished_job_is_409(self, client: TestClient, jobs: JobService) -> None:
        job_id = await queued(jobs)
        await jobs.start(job_id, worker="w")
        await jobs.succeed(job_id)

        assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 409

    async def _dead(self, jobs: JobService) -> uuid.UUID:
        job = await jobs.create(
            task_type=JobType.DIAGNOSTIC,
            queue="ingestion",
            task_name="diagnostics.exercise_job",
            payload={"steps": 1, "step_seconds": 0.0},
        )
        await jobs.start(job.id, worker="w")
        await jobs.fail(job.id, "ValueError: bad input")
        return job.id

    async def test_dead_letters_list_failed_jobs_with_why(
        self, client: TestClient, jobs: JobService
    ) -> None:
        dead = await self._dead(jobs)
        await queued(jobs)  # a live job is not a dead letter

        items = client.get("/api/v1/dead-letters").json()["items"]

        assert [(i["id"], i["failure_kind"]) for i in items] == [(str(dead), "permanent")]

    async def test_replay_sends_the_recorded_task_as_a_new_job(
        self, client: TestClient, jobs: JobService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.workers.tasks.diagnostics import exercise_job

        sent: list[dict[str, Any]] = []
        monkeypatch.setattr(
            exercise_job, "apply_async", lambda *, kwargs, task_id: sent.append(kwargs)
        )
        dead = await self._dead(jobs)

        response = client.post(f"/api/v1/dead-letters/{dead}/replay")

        assert response.status_code == 202
        body = response.json()
        assert body["replay_of_id"] == str(dead)
        assert body["status"] == "queued"
        assert sent == [{"job_id": body["id"], "steps": 1, "step_seconds": 0.0}]
        assert client.get("/api/v1/dead-letters").json()["items"] == []
        assert client.post(f"/api/v1/dead-letters/{dead}/replay").status_code == 409

    async def test_a_live_job_cannot_be_replayed(
        self, client: TestClient, jobs: JobService
    ) -> None:
        job_id = await queued(jobs)

        assert client.post(f"/api/v1/dead-letters/{job_id}/replay").status_code == 409

    async def test_resuming_an_unknown_version_is_404(
        self, client: TestClient, document_id: uuid.UUID
    ) -> None:
        response = client.post(f"/api/v1/documents/{document_id}/versions/{uuid.uuid4()}/resume")

        assert response.status_code == 404
