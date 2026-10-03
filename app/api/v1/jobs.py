"""Job status and control (Tasks 7.2, 7.5, master §30).

Job state is read from PostgreSQL, where workers commit it as they go, so these
endpoints show a long ingestion's progress while it is still running.

The dead-letter queue is the set of failed jobs no replay has picked up: a job
lands there when its error is permanent, its retries ran out, or it was
delivered too often to trust (Task 7.5). Replaying one runs it again as a new
job; resuming a version restarts its chain at the first unfinished step.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.jobs import JobListResponse, JobResponse
from app.core.exceptions import NotFoundException
from app.db.models.document import Document
from app.db.models.version import DocumentVersion
from app.db.session import get_db_session
from app.jobs.service import JobService, get_job_service
from app.workers.job_task import replay
from app.workers.tasks.ingestion import resume_ingestion

router = APIRouter(tags=["Jobs"])


@router.get(
    "/documents/{document_id}/jobs",
    response_model=JobListResponse,
    summary="Every job run for a document, oldest first",
)
async def list_document_jobs(
    document_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    jobs: JobService = Depends(get_job_service),
) -> JobListResponse:
    if await session.get(Document, document_id) is None:
        raise NotFoundException(f"Document with ID '{document_id}' was not found")
    items = await jobs.list_for_document(document_id)
    return JobListResponse(items=[JobResponse.model_validate(j) for j in items])


@router.get("/jobs/{job_id}", response_model=JobResponse, summary="One job's current state")
async def get_job(job_id: uuid.UUID, jobs: JobService = Depends(get_job_service)) -> JobResponse:
    return JobResponse.model_validate(await jobs.get(job_id))


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Cancel a queued job, or ask a running one to stop at its next checkpoint",
)
async def cancel_job(job_id: uuid.UUID, jobs: JobService = Depends(get_job_service)) -> JobResponse:
    return JobResponse.model_validate(await jobs.request_cancel(job_id))


@router.get(
    "/dead-letters",
    response_model=JobListResponse,
    summary="Failed jobs awaiting inspection or replay, newest first",
)
async def list_dead_letters(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    jobs: JobService = Depends(get_job_service),
) -> JobListResponse:
    items = await jobs.dead_letters(limit=limit, offset=offset)
    return JobListResponse(items=[JobResponse.model_validate(j) for j in items])


@router.post(
    "/dead-letters/{job_id}/replay",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run a dead-lettered job again as a new job",
)
async def replay_dead_letter(
    job_id: uuid.UUID, jobs: JobService = Depends(get_job_service)
) -> JobResponse:
    return JobResponse.model_validate(await replay(await jobs.get(job_id), jobs))


@router.post(
    "/documents/{document_id}/versions/{version_id}/resume",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Restart a version's ingestion at its first unfinished step",
)
async def resume_version(
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    session: AsyncSession = Depends(get_db_session),
    jobs: JobService = Depends(get_job_service),
) -> JobResponse:
    version = await session.get(DocumentVersion, version_id)
    if version is None or version.document_id != document_id:
        raise NotFoundException(f"Version '{version_id}' of document '{document_id}' was not found")
    return JobResponse.model_validate(await resume_ingestion(version_id, jobs))
