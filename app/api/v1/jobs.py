"""Job status and control (Task 7.2, master §30).

Job state is read from PostgreSQL, where workers commit it as they go, so these
endpoints show a long ingestion's progress while it is still running.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.schemas.jobs import JobListResponse, JobResponse
from app.core.exceptions import NotFoundException
from app.db.models.document import Document
from app.db.session import get_db_session
from app.jobs.service import JobService, get_job_service

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
