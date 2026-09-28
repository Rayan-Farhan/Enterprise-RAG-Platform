"""Schemas for job status and control (Task 7.2)."""

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class JobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID | None
    version_id: uuid.UUID | None
    task_type: str
    queue: str
    status: str
    attempt: int
    worker: str | None
    progress: float = Field(description="0.0 to 1.0")
    progress_message: str | None
    error: str | None
    cancel_requested_at: datetime | None = Field(
        description="Set while a running job is winding down after a cancel request"
    )
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    updated_at: datetime


class JobListResponse(BaseModel):
    items: list[JobResponse]
