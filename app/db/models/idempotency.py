"""Idempotency records for retried API operations (Task 7.4, ADR-036, master §31).

A client that sends `Idempotency-Key` and retries after a timeout gets the
response of the first attempt instead of starting the work again. A record
holds the request's fingerprint, so the same key on a different request is
refused rather than silently answered with someone else's result.
"""

import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Integer, String, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base, TimestampMixin


class IdempotencyState(StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class IdempotencyRecord(Base, TimestampMixin):
    __tablename__ = "idempotency_records"
    __table_args__ = (
        UniqueConstraint("operation", "key", name="uq_idempotency_records_operation_key"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    operation: Mapped[str] = mapped_column(String(64), nullable=False)
    key: Mapped[str] = mapped_column(String(255), nullable=False)
    request_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(
        String(16), default=IdempotencyState.IN_PROGRESS.value, nullable=False
    )
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_body: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<IdempotencyRecord(operation={self.operation}, key={self.key}, state={self.state})>"
        )
