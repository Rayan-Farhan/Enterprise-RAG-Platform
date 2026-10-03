"""Idempotency keys for retried API operations (Task 7.4, ADR-036, master §31).

    POST /documents                 Idempotency-Key: 6f1c...
    → 202 {"job_id": "..."}         (times out on the client)
    POST /documents                 Idempotency-Key: 6f1c...   (same file)
    → 202 {"job_id": "..."}         Idempotent-Replayed: true  (same job, no new work)

The key is optional; without one, an operation still cannot duplicate data,
because uploads deduplicate by content hash and every downstream identity is
deterministic. What the key adds is the *response*: a client that never saw
the first answer gets it back, job id included.

A key is claimed before the operation runs, in its own committed transaction,
so a concurrent retry sees the claim and gets 409 instead of racing the first
attempt. A failed operation releases its claim, so a retry can try again.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from fastapi import Response
from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.exceptions import ConflictException, ValidationException
from app.db.models.base import utc_now
from app.db.models.idempotency import IdempotencyRecord, IdempotencyState
from app.db.session import get_session_factory

IDEMPOTENCY_HEADER = "Idempotency-Key"
REPLAYED_HEADER = "Idempotent-Replayed"
MAX_KEY_LENGTH = 255

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


def fingerprint(*parts: Any) -> str:
    """A stable digest of what makes two requests the same request."""
    encoded = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StoredResponse:
    status_code: int
    body: dict[str, Any]


class IdempotencyService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession] | None = None) -> None:
        self._sessions = session_factory or get_session_factory()

    async def claim(
        self, operation: str, key: str, request_fingerprint: str
    ) -> StoredResponse | None:
        """Claim ``key`` for this request, or return the response it already produced.

        Raises ValidationException when the key was used for a different request
        and ConflictException while the first attempt is still running.
        """
        if not key or len(key) > MAX_KEY_LENGTH:
            raise ValidationException(
                f"{IDEMPOTENCY_HEADER} must be 1 to {MAX_KEY_LENGTH} characters"
            )
        async with self._sessions() as session:
            session.add(
                IdempotencyRecord(
                    operation=operation,
                    key=key,
                    request_fingerprint=request_fingerprint,
                    state=IdempotencyState.IN_PROGRESS.value,
                )
            )
            try:
                await session.commit()
                return None
            except IntegrityError:
                await session.rollback()
            record = (
                await session.execute(
                    select(IdempotencyRecord).where(
                        IdempotencyRecord.operation == operation, IdempotencyRecord.key == key
                    )
                )
            ).scalar_one()

        if record.request_fingerprint != request_fingerprint:
            raise ValidationException(
                f"{IDEMPOTENCY_HEADER} '{key}' was already used for a different request"
            )
        if record.state != IdempotencyState.COMPLETED or record.status_code is None:
            raise ConflictException(
                f"A request with {IDEMPOTENCY_HEADER} '{key}' is still in progress; retry shortly"
            )
        return StoredResponse(status_code=record.status_code, body=record.response_body or {})

    async def complete(
        self, operation: str, key: str, status_code: int, body: dict[str, Any]
    ) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(IdempotencyRecord)
                .where(IdempotencyRecord.operation == operation, IdempotencyRecord.key == key)
                .values(
                    state=IdempotencyState.COMPLETED.value,
                    status_code=status_code,
                    response_body=body,
                    updated_at=utc_now(),
                )
            )
            await session.commit()

    async def release(self, operation: str, key: str) -> None:
        """Drop an unfinished claim so a retry can run the operation."""
        async with self._sessions() as session:
            await session.execute(
                delete(IdempotencyRecord).where(
                    IdempotencyRecord.operation == operation,
                    IdempotencyRecord.key == key,
                    IdempotencyRecord.state == IdempotencyState.IN_PROGRESS.value,
                )
            )
            await session.commit()


def get_idempotency_service() -> IdempotencyService:
    return IdempotencyService()


async def run_idempotently(
    service: IdempotencyService,
    *,
    operation: str,
    key: str | None,
    request_fingerprint: str,
    response: Response,
    default_status: int,
    model: type[ResponseModel],
    run: Callable[[], Awaitable[ResponseModel]],
) -> ResponseModel:
    """Run ``run`` once per key; later calls with the key get the first response back.

    ``run`` must commit its own work before returning: the stored response is a
    promise that the work it describes exists.
    """
    if key is None:
        return await run()

    stored = await service.claim(operation, key, request_fingerprint)
    if stored is not None:
        response.status_code = stored.status_code
        response.headers[REPLAYED_HEADER] = "true"
        return model.model_validate(stored.body)

    try:
        result = await run()
    except BaseException:
        await service.release(operation, key)
        raise
    await service.complete(
        operation, key, response.status_code or default_status, result.model_dump(mode="json")
    )
    return result
