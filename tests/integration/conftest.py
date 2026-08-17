"""Integration fixtures: a real database plus in-memory doubles for external services."""

from __future__ import annotations

import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import AppSettings
from app.db.models.base import Base
from app.db.models.document import Document
from app.db.models.element import Element
from app.db.models.metadata import DocumentMetadata
from app.db.models.page import Page
from app.db.models.version import DocumentVersion, VersionStatus


@pytest.fixture
def settings() -> AppSettings:
    """Deterministic settings for the Stage 3 pipeline."""
    return AppSettings(
        APP_ENV="testing",
        CHUNK_SIZE_TOKENS=80,
        CHUNK_OVERLAP_TOKENS=16,
        CHUNKING_VERSION="fixed-v1",
        EMBEDDING_VERSION="test-embed-v1",
        EMBEDDING_DIMENSIONS=4,
        EMBEDDING_BATCH_SIZE=4,
        EMBEDDING_MAX_RPM=10_000,
        RETRIEVAL_TOP_K=5,
        RETRIEVAL_MIN_SCORE=0.0,
        GENERATION_MAX_CONTEXT_TOKENS=2000,
    )


@pytest_asyncio.fixture
async def session() -> AsyncGenerator[AsyncSession, None]:
    """A real SQLAlchemy session over in-memory SQLite.

    SQLite is used rather than mocks so the ORM relationships, JSON columns, and
    cascade behaviour are genuinely exercised. The PostgreSQL-specific uniqueness
    constraint is created from the model metadata, so idempotency is enforced here
    exactly as it will be in Postgres.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as db_session:
        yield db_session

    await engine.dispose()


@pytest_asyncio.fixture
async def hr_document(session: AsyncSession) -> DocumentVersion:
    """Persist a small but realistic HR policy document with heading hierarchy."""
    document = Document(
        id=uuid.uuid4(),
        title="Staff Handbook 2026",
        mime_type="application/pdf",
        file_size_bytes=204_800,
        file_hash="a" * 64,
        storage_key="original/aaaa/staff_handbook.pdf",
    )
    session.add(document)
    await session.flush()

    version = DocumentVersion(
        id=uuid.uuid4(),
        document_id=document.id,
        version_number=1,
        status=str(VersionStatus.ACTIVE),
        total_pages=2,
        total_elements=0,
        parser_name="docling",
        parsing_duration_ms=1234.5,
        effective_from=datetime(2026, 1, 1, tzinfo=UTC),
    )
    session.add(version)
    await session.flush()

    session.add(
        DocumentMetadata(
            id=uuid.uuid4(),
            version_id=version.id,
            department="Human Resources",
            policy_type="leave",
            policy_status="active",
            country="US",
            employee_type="full_time",
        )
    )

    pages = [
        Page(
            id=uuid.uuid4(),
            version_id=version.id,
            page_number=n,
            width=595.0,
            height=842.0,
            content_hash=f"{n:064d}",
        )
        for n in (1, 2)
    ]
    for page in pages:
        session.add(page)
    await session.flush()

    page_by_number = {p.page_number: p for p in pages}

    specs: list[tuple[str, str, str, int, int | None]] = [
        ("h1", "Leave Policy", "heading", 1, 1),
        ("h2", "Annual Leave", "heading", 1, 2),
        (
            "p1",
            "All full-time employees are entitled to 21 days of paid annual leave "
            "per calendar year. Leave accrues monthly from the date of joining.",
            "paragraph",
            1,
            None,
        ),
        (
            "p2",
            "Unused annual leave may be carried forward to the following year up to "
            "a maximum of 5 days. Any balance above 5 days is forfeited.",
            "paragraph",
            1,
            None,
        ),
        ("h3", "Sick Leave", "heading", 2, 2),
        (
            "p3",
            "Employees are entitled to 10 days of paid sick leave per year. A medical "
            "certificate is required for absences exceeding two consecutive days.",
            "paragraph",
            2,
            None,
        ),
        ("f1", "Confidential — Staff Handbook — Page footer", "footer", 2, None),
    ]

    for index, (element_id, text, element_type, page_number, level) in enumerate(specs):
        session.add(
            Element(
                id=uuid.uuid4(),
                version_id=version.id,
                page_id=page_by_number[page_number].id,
                page_number=page_number,
                element_id=element_id,
                element_type=element_type,
                sequence_index=index,
                text_content=text,
                content_hash=f"{index:064d}",
                bounding_box={
                    "x0": 50.0,
                    "y0": 100.0 + index * 20,
                    "x1": 500.0,
                    "y1": 120.0 + index * 20,
                    "page_number": page_number,
                    "unit": "pt",
                },
                is_boilerplate=element_type == "footer",
                boilerplate_reason="repeated footer" if element_type == "footer" else None,
                extra_metadata={"heading_level": level} if level else {},
            )
        )

    version.total_elements = len(specs)
    await session.flush()
    await session.commit()

    version.document = document
    return version
