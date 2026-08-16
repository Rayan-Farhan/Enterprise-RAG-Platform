"""Unit tests for the end-to-end Ingestion Service (Task 2.6, Stage 2 Exit Gate)."""

from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.repositories.document_repo import DocumentRepository
from app.ingestion.parsers.base import (
    DocumentParser,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedPage,
)
from app.ingestion.service import IngestionService
from app.storage.base import ObjectStorageProtocol


class DummyMockParser(DocumentParser):
    """Deterministic parser mock for testing the ingestion pipeline."""

    parser_name: str = "mock_test_parser"

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        return ParsedDocument(
            filename="mock_handbook.pdf",
            file_type="application/pdf",
            total_pages=1,
            parser_name=self.parser_name,
            parsing_duration_ms=50.0,
            pages=[
                ParsedPage(
                    page_number=1,
                    width=612.0,
                    height=792.0,
                    elements=[
                        ParsedElement(
                            element_id="hdr_1",
                            element_type=ElementType.HEADING,
                            text="Corporate Code of Conduct",
                            page_number=1,
                            level=1,
                        ),
                        ParsedElement(
                            element_id="p_1",
                            element_type=ElementType.PARAGRAPH,
                            text="All staff are required to uphold high ethical standards.",
                            page_number=1,
                        ),
                    ],
                )
            ],
        )


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide isolated in-memory database session."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()



@pytest.mark.asyncio
async def test_ingest_document_end_to_end(db_session: AsyncSession) -> None:
    """Assert a file is parsed, uploaded to storage, and persisted to database."""
    mock_storage = MagicMock(spec=ObjectStorageProtocol)
    mock_storage.upload_file.return_value = "original/mock_hash.pdf"

    service = IngestionService(
        storage_service=mock_storage,
    )

    file_bytes = b"%PDF-1.4 Mock PDF file binary data..."
    filename = "employee_handbook.pdf"
    metadata_input = {
        "department": "Legal & Compliance",
        "policy_type": "Ethics & Conduct",
        "country": "Global",
    }

    # Ingest document
    result = await service.ingest_document(
        session=db_session,
        file_content=file_bytes,
        filename=filename,
        metadata_dict=metadata_input,
        parser_override=DummyMockParser(),
    )

    # 1. Assert Result DTO
    assert result.is_duplicate is False
    assert result.total_pages == 1
    assert result.total_elements == 2
    assert result.filename == filename
    assert len(result.file_hash) == 64

    # 2. Assert Storage upload was called
    mock_storage.upload_file.assert_called()

    # 3. Assert Database Persistence via Repository
    repo = DocumentRepository(db_session)
    persisted_doc = await repo.get_by_id(result.document_id)
    assert persisted_doc is not None
    assert persisted_doc.title == filename
    assert len(persisted_doc.versions) == 1
    assert persisted_doc.versions[0].metadata_record is not None
    assert persisted_doc.versions[0].metadata_record.department == "Legal & Compliance"

    elements = await repo.get_elements_by_version(result.version_id)
    assert len(elements) == 2
    assert elements[0].text_content == "Corporate Code of Conduct"


@pytest.mark.asyncio
async def test_ingest_duplicate_short_circuits(db_session: AsyncSession) -> None:
    """Assert ingesting the same binary content twice short-circuits on exact hash match."""
    mock_storage = MagicMock(spec=ObjectStorageProtocol)
    service = IngestionService(storage_service=mock_storage)

    file_bytes = b"Identical binary payload for duplicate test"
    filename = "security_policy.pdf"

    # First Ingestion
    res1 = await service.ingest_document(
        session=db_session,
        file_content=file_bytes,
        filename=filename,
        parser_override=DummyMockParser(),
    )
    assert res1.is_duplicate is False

    # Second Ingestion with same bytes
    res2 = await service.ingest_document(
        session=db_session,
        file_content=file_bytes,
        filename="security_policy_copy.pdf",
        parser_override=DummyMockParser(),
    )

    assert res2.is_duplicate is True
    assert res2.document_id == res1.document_id
    assert res2.file_hash == res1.file_hash
    assert "already ingested" in res2.message
