"""Unit tests for Documents REST API endpoints (Task 2.6, ADR-030, ADR-035)."""

import json
import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.testclient import TestClient

from app.db.models import Base
from app.db.session import get_db_session
from app.ingestion.parsers.base import (
    DocumentParser,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedPage,
)
from app.ingestion.service import IngestionService, get_ingestion_service
from app.main import app
from app.storage.base import ObjectStorageProtocol
from app.storage.minio_service import get_storage_service


class DummyApiParser(DocumentParser):
    """Deterministic mock parser for API tests."""

    parser_name: str = "dummy_api_parser"

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        return ParsedDocument(
            filename="test_api_doc.pdf",
            file_type="application/pdf",
            total_pages=1,
            parser_name=self.parser_name,
            parsing_duration_ms=40.0,
            pages=[
                ParsedPage(
                    page_number=1,
                    width=612.0,
                    height=792.0,
                    elements=[
                        ParsedElement(
                            element_id="title_1",
                            element_type=ElementType.HEADING,
                            text="Remote Work Guidelines",
                            page_number=1,
                            level=1,
                        ),
                        ParsedElement(
                            element_id="para_1",
                            element_type=ElementType.PARAGRAPH,
                            text="Employees may work remotely up to 3 days per week with manager approval.",
                            page_number=1,
                        ),
                    ],
                )
            ],
        )


@pytest.fixture
async def test_db_session_factory() -> AsyncGenerator[async_sessionmaker[AsyncSession], None]:
    """Create in-memory SQLite database and session factory."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    yield session_factory

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture
def override_api_dependencies(
    test_db_session_factory: async_sessionmaker[AsyncSession],
) -> Generator[None, None, None]:
    """Override FastAPI dependencies with test database session and mocked storage."""
    mock_storage = MagicMock(spec=ObjectStorageProtocol)
    mock_storage.upload_file.return_value = "original/mock_api_test.pdf"
    mock_storage.get_presigned_url.return_value = "http://minio:9000/test/original/mock_api_test.pdf?sig=test"

    async def _get_test_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_db_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    test_ingestion_service = IngestionService(
        storage_service=mock_storage,
        format_router=None,
    )
    # Inject our mock parser directly into the router of the test ingestion service
    test_ingestion_service.router.pdf_primary = DummyApiParser()

    app.dependency_overrides[get_db_session] = _get_test_session
    app.dependency_overrides[get_storage_service] = lambda: mock_storage
    app.dependency_overrides[get_ingestion_service] = lambda: test_ingestion_service

    yield

    app.dependency_overrides.clear()


def test_document_ingestion_api_flow(client: TestClient, override_api_dependencies: None) -> None:
    """Test POST /api/v1/documents, duplicate check, and GET document endpoints."""
    # 1. Ingest document via POST multipart/form-data
    pdf_content = b"%PDF-1.4 Fake test PDF content for API test"
    metadata_json = json.dumps(
        {
            "department": "Engineering",
            "policy_type": "Remote Work",
            "policy_status": "active",
            "country": "US",
            "authority": "Head of Engineering",
        }
    )

    response = client.post(
        "/api/v1/documents",
        files={"file": ("remote_work_policy.pdf", pdf_content, "application/pdf")},
        data={"metadata": metadata_json},
    )

    assert response.status_code == 201, response.text
    data = response.json()
    assert data["filename"] == "remote_work_policy.pdf"
    assert data["total_pages"] == 1
    assert data["total_elements"] == 2
    assert data["is_duplicate"] is False
    doc_id = data["document_id"]
    ver_id = data["version_id"]

    # 2. Re-upload identical file -> Expect 200 OK and is_duplicate=True
    dup_response = client.post(
        "/api/v1/documents",
        files={"file": ("remote_work_policy_copy.pdf", pdf_content, "application/pdf")},
    )
    assert dup_response.status_code == 200
    dup_data = dup_response.json()
    assert dup_data["is_duplicate"] is True
    assert dup_data["document_id"] == doc_id

    # 3. GET /api/v1/documents (List)
    list_response = client.get("/api/v1/documents?department=Engineering")
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["total"] == 1
    assert list_data["items"][0]["department"] == "Engineering"

    # 4. GET /api/v1/documents/{id} (Details)
    detail_response = client.get(f"/api/v1/documents/{doc_id}")
    assert detail_response.status_code == 200
    detail_data = detail_response.json()
    assert detail_data["id"] == doc_id
    assert len(detail_data["versions"]) == 1
    assert detail_data["versions"][0]["metadata"]["department"] == "Engineering"

    # 5. GET /api/v1/documents/{id}/versions/{version_id}/elements
    elements_response = client.get(f"/api/v1/documents/{doc_id}/versions/{ver_id}/elements")
    assert elements_response.status_code == 200
    elements_data = elements_response.json()
    assert len(elements_data) == 2
    assert elements_data[0]["text_content"] == "Remote Work Guidelines"

    # 6. GET /api/v1/documents/{id}/presigned-url
    url_response = client.get(f"/api/v1/documents/{doc_id}/presigned-url")
    assert url_response.status_code == 200
    assert "presigned_url" in url_response.json()


def test_document_not_found(client: TestClient, override_api_dependencies: None) -> None:
    """Test 404 response when querying non-existent document ID."""
    random_id = uuid.uuid4()
    response = client.get(f"/api/v1/documents/{random_id}")
    assert response.status_code == 404
    data = response.json()
    assert data["code"] == "NOT_FOUND"

