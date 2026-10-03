"""Unit tests for Documents REST API endpoints (Task 2.6, ADR-030, ADR-035)."""

import json
import uuid
from collections.abc import AsyncGenerator, Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from starlette.testclient import TestClient

from app.api.idempotency import IdempotencyService, get_idempotency_service
from app.core.config import AppSettings
from app.core.exceptions import ConflictException
from app.db.models import Base
from app.db.session import get_db_session
from app.ingestion import pipeline
from app.ingestion.dedup import BoilerplateDetector
from app.ingestion.parsers.base import (
    DocumentParser,
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedPage,
)
from app.ingestion.parsers.router import FormatRouter
from app.jobs.service import JobService, get_job_service
from app.main import app
from app.storage.minio_service import get_storage_service
from app.workers.tasks import ingestion as ingestion_tasks
from tests.unit.test_ingestion_chain import MemoryStorage


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
def storage() -> MemoryStorage:
    return MemoryStorage()


@pytest.fixture
def published(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Messages the upload would have sent to RabbitMQ."""
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(
        ingestion_tasks.parse_document,
        "apply_async",
        lambda *, kwargs, task_id: sent.append(kwargs),
    )
    return sent


@pytest.fixture
def override_api_dependencies(
    test_db_session_factory: async_sessionmaker[AsyncSession],
    storage: MemoryStorage,
    published: list[dict[str, Any]],
) -> Generator[None, None, None]:
    """Override FastAPI dependencies with the test database and in-memory storage."""

    async def _get_test_session() -> AsyncGenerator[AsyncSession, None]:
        async with test_db_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = _get_test_session
    app.dependency_overrides[get_storage_service] = lambda: storage
    app.dependency_overrides[get_job_service] = lambda: JobService(test_db_session_factory)
    app.dependency_overrides[get_idempotency_service] = lambda: IdempotencyService(
        test_db_session_factory
    )

    yield

    app.dependency_overrides.clear()


async def _parse_and_normalize(
    sessions: async_sessionmaker[AsyncSession], storage: MemoryStorage, version_id: uuid.UUID
) -> None:
    """The chain steps that produce canonical elements, run in place of the workers."""
    settings = AppSettings(APP_ENV="testing")
    services = pipeline.PipelineServices(
        settings=settings,
        storage=storage,
        router=FormatRouter(pdf_primary=DummyApiParser()),
        boilerplate=BoilerplateDetector(),
        chunking=MagicMock(),
        chunk_indexer=MagicMock(),
        lexical_indexer=MagicMock(),
        sparse_indexer=MagicMock(),
    )
    for step in (pipeline.parse_document, pipeline.extract_pages, pipeline.ocr_pages):
        async with sessions() as session:
            await step(services, session, version_id)
    metadata = json.loads(METADATA_JSON)
    async with sessions() as session:
        await pipeline.normalize_document(services, session, version_id, metadata=metadata)


METADATA_JSON = json.dumps(
    {
        "department": "Engineering",
        "policy_type": "Remote Work",
        "policy_status": "active",
        "country": "US",
        "authority": "Head of Engineering",
    }
)


def test_upload_is_accepted_with_a_job_and_parses_nothing(
    client: TestClient,
    override_api_dependencies: None,
    published: list[dict[str, Any]],
    storage: MemoryStorage,
) -> None:
    response = client.post(
        "/api/v1/documents",
        files={"file": ("remote_work_policy.pdf", b"%PDF-1.4 accepted", "application/pdf")},
        data={"metadata": METADATA_JSON},
    )

    assert response.status_code == 202, response.text
    data = response.json()
    assert data["is_duplicate"] is False
    assert data["job_id"] is not None
    assert storage.objects[data["storage_key"]] == b"%PDF-1.4 accepted"
    assert published == [{"job_id": data["job_id"], "metadata": json.loads(METADATA_JSON)}]

    jobs = client.get(f"/api/v1/documents/{data['document_id']}/jobs").json()["items"]
    assert [(j["id"], j["task_type"], j["status"]) for j in jobs] == [
        (data["job_id"], "parse_document", "queued")
    ]
    detail = client.get(f"/api/v1/documents/{data['document_id']}").json()
    assert detail["versions"][0]["status"] == "draft"


async def test_document_ingestion_api_flow(
    client: TestClient,
    override_api_dependencies: None,
    test_db_session_factory: async_sessionmaker[AsyncSession],
    storage: MemoryStorage,
) -> None:
    """POST /api/v1/documents, the duplicate check, and the read endpoints once parsed."""
    pdf_content = b"%PDF-1.4 Fake test PDF content for API test"

    response = client.post(
        "/api/v1/documents",
        files={"file": ("remote_work_policy.pdf", pdf_content, "application/pdf")},
        data={"metadata": METADATA_JSON},
    )

    assert response.status_code == 202, response.text
    data = response.json()
    assert data["filename"] == "remote_work_policy.pdf"
    doc_id = data["document_id"]
    ver_id = data["version_id"]

    # Re-upload identical file -> 200 OK, the same document, and no new job
    dup_response = client.post(
        "/api/v1/documents",
        files={"file": ("remote_work_policy_copy.pdf", pdf_content, "application/pdf")},
    )
    assert dup_response.status_code == 200
    dup_data = dup_response.json()
    assert dup_data["is_duplicate"] is True
    assert dup_data["document_id"] == doc_id
    assert dup_data["job_id"] is None

    await _parse_and_normalize(test_db_session_factory, storage, uuid.UUID(ver_id))

    # GET /api/v1/documents (List)
    list_response = client.get("/api/v1/documents?department=Engineering")
    assert list_response.status_code == 200
    list_data = list_response.json()
    assert list_data["total"] == 1
    assert list_data["items"][0]["department"] == "Engineering"
    assert list_data["items"][0]["total_pages"] == 1

    # GET /api/v1/documents/{id} (Details)
    detail_response = client.get(f"/api/v1/documents/{doc_id}")
    assert detail_response.status_code == 200
    detail_data = detail_response.json()
    assert detail_data["id"] == doc_id
    assert len(detail_data["versions"]) == 1
    assert detail_data["versions"][0]["metadata"]["department"] == "Engineering"
    assert detail_data["versions"][0]["parser_name"] == "dummy_api_parser"

    # GET /api/v1/documents/{id}/versions/{version_id}/elements
    elements_response = client.get(f"/api/v1/documents/{doc_id}/versions/{ver_id}/elements")
    assert elements_response.status_code == 200
    elements_data = elements_response.json()
    assert len(elements_data) == 2
    assert elements_data[0]["text_content"] == "Remote Work Guidelines"

    # GET /api/v1/documents/{id}/presigned-url
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


def _upload(
    client: TestClient, content: bytes, key: str | None = None, name: str = "policy.pdf"
) -> Any:
    headers = {"Idempotency-Key": key} if key else {}
    return client.post(
        "/api/v1/documents",
        files={"file": (name, content, "application/pdf")},
        data={"metadata": METADATA_JSON},
        headers=headers,
    )


class TestIdempotentUpload:
    """Retrying an upload never starts a second ingestion (Task 7.4)."""

    def test_a_retry_with_the_same_key_replays_the_first_response(
        self, client: TestClient, override_api_dependencies: None, published: list[dict[str, Any]]
    ) -> None:
        first = _upload(client, b"%PDF-1.4 keyed", key="upload-1")
        retry = _upload(client, b"%PDF-1.4 keyed", key="upload-1")

        assert (first.status_code, retry.status_code) == (202, 202)
        assert retry.json() == first.json()
        assert retry.headers["Idempotent-Replayed"] == "true"
        assert "Idempotent-Replayed" not in first.headers
        assert len(published) == 1

    def test_without_a_key_a_retry_is_a_duplicate_with_no_new_job(
        self, client: TestClient, override_api_dependencies: None, published: list[dict[str, Any]]
    ) -> None:
        first = _upload(client, b"%PDF-1.4 unkeyed")
        retry = _upload(client, b"%PDF-1.4 unkeyed")

        assert (first.status_code, retry.status_code) == (202, 200)
        assert retry.json()["document_id"] == first.json()["document_id"]
        assert retry.json()["job_id"] is None
        assert len(published) == 1

    def test_the_same_key_on_a_different_file_is_refused(
        self, client: TestClient, override_api_dependencies: None, published: list[dict[str, Any]]
    ) -> None:
        _upload(client, b"%PDF-1.4 one", key="shared-key")
        other = _upload(client, b"%PDF-1.4 two", key="shared-key")

        assert other.status_code == 422
        assert "different request" in other.json()["detail"]
        assert len(published) == 1

    def test_an_overlong_key_is_refused(
        self, client: TestClient, override_api_dependencies: None
    ) -> None:
        assert _upload(client, b"%PDF-1.4 long", key="k" * 256).status_code == 422

    def test_a_failed_attempt_releases_its_key_and_the_retry_starts_the_chain(
        self,
        client: TestClient,
        override_api_dependencies: None,
        published: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def broker_down(**_: Any) -> None:
            raise ConnectionError("broker unreachable")

        monkeypatch.setattr(ingestion_tasks.parse_document, "apply_async", broker_down)
        with pytest.raises(ConnectionError):
            _upload(client, b"%PDF-1.4 outage", key="outage-1")

        monkeypatch.setattr(
            ingestion_tasks.parse_document,
            "apply_async",
            lambda *, kwargs, task_id: published.append(kwargs),
        )
        retry = _upload(client, b"%PDF-1.4 outage", key="outage-1")

        assert retry.status_code == 202, retry.text
        assert retry.json()["job_id"] == published[0]["job_id"]

    def test_a_stored_document_that_never_started_is_started_by_a_reupload(
        self,
        client: TestClient,
        override_api_dependencies: None,
        published: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The first request stored the document, then died before enqueueing.
        async def crash(*_: Any, **__: Any) -> None:
            raise RuntimeError("process killed")

        monkeypatch.setattr("app.api.v1.documents.start_ingestion", crash)
        with pytest.raises(RuntimeError):
            _upload(client, b"%PDF-1.4 orphan")
        monkeypatch.undo()
        monkeypatch.setattr(
            ingestion_tasks.parse_document,
            "apply_async",
            lambda *, kwargs, task_id: published.append(kwargs),
        )

        retry = _upload(client, b"%PDF-1.4 orphan")

        assert retry.status_code == 202, retry.text
        assert retry.json()["is_duplicate"] is True
        assert retry.json()["job_id"] == published[0]["job_id"]


class TestIdempotencyService:
    async def test_a_key_still_in_progress_is_a_conflict_until_completed(
        self, test_db_session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        service = IdempotencyService(test_db_session_factory)
        assert await service.claim("op", "k1", "fp") is None

        with pytest.raises(ConflictException):
            await service.claim("op", "k1", "fp")

        await service.complete("op", "k1", 202, {"job_id": "j"})
        stored = await service.claim("op", "k1", "fp")
        assert stored is not None
        assert (stored.status_code, stored.body) == (202, {"job_id": "j"})

    async def test_a_released_key_can_be_claimed_again(
        self, test_db_session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        service = IdempotencyService(test_db_session_factory)
        await service.claim("op", "k2", "fp")
        await service.release("op", "k2")

        assert await service.claim("op", "k2", "fp") is None

    async def test_keys_are_scoped_to_their_operation(
        self, test_db_session_factory: async_sessionmaker[AsyncSession]
    ) -> None:
        service = IdempotencyService(test_db_session_factory)
        await service.claim("documents.ingest", "same", "fp-a")

        assert await service.claim("documents.index_version", "same", "fp-b") is None
