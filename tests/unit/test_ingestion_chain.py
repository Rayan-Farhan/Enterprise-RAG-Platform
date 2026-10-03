"""The ingestion chain end to end, step replay, and failure behaviour (Task 7.3).

Each task runs through Celery's own `apply`, so the real wrapper executes: job
claim, body, success, and the hand-off that enqueues the next step. Publishing
is intercepted, which turns the broker into a list the test drains in order.
Object storage, Qdrant, OpenSearch and the embedding provider are in-memory
doubles; the database is SQLite through the real ORM.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, BinaryIO

import pytest
from celery import Task
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import AppSettings
from app.core.exceptions import ConflictException
from app.db.models.base import Base
from app.db.models.chunk import Chunk
from app.db.models.element import Element
from app.db.models.job import FailureKind, Job, JobStatus, JobType
from app.db.models.page import Page
from app.db.models.version import DocumentVersion, VersionStatus
from app.ingestion import pipeline
from app.ingestion.adapters.canonical_adapter import CanonicalAdapter
from app.ingestion.chunking.service import ChunkingService
from app.ingestion.dedup import BoilerplateDetector
from app.ingestion.parsers.base import (
    ElementType,
    ParsedDocument,
    ParsedElement,
    ParsedFigure,
    ParsedPage,
)
from app.ingestion.parsers.router import FormatRouter
from app.jobs.service import JobAlreadyActive, JobService
from app.models.schemas import EmbeddingResult, EmbeddingsResponse, ModelMetadata, TokenCounts
from app.retrieval.embedding import EmbeddingService
from app.retrieval.indexer import ChunkIndexer, LexicalIndexer, MissingVectorsError, SparseIndexer
from app.retrieval.schemas import ChunkPayload, RetrievalFilters
from app.retrieval.vector_store import VectorPoint
from app.workers import job_task as job_task_module
from app.workers.tasks import ingestion as ingestion_tasks

PDF = b"%PDF-1.4 chain test"
FIGURE_BYTES = b"\x89PNG\r\n\x1a\n\xff\xfe not utf-8"


class MemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(
        self,
        key: str,
        data: bytes | BinaryIO,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        self.objects[key] = data if isinstance(data, bytes) else data.read()
        return key

    def download_file(self, key: str) -> bytes:
        return self.objects[key]

    def get_presigned_url(self, key: str, expires_in_seconds: int = 3600) -> str:
        return f"memory://{key}"

    def delete_file(self, key: str) -> bool:
        return self.objects.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        return key in self.objects

    def ensure_bucket_exists(self) -> None:
        return None


class ChainParser:
    parser_name = "chain_parser"

    def __init__(self) -> None:
        self.calls = 0

    def parse(self, file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
        self.calls += 1
        body = (
            "Employees accrue annual leave monthly and may carry five days into the next "
            "year with written approval from their line manager. "
        )
        return ParsedDocument(
            filename="ignored.pdf",
            file_type="pdf",
            total_pages=3,
            parser_name=self.parser_name,
            parsing_duration_ms=12.0,
            pages=[
                ParsedPage(
                    page_number=1,
                    elements=[
                        ParsedElement(
                            element_id="h1",
                            element_type=ElementType.HEADING,
                            text="Leave Policy",
                            page_number=1,
                            level=1,
                        ),
                        ParsedElement(
                            element_id="p1",
                            element_type=ElementType.PARAGRAPH,
                            text=body * 4,
                            page_number=1,
                        ),
                    ],
                    figures=[
                        ParsedFigure(
                            figure_id="fig1",
                            caption="Accrual chart",
                            page_number=1,
                            image_bytes=FIGURE_BYTES,
                        )
                    ],
                ),
                ParsedPage(
                    page_number=2,
                    elements=[
                        ParsedElement(
                            element_id="p2",
                            element_type=ElementType.PARAGRAPH,
                            text=body * 3,
                            page_number=2,
                        )
                    ],
                ),
                ParsedPage(page_number=3),  # a scanned page: no text at all
            ],
        )


class CountingGateway:
    def __init__(self) -> None:
        self.embedded_texts = 0

    async def embed(self, texts: list[str], model_name: str | None = None) -> EmbeddingsResponse:
        self.embedded_texts += len(texts)
        return EmbeddingsResponse(
            embeddings=[
                EmbeddingResult(embedding=[float(len(t)), 1.0, 0.0, 0.5], index=i)
                for i, t in enumerate(texts)
            ],
            metadata=ModelMetadata(
                provider="fake",
                model_name="fake-embed",
                latency_ms=1.0,
                token_counts=TokenCounts(total_tokens=len(texts)),
            ),
        )


class MemoryVectorStore:
    def __init__(self) -> None:
        self.points: dict[str, VectorPoint] = {}
        self.drop_one = False

    def ensure_collection(self, dimensions: int | None = None) -> bool:
        return True

    def upsert(self, points: list[VectorPoint]) -> int:
        for point in points:
            self.points[point.point_id] = point
        return len(points)

    def count_matching(
        self,
        filters: RetrievalFilters | None = None,
        chunking_version: str | None = None,
        embedding_version: str | None = None,
    ) -> int:
        versions = {str(v) for v in filters.version_ids} if filters else set()
        n = sum(
            1
            for p in self.points.values()
            if (not versions or p.payload.version_id in versions)
            and (chunking_version is None or p.payload.chunking_version == chunking_version)
            and (embedding_version is None or p.payload.embedding_version == embedding_version)
        )
        return n - 1 if self.drop_one else n


class MemoryLexicalStore:
    def __init__(self) -> None:
        self.docs: dict[str, ChunkPayload] = {}

    def ensure_index(self) -> bool:
        return True

    def upsert(self, payloads: list[ChunkPayload], refresh: bool = True) -> int:
        for payload in payloads:
            self.docs[payload.chunk_id] = payload
        return len(payloads)

    def count(
        self, chunking_version: str | None = None, filters: RetrievalFilters | None = None
    ) -> int:
        versions = {str(v) for v in filters.version_ids} if filters else set()
        return sum(
            1
            for d in self.docs.values()
            if (not versions or d.version_id in versions)
            and (chunking_version is None or d.chunking_version == chunking_version)
        )


class Doubles:
    def __init__(self) -> None:
        self.settings = AppSettings(
            APP_ENV="testing",
            CHUNK_SIZE_TOKENS=40,
            CHUNK_OVERLAP_TOKENS=8,
            CHUNKING_STRATEGY="fixed",
            CHUNKING_VERSION="fixed-v1",
            EMBEDDING_VERSION="test-embed-v1",
            EMBEDDING_DIMENSIONS=4,
            EMBEDDING_BATCH_SIZE=4,
            EMBEDDING_MAX_RPM=10_000,
            ENABLE_LEXICAL_INDEXING=True,
            ENABLE_NEURAL_SPARSE=False,
        )
        self.storage = MemoryStorage()
        self.parser = ChainParser()
        self.gateway = CountingGateway()
        self.vectors = MemoryVectorStore()
        self.lexical = MemoryLexicalStore()

    def services(self) -> pipeline.PipelineServices:
        settings = self.settings
        return pipeline.PipelineServices(
            settings=settings,
            storage=self.storage,
            router=FormatRouter(pdf_primary=self.parser, office_parser=self.parser),
            boilerplate=BoilerplateDetector(),
            chunking=ChunkingService(settings=settings),
            chunk_indexer=ChunkIndexer(
                embedding_service=EmbeddingService(gateway=self.gateway, settings=settings),  # type: ignore[arg-type]
                vector_store=self.vectors,  # type: ignore[arg-type]
                settings=settings,
            ),
            lexical_indexer=LexicalIndexer(lexical_store=self.lexical, settings=settings),  # type: ignore[arg-type]
            sparse_indexer=SparseIndexer(sparse_store=self.lexical, settings=settings),  # type: ignore[arg-type]
        )


@pytest.fixture
async def engine(tmp_path: Path) -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'chain.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
def doubles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, engine: AsyncEngine) -> Doubles:
    url = f"sqlite+aiosqlite:///{tmp_path / 'chain.db'}"
    monkeypatch.setattr(job_task_module, "worker_engine_factory", lambda: create_async_engine(url))
    d = Doubles()
    monkeypatch.setattr(ingestion_tasks, "services_factory", d.services)
    return d


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Task, dict[str, Any]]]:
    """Published messages, in order, instead of RabbitMQ."""
    published: list[tuple[Task, dict[str, Any]]] = []
    for task in ingestion_tasks.CHAIN:

        def publish(*, kwargs: dict[str, Any], task_id: str, _task: Task = task) -> None:
            published.append((_task, kwargs))

        monkeypatch.setattr(task, "apply_async", publish)
    return published


async def drain(broker: list[tuple[Task, dict[str, Any]]]) -> list[str]:
    """Deliver every published message, including the ones each delivery publishes."""
    delivered: list[str] = []
    while broker:
        task, kwargs = broker.pop(0)
        # `apply` is synchronous and calls asyncio.run, so it gets its own thread
        # exactly as it would in a threads-pool worker.
        await asyncio.to_thread(task.apply, kwargs=kwargs, task_id=kwargs["job_id"])
        delivered.append(task.name)
    return delivered


async def accept(
    sessions: async_sessionmaker[AsyncSession], doubles: Doubles
) -> pipeline.AcceptedUpload:
    async with sessions() as session:
        accepted = await pipeline.accept_upload(
            session, doubles.storage, file_content=PDF, filename="leave_policy.pdf"
        )
        await session.commit()
    return accepted


async def ingest(
    sessions: async_sessionmaker[AsyncSession],
    doubles: Doubles,
    broker: list[tuple[Task, dict[str, Any]]],
) -> pipeline.AcceptedUpload:
    accepted = await accept(sessions, doubles)
    await ingestion_tasks.start_ingestion(
        accepted.document_id,
        accepted.version_id,
        {"department": "HR", "policy_type": "Leave"},
        jobs=JobService(sessions),
    )
    await drain(broker)
    return accepted


async def count(
    sessions: async_sessionmaker[AsyncSession], model: Any, version_id: uuid.UUID
) -> int:
    async with sessions() as session:
        return int(
            (
                await session.execute(
                    select(func.count()).select_from(model).where(model.version_id == version_id)
                )
            ).scalar_one()
        )


async def version_status(sessions: async_sessionmaker[AsyncSession], version_id: uuid.UUID) -> str:
    async with sessions() as session:
        version = await session.get(DocumentVersion, version_id)
        assert version is not None
        return version.status


async def jobs_of(sessions: async_sessionmaker[AsyncSession], version_id: uuid.UUID) -> list[Job]:
    async with sessions() as session:
        result = await session.execute(
            select(Job).where(Job.version_id == version_id).order_by(Job.created_at, Job.id)
        )
        return list(result.scalars().all())


CHAIN_ORDER = [
    JobType.PARSE_DOCUMENT,
    JobType.EXTRACT_PAGES,
    JobType.OCR_PAGES,
    JobType.NORMALIZE_DOCUMENT,
    JobType.CHUNK_DOCUMENT,
    JobType.GENERATE_EMBEDDINGS,
    JobType.INDEX_OPENSEARCH,
    JobType.INDEX_QDRANT,
    JobType.VALIDATE_INDEX,
    JobType.PUBLISH_VERSION,
]


class TestAcceptance:
    async def test_an_upload_is_stored_with_a_draft_version_and_nothing_parsed(
        self, sessions: async_sessionmaker[AsyncSession], doubles: Doubles
    ) -> None:
        accepted = await accept(sessions, doubles)

        assert accepted.is_duplicate is False
        assert doubles.storage.objects[accepted.storage_key] == PDF
        assert await version_status(sessions, accepted.version_id) == VersionStatus.DRAFT
        assert doubles.parser.calls == 0

    async def test_the_same_file_again_returns_the_existing_document(
        self, sessions: async_sessionmaker[AsyncSession], doubles: Doubles
    ) -> None:
        first = await accept(sessions, doubles)
        second = await accept(sessions, doubles)

        assert second.is_duplicate is True
        assert (second.document_id, second.version_id) == (first.document_id, first.version_id)


class TestFullChain:
    async def test_a_document_runs_every_step_in_order_and_is_published(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)

        jobs = await jobs_of(sessions, accepted.version_id)
        assert [j.task_type for j in jobs] == CHAIN_ORDER
        assert all(j.status == JobStatus.SUCCEEDED for j in jobs), [
            (j.task_type, j.error) for j in jobs
        ]
        assert all(j.document_id == accepted.document_id for j in jobs)
        assert await version_status(sessions, accepted.version_id) == VersionStatus.ACTIVE

    async def test_each_steps_output_persists_on_its_own(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)
        vid = accepted.version_id
        objects = doubles.storage.objects

        parsed = ParsedDocument.model_validate_json(objects[pipeline.parsed_key(vid)])
        assert parsed.all_figures[0].image_bytes == FIGURE_BYTES  # bytes survive JSON
        assert objects[pipeline.figure_key(vid, "fig1", "png")] == FIGURE_BYTES
        assert pipeline.page_manifest_key(vid) in objects
        assert b'"pages_needing_ocr": [3]' in objects[pipeline.ocr_result_key(vid)]
        assert pipeline.embeddings_key(vid, "test-embed-v1") in objects

        assert await count(sessions, Page, vid) == 3
        assert await count(sessions, Element, vid) == 4  # heading, two paragraphs, figure
        chunks = await count(sessions, Chunk, vid)
        assert chunks > 1
        assert len(doubles.vectors.points) == chunks
        assert len(doubles.lexical.docs) == chunks

    async def test_the_figure_element_points_at_the_stored_image(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        await ingest(sessions, doubles, broker)

        async with sessions() as session:
            figure = (
                await session.execute(select(Element).where(Element.element_type == "figure"))
            ).scalar_one()
        assert figure.asset_storage_key in doubles.storage.objects

    async def test_upload_metadata_reaches_the_version_and_the_index(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        await ingest(sessions, doubles, broker)

        assert {p.payload.department for p in doubles.vectors.points.values()} == {"HR"}


class TestReplay:
    async def test_rerunning_every_step_does_no_work_and_duplicates_nothing(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)
        vid = accepted.version_id
        before = (
            await count(sessions, Page, vid),
            await count(sessions, Element, vid),
            await count(sessions, Chunk, vid),
            len(doubles.vectors.points),
            len(doubles.lexical.docs),
        )
        embedded, parsed = doubles.gateway.embedded_texts, doubles.parser.calls

        steps = [
            pipeline.parse_document,
            pipeline.extract_pages,
            pipeline.ocr_pages,
            pipeline.normalize_document,
            pipeline.chunk_document,
            pipeline.generate_embeddings,
            pipeline.index_qdrant,
            pipeline.validate_index,
            pipeline.publish_version,
        ]
        for step in steps:
            async with sessions() as session:
                outcome = await step(doubles.services(), session, vid)
            if step is not pipeline.validate_index:
                assert outcome.noop, f"{step.__name__}: {outcome.summary}"

        after = (
            await count(sessions, Page, vid),
            await count(sessions, Element, vid),
            await count(sessions, Chunk, vid),
            len(doubles.vectors.points),
            len(doubles.lexical.docs),
        )
        assert after == before
        assert (doubles.gateway.embedded_texts, doubles.parser.calls) == (embedded, parsed)

    async def test_redelivering_a_finished_step_enqueues_its_follow_up_once(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)
        first = (await jobs_of(sessions, accepted.version_id))[0]

        broker.append((ingestion_tasks.parse_document, {"job_id": str(first.id), "metadata": {}}))
        await drain(broker)

        assert len(await jobs_of(sessions, accepted.version_id)) == len(CHAIN_ORDER)

    async def test_a_lost_hand_off_is_repaired_by_the_redelivery(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        # The worker died after recording success but before publishing the
        # next step: the broker redelivers, and the redelivery publishes it.
        accepted = await accept(sessions, doubles)
        job = await ingestion_tasks.start_ingestion(
            accepted.document_id, accepted.version_id, {}, jobs=JobService(sessions)
        )
        await drain(broker[:1])
        broker.clear()
        assert [j.task_type for j in await jobs_of(sessions, accepted.version_id)] == CHAIN_ORDER[
            :2
        ]

        async with sessions() as session:
            await session.execute(
                Job.__table__.delete().where(Job.task_type == JobType.EXTRACT_PAGES)
            )
            await session.commit()
        broker.append((ingestion_tasks.parse_document, {"job_id": str(job.id), "metadata": {}}))
        await drain(broker)

        assert await version_status(sessions, accepted.version_id) == VersionStatus.ACTIVE

    async def test_embedding_again_reuses_persisted_vectors(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        # Qdrant lost the points and the chunks forgot them; the vectors persisted
        # by generate_embeddings are still valid for the unchanged content.
        accepted = await ingest(sessions, doubles, broker)
        async with sessions() as session:
            await session.execute(
                Chunk.__table__.update().values(embedding_id=None, embedding_version=None)
            )
            await session.commit()
        doubles.vectors.points.clear()
        embedded = doubles.gateway.embedded_texts

        async with sessions() as session:
            outcome = await pipeline.generate_embeddings(
                doubles.services(), session, accepted.version_id
            )
        async with sessions() as session:
            await pipeline.index_qdrant(doubles.services(), session, accepted.version_id)

        assert doubles.gateway.embedded_texts == embedded
        assert "0 chunks embedded" in outcome.summary
        assert len(doubles.vectors.points) == await count(sessions, Chunk, accepted.version_id)

    async def test_index_qdrant_never_calls_the_embedding_provider(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)
        async with sessions() as session:
            await session.execute(
                Chunk.__table__.update().values(embedding_id=None, embedding_version=None)
            )
            await session.commit()
        del doubles.storage.objects[pipeline.embeddings_key(accepted.version_id, "test-embed-v1")]
        embedded = doubles.gateway.embedded_texts

        async with sessions() as session:
            with pytest.raises(MissingVectorsError, match="run generate_embeddings first"):
                await pipeline.index_qdrant(doubles.services(), session, accepted.version_id)
        assert doubles.gateway.embedded_texts == embedded


class TestFailure:
    async def test_a_count_mismatch_fails_validation_and_the_version_stays_draft(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        doubles.vectors.drop_one = True

        accepted = await ingest(sessions, doubles, broker)

        jobs = await jobs_of(sessions, accepted.version_id)
        assert jobs[-1].task_type == JobType.VALIDATE_INDEX
        assert jobs[-1].status == JobStatus.FAILED
        assert "qdrant points" in (jobs[-1].error or "")
        assert await version_status(sessions, accepted.version_id) == VersionStatus.DRAFT

    async def test_a_parse_failure_stops_the_chain_at_the_first_step(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def broken(file_path: Path | str, mime_type: str | None = None) -> ParsedDocument:
            raise ValueError("corrupt PDF")

        monkeypatch.setattr(doubles.parser, "parse", broken)
        monkeypatch.setattr(
            FormatRouter, "route_and_parse", lambda self, path, mime=None: broken(path)
        )

        accepted = await ingest(sessions, doubles, broker)

        jobs = await jobs_of(sessions, accepted.version_id)
        assert [(j.task_type, j.status) for j in jobs] == [
            (JobType.PARSE_DOCUMENT, JobStatus.FAILED)
        ]
        assert "corrupt PDF" in (jobs[0].error or "")


class TestIdempotency:
    """Task 7.4: deliberately re-running every task duplicates nothing."""

    async def test_rerunning_every_task_as_a_new_job_duplicates_nothing(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)
        vid = accepted.version_id

        async def snapshot() -> tuple[int, ...]:
            return (
                await count(sessions, Page, vid),
                await count(sessions, Element, vid),
                await count(sessions, Chunk, vid),
                len(doubles.vectors.points),
                len(doubles.lexical.docs),
                doubles.gateway.embedded_texts,
                doubles.parser.calls,
            )

        before = await snapshot()
        # Every step enqueued afresh; each one also re-runs the rest of the
        # chain behind it, so later steps are replayed many times over.
        for task in ingestion_tasks.CHAIN:
            await job_task_module.enqueue(
                task,
                document_id=accepted.document_id,
                version_id=vid,
                jobs=JobService(sessions),
                metadata={"department": "HR", "policy_type": "Leave"},
            )
            await drain(broker)

        jobs = await jobs_of(sessions, vid)
        assert len(jobs) == len(CHAIN_ORDER) + sum(range(1, len(CHAIN_ORDER) + 1))
        assert all(j.status == JobStatus.SUCCEEDED for j in jobs), [
            (j.task_type, j.error) for j in jobs if j.status != JobStatus.SUCCEEDED
        ]
        assert await snapshot() == before

    async def test_a_step_already_queued_is_not_queued_twice(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await accept(sessions, doubles)
        jobs = JobService(sessions)
        first = await ingestion_tasks.start_ingestion(
            accepted.document_id, accepted.version_id, {}, jobs=jobs
        )

        with pytest.raises(JobAlreadyActive) as raised:
            await ingestion_tasks.start_ingestion(
                accepted.document_id, accepted.version_id, {}, jobs=jobs
            )

        assert raised.value.job.id == first.id
        assert len(broker) == 1

    async def test_a_finished_step_can_be_run_again(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)

        again = await ingestion_tasks.start_ingestion(
            accepted.document_id, accepted.version_id, {}, jobs=JobService(sessions)
        )

        assert again.status == JobStatus.QUEUED

    def test_adapting_the_same_output_twice_yields_the_same_identities(self) -> None:
        parsed = ChainParser().parse("ignored.pdf")
        version_id, document_id = uuid.uuid4(), uuid.uuid4()

        def adapt() -> tuple[list[uuid.UUID], list[uuid.UUID], uuid.UUID | None]:
            _, _, pages, elements, meta = CanonicalAdapter.to_canonical_models(
                parsed_doc=parsed,
                file_hash="h",
                storage_key="k",
                metadata_dict={"department": "HR"},
                document_id=document_id,
                version_id=version_id,
            )
            return [p.id for p in pages], [e.id for e in elements], meta.id if meta else None

        first, second = adapt(), adapt()

        assert first == second
        assert len(set(first[1])) == len(first[1])

    async def test_the_database_refuses_a_second_copy_of_a_page_or_element(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)
        async with sessions() as session:
            page = (
                (await session.execute(select(Page).where(Page.version_id == accepted.version_id)))
                .scalars()
                .first()
            )
            element = (
                (
                    await session.execute(
                        select(Element).where(Element.version_id == accepted.version_id)
                    )
                )
                .scalars()
                .first()
            )
        assert page is not None and element is not None

        for clone in (
            Page(
                version_id=page.version_id,
                page_number=page.page_number,
                content_hash="x",
            ),
            Element(
                version_id=element.version_id,
                page_id=element.page_id,
                page_number=element.page_number,
                element_id=element.element_id,
                element_type=element.element_type,
                sequence_index=999,
                text_content="copy",
                content_hash="x",
            ),
        ):
            async with sessions() as session:
                session.add(clone)
                with pytest.raises(IntegrityError):
                    await session.commit()


class FlakyGateway(CountingGateway):
    """Fails the first ``failures`` embedding calls the way a provider outage does."""

    def __init__(self, failures: int) -> None:
        super().__init__()
        self.failures = failures

    async def embed(self, texts: list[str], model_name: str | None = None) -> EmbeddingsResponse:
        if self.failures:
            self.failures -= 1
            raise ConnectionError("embedding provider unreachable")
        return await super().embed(texts, model_name)


class TestRetriesAndDeadLetters:
    """Task 7.5: transient failures retry, permanent ones dead-letter and replay."""

    async def test_a_transient_failure_is_retried_and_the_chain_completes(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        # The embedding service retries internally first; with that off, the
        # outage reaches the job, which is the layer under test.
        doubles.settings = doubles.settings.model_copy(update={"EMBEDDING_MAX_RETRIES": 0})
        doubles.gateway = FlakyGateway(failures=2)

        accepted = await ingest(sessions, doubles, broker)

        jobs = {j.task_type: j for j in await jobs_of(sessions, accepted.version_id)}
        embed = jobs[JobType.GENERATE_EMBEDDINGS]
        assert (embed.status, embed.attempt) == (JobStatus.SUCCEEDED, 3)
        assert len(jobs) == len(CHAIN_ORDER)  # the retries are the same job, not new ones
        assert await version_status(sessions, accepted.version_id) == VersionStatus.ACTIVE

    async def test_a_dead_letter_replays_after_the_fault_is_fixed(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        doubles.vectors.drop_one = True
        accepted = await ingest(sessions, doubles, broker)
        service = JobService(sessions)
        (dead,) = await service.dead_letters()
        assert (dead.task_type, dead.failure_kind) == (
            JobType.VALIDATE_INDEX,
            FailureKind.PERMANENT,
        )
        embedded, parsed = doubles.gateway.embedded_texts, doubles.parser.calls

        doubles.vectors.drop_one = False
        replayed = await job_task_module.replay(dead, service)
        await drain(broker)

        assert replayed.replay_of_id == dead.id
        assert await service.dead_letters() == []
        assert await version_status(sessions, accepted.version_id) == VersionStatus.ACTIVE
        assert (doubles.gateway.embedded_texts, doubles.parser.calls) == (embedded, parsed)
        with pytest.raises(ConflictException, match="already replayed"):
            await job_task_module.replay(dead, service)

    async def test_only_failed_jobs_replay(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)
        done = (await jobs_of(sessions, accepted.version_id))[0]

        with pytest.raises(ConflictException, match="only failed jobs replay"):
            await job_task_module.replay(done, JobService(sessions))


class TestResume:
    async def test_resume_starts_at_the_first_unfinished_step_and_reparses_nothing(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        doubles.vectors.drop_one = True
        accepted = await ingest(sessions, doubles, broker)
        parsed = doubles.parser.calls
        doubles.vectors.drop_one = False
        service = JobService(sessions)

        resumed = await ingestion_tasks.resume_ingestion(accepted.version_id, service)
        await drain(broker)

        assert resumed.task_type == JobType.VALIDATE_INDEX
        assert resumed.payload == {"metadata": {"department": "HR", "policy_type": "Leave"}}
        assert await service.dead_letters() == []  # the resume picked up the dead letter
        assert doubles.parser.calls == parsed
        assert await version_status(sessions, accepted.version_id) == VersionStatus.ACTIVE

    async def test_a_fully_ingested_version_has_nothing_to_resume(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        accepted = await ingest(sessions, doubles, broker)

        with pytest.raises(ConflictException, match="already succeeded"):
            await ingestion_tasks.resume_ingestion(accepted.version_id, JobService(sessions))

    async def test_a_worker_killed_mid_step_resumes_there_with_no_duplicates(
        self,
        sessions: async_sessionmaker[AsyncSession],
        doubles: Doubles,
        broker: list[tuple[Task, dict[str, Any]]],
    ) -> None:
        # Run the chain up to generate_embeddings, then "kill" that worker: the
        # job is left running and its message goes back to the queue.
        accepted = await accept(sessions, doubles)
        service = JobService(sessions)
        await ingestion_tasks.start_ingestion(
            accepted.document_id, accepted.version_id, {}, jobs=service
        )
        while broker[0][0] is not ingestion_tasks.generate_embeddings:
            await drain(broker[:1])
            broker.pop(0)
        task, kwargs = broker[0]
        await service.start(uuid.UUID(kwargs["job_id"]), worker="killed@host")
        parsed = doubles.parser.calls
        with pytest.raises(ConflictException, match="already running"):
            await ingestion_tasks.resume_ingestion(accepted.version_id, service)

        await drain(broker)  # the redelivery, on a restarted worker

        jobs = await jobs_of(sessions, accepted.version_id)
        embed = next(j for j in jobs if j.task_type == JobType.GENERATE_EMBEDDINGS)
        assert (embed.status, embed.attempt) == (JobStatus.SUCCEEDED, 2)
        assert [j.task_type for j in jobs] == CHAIN_ORDER
        assert doubles.parser.calls == parsed
        chunks = await count(sessions, Chunk, accepted.version_id)
        assert len(doubles.vectors.points) == len(doubles.lexical.docs) == chunks
        assert await version_status(sessions, accepted.version_id) == VersionStatus.ACTIVE
