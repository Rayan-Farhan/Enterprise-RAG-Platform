"""Integration tests for the Stage 3 chunk -> index -> retrieve -> answer path.

These cross the ORM boundary against a real database and use in-memory doubles
only for the two genuinely external services (the embedding provider and Qdrant).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings
from app.db.models.chunk import Chunk
from app.db.models.version import DocumentVersion
from app.db.repositories.chunk_repo import ChunkRepository
from app.generation.citation import SupportState
from app.generation.context import ContextAssembler
from app.generation.service import GenerationService
from app.ingestion.chunking.base import compute_chunk_id, compute_point_id
from app.ingestion.chunking.service import ChunkingService, build_strategy
from app.models.schemas import (
    EmbeddingResult,
    EmbeddingsResponse,
    GenerationResult,
    ModelMetadata,
    TokenCounts,
)
from app.retrieval.dense import DenseRetriever
from app.retrieval.embedding import EmbeddingService
from app.retrieval.indexer import ChunkIndexer
from app.retrieval.schemas import RetrievalFilters
from app.retrieval.vector_store import QdrantVectorStore
from tests.unit.test_vector_store import FakeQdrantClient


class HashEmbeddingGateway:
    """Deterministic pseudo-embeddings derived from token overlap.

    Real semantics are not needed; what the pipeline tests require is that the
    same text always yields the same vector and that lexically similar texts land
    closer together than unrelated ones.
    """

    DIMENSIONS = 16

    def __init__(self) -> None:
        self.embed_calls = 0
        self.generate_calls: list[dict[str, object]] = []
        self.answer_text = "Employees receive 21 days of annual leave [1].\n\nSUPPORT: grounded"

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.DIMENSIONS
        for token in text.lower().split():
            vector[hash(token) % self.DIMENSIONS] += 1.0
        norm = sum(v * v for v in vector) ** 0.5 or 1.0
        return [v / norm for v in vector]

    async def embed(self, texts: list[str], model_name: str | None = None) -> EmbeddingsResponse:
        self.embed_calls += 1
        return EmbeddingsResponse(
            embeddings=[
                EmbeddingResult(embedding=self._vector(text), index=i)
                for i, text in enumerate(texts)
            ],
            metadata=ModelMetadata(
                provider="fake",
                model_name="hash-embed",
                latency_ms=2.0,
                token_counts=TokenCounts(total_tokens=sum(len(t.split()) for t in texts)),
            ),
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        self.generate_calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "prompt_version": prompt_version,
            }
        )
        return GenerationResult(
            text=self.answer_text,
            metadata=ModelMetadata(
                provider="fake",
                model_name="fake-llm",
                model_version="v1",
                prompt_version=prompt_version,
                latency_ms=25.0,
                token_counts=TokenCounts(prompt_tokens=200, completion_tokens=30, total_tokens=230),
            ),
        )

    async def rerank(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
        raise NotImplementedError

    async def vision(self, *args: object, **kwargs: object) -> object:  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def gateway() -> HashEmbeddingGateway:
    return HashEmbeddingGateway()


@pytest.fixture
def vector_store(settings: AppSettings) -> QdrantVectorStore:
    return QdrantVectorStore(client=FakeQdrantClient(), settings=settings)


@pytest.fixture
def chunking_service(settings: AppSettings) -> ChunkingService:
    return ChunkingService(strategy=build_strategy(settings), settings=settings)


@pytest.fixture
def indexer(
    settings: AppSettings,
    gateway: HashEmbeddingGateway,
    vector_store: QdrantVectorStore,
) -> ChunkIndexer:
    return ChunkIndexer(
        embedding_service=EmbeddingService(gateway=gateway, settings=settings),  # type: ignore[arg-type]
        vector_store=vector_store,
        settings=settings,
    )


@pytest.fixture
def retriever(
    settings: AppSettings,
    gateway: HashEmbeddingGateway,
    vector_store: QdrantVectorStore,
) -> DenseRetriever:
    return DenseRetriever(
        embedding_service=EmbeddingService(gateway=gateway, settings=settings),  # type: ignore[arg-type]
        vector_store=vector_store,
        settings=settings,
    )


class TestChunkingPersistence:
    async def test_chunks_persist_with_full_ancestry(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
    ) -> None:
        result = await chunking_service.chunk_version(session, hr_document.id)
        await session.commit()

        assert result.total_chunks > 0
        assert result.chunks_created == result.total_chunks

        chunks = await ChunkRepository(session).list_by_version(hr_document.id)
        assert len(chunks) == result.total_chunks

        for chunk in chunks:
            assert chunk.document_id == hr_document.document_id
            assert chunk.version_id == hr_document.id
            assert chunk.element_ids
            assert chunk.section_path
            assert chunk.primary_page_number >= 1
            assert chunk.page_span
            assert chunk.token_count > 0
            assert chunk.chunking_version == "fixed-v1"

    async def test_boilerplate_is_excluded_from_chunks(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        await session.commit()

        chunks = await ChunkRepository(session).list_by_version(hr_document.id)
        assert all("Confidential" not in c.content for c in chunks)
        assert all("f1" not in c.element_ids for c in chunks)

    async def test_section_paths_reflect_heading_hierarchy(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        await session.commit()

        chunks = await ChunkRepository(session).list_by_version(hr_document.id)
        paths = {tuple(c.section_path) for c in chunks}
        assert ("Leave Policy", "Annual Leave") in paths
        assert ("Leave Policy", "Sick Leave") in paths

    async def test_rechunking_is_idempotent_and_ids_are_stable(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
    ) -> None:
        first = await chunking_service.chunk_version(session, hr_document.id)
        await session.commit()
        first_ids = [c.id for c in await ChunkRepository(session).list_by_version(hr_document.id)]

        second = await chunking_service.chunk_version(session, hr_document.id)
        await session.commit()
        second_ids = [c.id for c in await ChunkRepository(session).list_by_version(hr_document.id)]

        assert second.chunks_created == 0
        assert second.chunks_removed == 0
        assert second.is_noop
        assert first_ids == second_ids
        assert len(first_ids) == first.total_chunks

    async def test_chunk_ids_match_the_deterministic_formula(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        await session.commit()

        for chunk in await ChunkRepository(session).list_by_version(hr_document.id):
            assert chunk.id == compute_chunk_id(
                version_id=hr_document.id,
                element_ids=list(chunk.element_ids),
                chunk_index=chunk.chunk_index,
                chunking_version="fixed-v1",
            )

    async def test_bumping_chunking_version_creates_a_parallel_set(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        settings: AppSettings,
    ) -> None:
        v1 = ChunkingService(strategy=build_strategy(settings), settings=settings)
        await v1.chunk_version(session, hr_document.id)
        await session.commit()

        bumped = settings.model_copy(update={"CHUNKING_VERSION": "fixed-v2"})
        v2 = ChunkingService(strategy=build_strategy(bumped), settings=bumped)
        result = await v2.chunk_version(session, hr_document.id)
        await session.commit()

        # Losing the old chunks would break Stage 5's strategy comparison.
        assert result.chunks_created > 0
        assert result.chunks_removed == 0

        all_chunks = (await session.execute(select(Chunk))).scalars().all()
        assert {c.chunking_version for c in all_chunks} == {"fixed-v1", "fixed-v2"}

    async def test_missing_version_raises_not_found(
        self, session: AsyncSession, chunking_service: ChunkingService
    ) -> None:
        from app.core.exceptions import NotFoundException

        with pytest.raises(NotFoundException):
            await chunking_service.chunk_version(session, uuid.uuid4())


class TestIndexing:
    async def test_full_corpus_indexes(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
        indexer: ChunkIndexer,
        vector_store: QdrantVectorStore,
    ) -> None:
        chunking = await chunking_service.chunk_version(session, hr_document.id)
        result = await indexer.index_version(session, hr_document.id)
        await session.commit()

        assert result.chunks_embedded == chunking.total_chunks
        assert result.points_upserted == chunking.total_chunks
        assert vector_store.count() == chunking.total_chunks
        assert result.dimensions == HashEmbeddingGateway.DIMENSIONS

    async def test_reindexing_creates_zero_duplicate_points(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
        indexer: ChunkIndexer,
        vector_store: QdrantVectorStore,
        gateway: HashEmbeddingGateway,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        first = await indexer.index_version(session, hr_document.id)
        await session.commit()

        points_after_first = vector_store.count()
        calls_after_first = gateway.embed_calls

        second = await indexer.index_version(session, hr_document.id)
        await session.commit()

        assert second.chunks_embedded == 0
        assert second.is_noop
        assert second.chunks_skipped == first.chunks_embedded
        assert vector_store.count() == points_after_first
        # Nothing pending means no provider calls at all — not just no new points.
        assert gateway.embed_calls == calls_after_first

    async def test_force_reindex_reembeds_without_duplicating(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
        indexer: ChunkIndexer,
        vector_store: QdrantVectorStore,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        first = await indexer.index_version(session, hr_document.id)
        await session.commit()

        forced = await indexer.index_version(session, hr_document.id, force=True)
        await session.commit()

        assert forced.chunks_embedded == first.chunks_embedded
        assert vector_store.count() == first.points_upserted

    async def test_point_ids_are_deterministic(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
        indexer: ChunkIndexer,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        await indexer.index_version(session, hr_document.id)
        await session.commit()

        for chunk in await ChunkRepository(session).list_by_version(hr_document.id):
            assert chunk.embedding_id == compute_point_id(chunk.id, "test-embed-v1")
            assert chunk.embedding_version == "test-embed-v1"

    async def test_changing_embedding_version_marks_chunks_unindexed(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
        indexer: ChunkIndexer,
        settings: AppSettings,
        gateway: HashEmbeddingGateway,
        vector_store: QdrantVectorStore,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        first = await indexer.index_version(session, hr_document.id)
        await session.commit()

        bumped = settings.model_copy(update={"EMBEDDING_VERSION": "test-embed-v2"})
        new_indexer = ChunkIndexer(
            embedding_service=EmbeddingService(gateway=gateway, settings=bumped),  # type: ignore[arg-type]
            vector_store=vector_store,
            settings=bumped,
        )
        result = await new_indexer.index_version(session, hr_document.id)
        await session.commit()

        assert result.chunks_embedded == first.chunks_embedded
        assert result.embedding_version == "test-embed-v2"
        # New embedding version writes new points rather than overwriting old ones.
        assert vector_store.count() == first.points_upserted * 2

    async def test_payload_carries_metadata_and_acl_placeholders(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
        indexer: ChunkIndexer,
        vector_store: QdrantVectorStore,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        await indexer.index_version(session, hr_document.id)
        await session.commit()

        payloads = [p["payload"] for p in vector_store.client.points.values()]  # type: ignore[union-attr]
        assert payloads
        for payload in payloads:
            assert payload["department"] == "Human Resources"
            assert payload["policy_type"] == "leave"
            assert payload["document_title"] == "Staff Handbook 2026"
            assert payload["page_number"] >= 1
            assert payload["section_path"]
            # Stage 8 fields present from day one.
            assert "allowed_roles" in payload
            assert "tenant_id" in payload
            assert payload["classification"] == "internal"


class TestDenseRetrieval:
    @pytest.fixture(autouse=True)
    async def _indexed(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
        indexer: ChunkIndexer,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        await indexer.index_version(session, hr_document.id)
        await session.commit()

    async def test_query_returns_scored_chunks_with_provenance(
        self, session: AsyncSession, retriever: DenseRetriever
    ) -> None:
        result = await retriever.retrieve(
            query="How many annual leave days do employees get?", session=session
        )

        assert not result.is_empty
        for chunk in result.chunks:
            assert chunk.score > 0
            assert chunk.rank >= 1
            assert chunk.page_number >= 1
            assert chunk.element_ids
            assert chunk.document_title == "Staff Handbook 2026"
            assert chunk.version_number == 1
            assert chunk.channel == "dense"

    async def test_hydrated_content_matches_postgres(
        self, session: AsyncSession, retriever: DenseRetriever
    ) -> None:
        result = await retriever.retrieve(query="annual leave", session=session)
        stored = {
            c.id: c.content for c in await ChunkRepository(session).list_by_version(
                result.chunks[0].version_id
            )
        }
        for chunk in result.chunks:
            assert chunk.content == stored[chunk.chunk_id]

    async def test_ranks_are_contiguous_from_one(
        self, session: AsyncSession, retriever: DenseRetriever
    ) -> None:
        result = await retriever.retrieve(query="sick leave certificate", session=session)
        assert [c.rank for c in result.chunks] == list(range(1, len(result.chunks) + 1))

    async def test_retrieval_config_is_recorded(
        self, session: AsyncSession, retriever: DenseRetriever
    ) -> None:
        result = await retriever.retrieve(query="annual leave", session=session, top_k=3)
        config = result.retrieval_config

        assert config["channels"] == ["dense"]
        assert config["top_k"] == 3
        assert config["chunking_version"] == "fixed-v1"
        assert config["embedding_version"] == "test-embed-v1"
        assert config["reranking_enabled"] is False

    async def test_top_k_limits_results(
        self, session: AsyncSession, retriever: DenseRetriever
    ) -> None:
        result = await retriever.retrieve(query="leave", session=session, top_k=1)
        assert len(result.chunks) <= 1

    async def test_metadata_filter_is_pushed_into_the_query(
        self, session: AsyncSession, retriever: DenseRetriever, vector_store: QdrantVectorStore
    ) -> None:
        await retriever.retrieve(
            query="annual leave",
            session=session,
            filters=RetrievalFilters(department="Human Resources"),
        )
        query = vector_store.client.last_query  # type: ignore[union-attr]
        assert query is not None and query["query_filter"] is not None


class TestEndToEndAnswer:
    @pytest.fixture(autouse=True)
    async def _indexed(
        self,
        session: AsyncSession,
        hr_document: DocumentVersion,
        chunking_service: ChunkingService,
        indexer: ChunkIndexer,
    ) -> None:
        await chunking_service.chunk_version(session, hr_document.id)
        await indexer.index_version(session, hr_document.id)
        await session.commit()

    @pytest.fixture
    def service(
        self,
        settings: AppSettings,
        retriever: DenseRetriever,
        gateway: HashEmbeddingGateway,
    ) -> GenerationService:
        return GenerationService(
            retriever=retriever,
            assembler=ContextAssembler(settings),
            gateway=gateway,  # type: ignore[arg-type]
            settings=settings,
        )

    async def test_question_yields_grounded_cited_answer(
        self, session: AsyncSession, service: GenerationService
    ) -> None:
        result = await service.answer(
            query="How many annual leave days do employees receive?", session=session
        )

        assert result.support is SupportState.GROUNDED
        assert not result.abstained
        assert not result.rejected
        assert result.citations

    async def test_citations_resolve_to_real_elements(
        self, session: AsyncSession, service: GenerationService
    ) -> None:
        from app.db.repositories.document_repo import DocumentRepository

        result = await service.answer(query="annual leave entitlement", session=session)
        repo = DocumentRepository(session)

        for citation in result.citations:
            elements = await repo.get_elements_by_element_ids(
                version_id=citation.version_id,
                element_ids=citation.element_ids,
            )
            assert len(elements) == len(citation.element_ids)
            assert citation.page_number in {e.page_number for e in elements}
            assert citation.bounding_box is not None

    async def test_answer_records_full_reproducibility_metadata(
        self, session: AsyncSession, service: GenerationService
    ) -> None:
        result = await service.answer(query="annual leave", session=session)

        assert result.model_name == "fake-llm"
        assert result.model_version == "v1"
        assert result.prompt_versions["answer"] == "answer_v1"
        assert result.prompt_versions["citation"] == "citation_v1"
        assert len(result.prompt_hashes["answer"]) == 64
        assert result.retrieval_config["embedding_version"] == "test-embed-v1"
        assert result.retrieved_chunk_ids
        assert result.token_counts["total_tokens"] == 230
        assert result.total_latency_ms > 0

    async def test_retrieved_chunk_ids_are_real_chunks(
        self, session: AsyncSession, service: GenerationService
    ) -> None:
        result = await service.answer(query="sick leave", session=session)
        found = await ChunkRepository(session).get_many_by_ids(result.retrieved_chunk_ids)
        assert len(found) == len(result.retrieved_chunk_ids)

    async def test_out_of_corpus_question_abstains(
        self, session: AsyncSession, service: GenerationService, gateway: HashEmbeddingGateway
    ) -> None:
        """The exit-gate scenario: a question the corpus cannot answer."""
        # Force the retrieval floor above any achievable score so nothing passes.
        service.retriever.settings = service.retriever.settings.model_copy(
            update={"RETRIEVAL_MIN_SCORE": 0.999999}
        )
        gateway.answer_text = (
            "The HR knowledge base does not contain information about housing.\n\n"
            "SUPPORT: insufficient"
        )

        result = await service.answer(
            query="Does the company provide housing to employees?", session=session
        )

        assert result.abstained
        assert result.support is SupportState.INSUFFICIENT
        assert result.citations == []
        assert result.prompt_versions == {"abstention": "abstention_v1"}

    async def test_fabricated_citation_is_stripped(
        self, session: AsyncSession, service: GenerationService, gateway: HashEmbeddingGateway
    ) -> None:
        gateway.answer_text = (
            "Annual leave is 21 days [1]. Housing is also provided [99].\n\nSUPPORT: grounded"
        )
        result = await service.answer(query="annual leave", session=session)

        assert result.fabricated_markers == ["99"]
        assert "[99]" not in result.answer
        assert all(c.marker != "99" for c in result.citations)

    async def test_uncitable_answer_is_rejected_and_replaced(
        self, session: AsyncSession, service: GenerationService, gateway: HashEmbeddingGateway
    ) -> None:
        gateway.answer_text = (
            "The company provides free housing to all employees [42].\n\nSUPPORT: grounded"
        )
        result = await service.answer(query="housing policy", session=session)

        assert result.rejected
        assert result.abstained
        assert result.citations == []
        assert "free housing" not in result.answer
        assert "answer_rejected_citation_validation" in result.degradations

    async def test_evidence_is_untrusted_in_the_prompt_sent_to_the_model(
        self, session: AsyncSession, service: GenerationService, gateway: HashEmbeddingGateway
    ) -> None:
        await service.answer(query="annual leave", session=session)
        call = gateway.generate_calls[-1]

        prompt = str(call["prompt"])
        system_prompt = str(call["system_prompt"])

        assert "RETRIEVED EVIDENCE (UNTRUSTED DATA" in prompt
        assert "--- BEGIN EVIDENCE [1] ---" in prompt
        # Document text never reaches the trusted system role. The system prompt
        # does describe the fence *format*, so the check is on actual corpus
        # content, not on the marker syntax.
        assert "21 days of paid annual leave" in prompt
        assert "21 days of paid annual leave" not in system_prompt
        assert "Staff Handbook 2026" not in system_prompt
        assert call["prompt_version"] == "answer_v1"

    async def test_streaming_emits_ordered_events(
        self, session: AsyncSession, service: GenerationService
    ) -> None:
        events = [
            (name, payload)
            async for name, payload in service.stream_answer(
                query="annual leave", session=session
            )
        ]

        assert [name for name, _ in events] == ["metadata", "token", "citations", "done"]

        payloads = dict(events)
        assert payloads["token"]["text"]
        assert payloads["citations"]["citations"]
        assert payloads["done"]["citation_count"] == len(payloads["citations"]["citations"])

    async def test_streaming_and_nonstreaming_agree(
        self, session: AsyncSession, service: GenerationService
    ) -> None:
        direct = await service.answer(query="annual leave", session=session)
        streamed = {
            name: payload
            async for name, payload in service.stream_answer(
                query="annual leave", session=session
            )
        }

        assert streamed["token"]["text"] == direct.answer
        assert streamed["done"]["support"] == str(direct.support)
        assert len(streamed["citations"]["citations"]) == len(direct.citations)
