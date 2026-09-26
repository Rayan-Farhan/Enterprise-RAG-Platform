"""PostgreSQL -> BM25 index -> PostgreSQL round trip (Task 6.1), with no OpenSearch.

The store is a recording double, so this runs anywhere; the analyzers themselves
are covered against a real OpenSearch in ``test_lexical_store_live.py``.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings
from app.db.models.version import DocumentVersion
from app.db.repositories.chunk_repo import ChunkRepository
from app.ingestion.chunking.service import ChunkingService, build_strategy
from app.retrieval.indexer import LexicalIndexer
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.lexical_store import LexicalHit, to_document
from app.retrieval.schemas import ChunkPayload


class RecordingStore:
    """Keeps what was indexed and serves it back in a caller-chosen order."""

    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.ensure_calls = 0

    def ensure_index(self) -> bool:
        self.ensure_calls += 1
        return self.ensure_calls == 1

    def upsert(self, payloads: list[ChunkPayload], refresh: bool = True) -> int:
        for item in payloads:
            self.documents[item.chunk_id] = to_document(item)
        return len(payloads)

    def search(self, **kwargs: Any) -> list[LexicalHit]:
        ranked = sorted(self.documents.values(), key=lambda d: d["chunk_index"], reverse=True)
        return [
            LexicalHit(chunk_id=d["chunk_id"], score=10.0 - rank, payload=d)
            for rank, d in enumerate(ranked[: kwargs["limit"]])
        ]


@pytest.fixture
async def chunked(
    session: AsyncSession, hr_document: DocumentVersion, settings: AppSettings
) -> DocumentVersion:
    service = ChunkingService(strategy=build_strategy(settings), settings=settings)
    await service.chunk_version(session, hr_document.id)
    await session.commit()
    return hr_document


class TestLexicalIndexer:
    async def test_every_chunk_of_the_version_is_indexed_once(
        self, session: AsyncSession, chunked: DocumentVersion, settings: AppSettings
    ) -> None:
        store = RecordingStore()
        indexer = LexicalIndexer(lexical_store=store, settings=settings)  # type: ignore[arg-type]

        first = await indexer.index_version(session, chunked.id)
        second = await indexer.index_version(session, chunked.id)

        chunks = await ChunkRepository(session).list_by_version(
            chunked.id, settings.CHUNKING_VERSION
        )
        assert first.documents_indexed == second.documents_indexed == len(chunks)
        # Keyed on the deterministic chunk ID: the re-run overwrote, it did not add.
        assert set(store.documents) == {str(c.id) for c in chunks}

    async def test_indexed_documents_carry_filter_and_acl_fields(
        self, session: AsyncSession, chunked: DocumentVersion, settings: AppSettings
    ) -> None:
        store = RecordingStore()
        await LexicalIndexer(lexical_store=store, settings=settings).index_version(  # type: ignore[arg-type]
            session, chunked.id
        )

        document = next(iter(store.documents.values()))
        assert document["chunking_version"] == settings.CHUNKING_VERSION
        assert document["document_title"] == "Staff Handbook 2026"
        assert document["classification"] == "internal"

    async def test_retrieval_rehydrates_bm25_hits_from_postgres_in_rank_order(
        self, session: AsyncSession, chunked: DocumentVersion, settings: AppSettings
    ) -> None:
        store = RecordingStore()
        await LexicalIndexer(lexical_store=store, settings=settings).index_version(  # type: ignore[arg-type]
            session, chunked.id
        )
        # Corrupt the index copy: rehydration must return PostgreSQL's text.
        for document in store.documents.values():
            document["content"] = "stale index copy"

        retriever = LexicalRetriever(lexical_store=store, settings=settings)  # type: ignore[arg-type]
        result = await retriever.retrieve("annual leave", session=session, top_k=10)

        assert len(result.chunks) == len(store.documents) >= 2
        assert [c.rank for c in result.chunks] == list(range(1, len(result.chunks) + 1))
        assert [c.chunk_index for c in result.chunks] == sorted(
            (c.chunk_index for c in result.chunks), reverse=True
        )
        assert all(c.content != "stale index copy" for c in result.chunks)
        assert all(c.element_ids for c in result.chunks)
        assert all(c.channel == "bm25" for c in result.chunks)

    async def test_an_unknown_version_is_a_not_found(
        self, session: AsyncSession, settings: AppSettings
    ) -> None:
        from app.core.exceptions import NotFoundException

        indexer = LexicalIndexer(lexical_store=RecordingStore(), settings=settings)  # type: ignore[arg-type]
        with pytest.raises(NotFoundException):
            await indexer.index_version(session, uuid.uuid4())


class RecordingSparseStore(RecordingStore):
    """Adds the skip-existing lookup the sparse indexer relies on."""

    def __init__(self) -> None:
        super().__init__()
        self.encoded: list[str] = []

    def existing_ids(self, chunk_ids: list[str]) -> set[str]:
        return {cid for cid in chunk_ids if cid in self.documents}

    def upsert(self, payloads: list[ChunkPayload], refresh: bool = True) -> int:
        self.encoded.extend(p.chunk_id for p in payloads)
        return super().upsert(payloads, refresh)


class TestSparseIndexer:
    """Encoding costs about a second per chunk, so re-runs must not repeat it."""

    async def test_a_rerun_encodes_nothing_already_present(
        self, session: AsyncSession, chunked: DocumentVersion, settings: AppSettings
    ) -> None:
        from app.retrieval.indexer import SparseIndexer

        store = RecordingSparseStore()
        indexer = SparseIndexer(sparse_store=store, settings=settings)  # type: ignore[arg-type]

        first = await indexer.index_version(session, chunked.id)
        second = await indexer.index_version(session, chunked.id)

        assert first.documents_encoded > 0 and first.documents_skipped == 0
        assert second.documents_encoded == 0
        assert second.documents_skipped == first.documents_encoded
        assert len(store.encoded) == first.documents_encoded

    async def test_force_re_encodes_everything(
        self, session: AsyncSession, chunked: DocumentVersion, settings: AppSettings
    ) -> None:
        from app.retrieval.indexer import SparseIndexer

        store = RecordingSparseStore()
        indexer = SparseIndexer(sparse_store=store, settings=settings)  # type: ignore[arg-type]

        first = await indexer.index_version(session, chunked.id)
        forced = await indexer.index_version(session, chunked.id, force=True)

        assert forced.documents_encoded == first.documents_encoded
        assert forced.documents_skipped == 0
