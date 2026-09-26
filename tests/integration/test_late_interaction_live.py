"""Late interaction against a real Qdrant multivector collection (Task 6.6).

Chunks come from the SQLite pipeline fixtures and token vectors from the stub
gateway, so no model is called; what is real is Qdrant's MAX_SIM scoring over
multivectors and the indexer's skip-existing behaviour. Each test gets its own
collection. Skips when Qdrant is unreachable, unless RAG_REQUIRE_SERVICES=1.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings
from app.db.models.version import DocumentVersion
from app.db.repositories.chunk_repo import ChunkRepository
from app.ingestion.chunking.service import ChunkingService, build_strategy
from app.models.gateway import StubModelGateway
from app.retrieval.late_interaction import LateInteractionIndexer, LateInteractionStore


@pytest.fixture
def li_settings(settings: AppSettings) -> AppSettings:
    return settings.model_copy(
        update={
            "INFERENCE_PROFILE": "stub",
            "LATE_INTERACTION_COLLECTION": f"test_colbert_{uuid.uuid4().hex[:12]}",
        }
    )


@pytest.fixture
def store(li_settings: AppSettings) -> Iterator[LateInteractionStore]:
    candidate = LateInteractionStore(settings=li_settings)
    try:
        candidate.client.get_collections()
    except Exception as exc:  # noqa: BLE001 - any connection failure means unavailable
        if os.getenv("RAG_REQUIRE_SERVICES") == "1":
            pytest.fail(f"Qdrant is required (RAG_REQUIRE_SERVICES=1) but not reachable: {exc}")
        pytest.skip("Qdrant not reachable; start it with `make up`")
    yield candidate
    candidate.client.delete_collection(candidate.collection_name)


@pytest.fixture
async def chunked(
    session: AsyncSession, hr_document: DocumentVersion, li_settings: AppSettings
) -> DocumentVersion:
    service = ChunkingService(strategy=build_strategy(li_settings), settings=li_settings)
    await service.chunk_version(session, hr_document.id)
    await session.commit()
    return hr_document


class TestLateInteractionLive:
    async def test_the_collection_is_a_maxsim_multivector_collection(
        self, store: LateInteractionStore
    ) -> None:
        store.ensure_collection()
        config = store.client.get_collection(store.collection_name).config.params.vectors

        assert config.size == 128  # type: ignore[union-attr]
        assert config.multivector_config.comparator.value == "max_sim"  # type: ignore[union-attr]

    async def test_indexing_is_idempotent_and_skips_encoded_chunks(
        self,
        session: AsyncSession,
        chunked: DocumentVersion,
        li_settings: AppSettings,
        store: LateInteractionStore,
    ) -> None:
        indexer = LateInteractionIndexer(
            gateway=StubModelGateway(li_settings), store=store, settings=li_settings
        )

        first = await indexer.index_version(session, chunked.id)
        second = await indexer.index_version(session, chunked.id)

        assert first.encoded > 0 and first.skipped == 0
        assert (second.encoded, second.skipped) == (0, first.encoded)
        assert store.client.count(store.collection_name).count == first.encoded

    async def test_maxsim_ranks_the_chunk_sharing_the_querys_tokens_first(
        self,
        session: AsyncSession,
        chunked: DocumentVersion,
        li_settings: AppSettings,
        store: LateInteractionStore,
    ) -> None:
        gateway = StubModelGateway(li_settings)
        await LateInteractionIndexer(
            gateway=gateway, store=store, settings=li_settings
        ).index_version(session, chunked.id)
        chunks = await ChunkRepository(session).list_by_version(
            chunked.id, li_settings.CHUNKING_VERSION
        )
        target = chunks[-1]
        # The target chunk's own words as the query: MaxSim must find it.
        query = " ".join(target.content.split()[:12])

        encoded = await gateway.embed_multivector([query], input_type="query")
        scored = store.maxsim(
            encoded.embeddings[0].vectors, [store.point_id(c.id) for c in chunks], limit=3
        )

        assert scored[0][0] == store.point_id(target.id)
        assert scored[0][1] > scored[-1][1]
