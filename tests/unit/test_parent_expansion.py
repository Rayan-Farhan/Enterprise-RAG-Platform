"""Budgeted parent-child expansion (Task 5.2, master §19).

Small chunks retrieve well and read badly. These tests pin the three properties
that make expansion safe: it stays inside a token budget, it preserves the
ordering the downstream context budget depends on, and it never loses evidence
retrieval had already found.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.core.config import AppSettings
from app.db.models.chunk import Chunk
from app.retrieval.expansion import ParentExpander
from app.retrieval.schemas import RetrievedChunk

DOCUMENT_ID = uuid.uuid4()
VERSION_ID = uuid.uuid4()


def parent_chunk(chunk_id: uuid.UUID, tokens: int = 400, title: str = "Sick Leave") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id=DOCUMENT_ID,
        version_id=VERSION_ID,
        chunk_index=0,
        chunk_type="section",
        chunking_version="hierarchical-v1",
        content=f"Full text of the {title} section. " * 5,
        token_count=tokens,
        element_ids=["e1", "e2", "e3"],
        section_path=["Time Off Work", title],
        primary_page_number=21,
        page_span=[21, 22],
    )


def leaf(
    parent_id: uuid.UUID | None,
    rank: int = 0,
    score: float = 0.9,
    tokens: int = 100,
    chunk_id: uuid.UUID | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=DOCUMENT_ID,
        version_id=VERSION_ID,
        content="Employees are entitled to 96 work hours of sick leave.",
        score=score,
        rank=rank,
        token_count=tokens,
        page_number=21,
        section_path=["Time Off Work", "Sick Leave"],
        element_ids=["e2"],
        metadata={"parent_chunk_id": str(parent_id) if parent_id else None},
    )


class FakeSession:
    """Returns the parents it was seeded with, and counts round trips."""

    def __init__(self, parents: list[Chunk]) -> None:
        self._parents = parents
        self.queries = 0

    async def execute(self, _stmt: Any) -> Any:
        self.queries += 1
        parents = self._parents

        class Result:
            @staticmethod
            def scalars() -> Any:
                class Scalars:
                    @staticmethod
                    def all() -> list[Chunk]:
                        return parents

                return Scalars()

        return Result()


def expander(**overrides: Any) -> ParentExpander:
    settings = AppSettings(
        ENABLE_PARENT_EXPANSION=overrides.pop("enabled", True),
        PARENT_EXPANSION_BUDGET_TOKENS=overrides.pop("budget", 6000),
        **overrides,
    )
    return ParentExpander(settings=settings)


class TestFeatureFlag:
    @pytest.mark.asyncio
    async def test_disabled_by_default_so_the_baseline_stays_reproducible(self) -> None:
        # experiment-001-baseline was measured without expansion; turning it on
        # by default would silently invalidate the number every stage cites.
        assert AppSettings().ENABLE_PARENT_EXPANSION is False

    @pytest.mark.asyncio
    async def test_disabled_returns_the_input_untouched(self) -> None:
        parent_id = uuid.uuid4()
        chunks = [leaf(parent_id)]
        session = FakeSession([parent_chunk(parent_id)])

        result = await expander(enabled=False).expand(chunks, session)  # type: ignore[arg-type]

        assert result.chunks == chunks
        assert result.was_noop
        assert session.queries == 0


class TestExpansion:
    @pytest.mark.asyncio
    async def test_a_matched_leaf_is_replaced_by_its_section(self) -> None:
        parent_id = uuid.uuid4()
        session = FakeSession([parent_chunk(parent_id)])

        result = await expander().expand([leaf(parent_id)], session)  # type: ignore[arg-type]

        assert result.expanded == 1
        assert len(result.chunks) == 1
        assert result.chunks[0].chunk_id == parent_id
        assert result.chunks[0].chunk_type == "section"

    @pytest.mark.asyncio
    async def test_the_parent_inherits_its_child_rank_and_score(self) -> None:
        # The parent was never scored. Inventing a score would corrupt the
        # ordering that the downstream context budget truncates against.
        parent_id = uuid.uuid4()
        session = FakeSession([parent_chunk(parent_id)])

        result = await expander().expand(
            [leaf(parent_id, rank=3, score=0.77)], session  # type: ignore[arg-type]
        )

        assert result.chunks[0].rank == 3
        assert result.chunks[0].score == pytest.approx(0.77)

    @pytest.mark.asyncio
    async def test_the_expansion_is_traceable_back_to_the_leaf(self) -> None:
        parent_id = uuid.uuid4()
        child_id = uuid.uuid4()
        session = FakeSession([parent_chunk(parent_id)])

        result = await expander().expand(
            [leaf(parent_id, chunk_id=child_id)], session  # type: ignore[arg-type]
        )

        assert result.chunks[0].metadata["expanded_from_chunk_id"] == str(child_id)
        assert result.chunks[0].channel.endswith("+parent")

    @pytest.mark.asyncio
    async def test_several_leaves_in_one_section_collapse_to_one_parent(self) -> None:
        # The common case, and the reason expansion does not multiply context size.
        parent_id = uuid.uuid4()
        session = FakeSession([parent_chunk(parent_id)])
        chunks = [leaf(parent_id, rank=i) for i in range(4)]

        result = await expander().expand(chunks, session)  # type: ignore[arg-type]

        assert len(result.chunks) == 1
        assert result.expanded == 1
        assert result.parents_deduplicated == 3

    @pytest.mark.asyncio
    async def test_parents_are_fetched_in_one_query(self) -> None:
        parents = [parent_chunk(uuid.uuid4()) for _ in range(3)]
        session = FakeSession(parents)
        chunks = [leaf(p.id, rank=i) for i, p in enumerate(parents)]

        await expander().expand(chunks, session)  # type: ignore[arg-type]

        assert session.queries == 1


class TestBudget:
    @pytest.mark.asyncio
    async def test_expansion_stops_at_the_budget(self) -> None:
        parents = [parent_chunk(uuid.uuid4(), tokens=400) for _ in range(5)]
        session = FakeSession(parents)
        chunks = [leaf(p.id, rank=i) for i, p in enumerate(parents)]

        result = await expander(budget=1000).expand(chunks, session)  # type: ignore[arg-type]

        assert result.tokens_after <= 1000
        assert sum(c.token_count for c in result.chunks) <= 1000

    @pytest.mark.asyncio
    async def test_a_leaf_is_kept_when_its_parent_does_not_fit(self) -> None:
        # Dropping it would lose evidence retrieval had already found — worse
        # than handing the generator the fragment.
        big_parent = parent_chunk(uuid.uuid4(), tokens=5000)
        session = FakeSession([big_parent])

        result = await expander(budget=500).expand(
            [leaf(big_parent.id, tokens=100)], session  # type: ignore[arg-type]
        )

        assert result.expanded == 0
        assert result.kept_as_leaf == 1
        assert len(result.chunks) == 1
        assert result.chunks[0].token_count == 100

    @pytest.mark.asyncio
    async def test_the_highest_ranked_material_survives_the_budget(self) -> None:
        parents = [parent_chunk(uuid.uuid4(), tokens=400) for _ in range(4)]
        session = FakeSession(parents)
        # Deliberately supplied worst-first to prove ordering is by rank.
        chunks = [leaf(p.id, rank=3 - i, score=0.5 + 0.1 * i) for i, p in enumerate(parents)]

        result = await expander(budget=800).expand(chunks, session)  # type: ignore[arg-type]

        assert [c.rank for c in result.chunks] == [0, 1]

    @pytest.mark.asyncio
    async def test_the_worst_case_corpus_section_cannot_blow_the_budget(self) -> None:
        # Stage 5's exit gate: context stays within budget on the worst case.
        huge = [parent_chunk(uuid.uuid4(), tokens=9000) for _ in range(8)]
        session = FakeSession(huge)
        chunks = [leaf(p.id, rank=i, tokens=200) for i, p in enumerate(huge)]

        result = await expander(budget=6000).expand(chunks, session)  # type: ignore[arg-type]

        assert result.tokens_after <= 6000


class TestNonHierarchicalCorpus:
    @pytest.mark.asyncio
    async def test_chunks_without_parents_pass_through(self) -> None:
        # `fixed` remains a supported configuration and Task 5.3 compares against
        # it, so a corpus with no hierarchy must not error.
        session = FakeSession([])
        chunks = [leaf(None, rank=i) for i in range(3)]

        result = await expander().expand(chunks, session)  # type: ignore[arg-type]

        assert result.chunks == chunks
        assert result.expanded == 0
        assert result.kept_as_leaf == 3
        assert session.queries == 0

    @pytest.mark.asyncio
    async def test_a_dangling_parent_reference_keeps_the_leaf(self) -> None:
        # The parent row was deleted (ondelete=SET NULL races, or a partial
        # re-chunk). Losing the leaf too would compound the problem.
        session = FakeSession([])

        result = await expander().expand([leaf(uuid.uuid4())], session)  # type: ignore[arg-type]

        assert result.kept_as_leaf == 1
        assert len(result.chunks) == 1

    @pytest.mark.asyncio
    async def test_an_empty_retrieval_is_handled(self) -> None:
        result = await expander().expand([], FakeSession([]))  # type: ignore[arg-type]

        assert result.chunks == []
        assert result.was_noop
