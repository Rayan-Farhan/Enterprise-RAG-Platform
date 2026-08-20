"""Parent-child expansion of retrieved chunks (Task 5.2, master §19).

Small chunks retrieve well and read badly. A leaf that matches "96 work hours"
is the right thing to *find* and the wrong thing to hand a generator, which then
has to answer a question about sick leave from a sentence with no subject.

So retrieval matches leaves and generation reads sections: each retrieved leaf is
replaced by the section chunk it belongs to, which the hierarchical strategy
recorded as ``parent_chunk_id`` at chunking time.

Three properties this has to hold, in descending order of how expensive it is to
get wrong:

**Expansion is budgeted.** A section is several times the size of a leaf, so
naively expanding eight hits can exceed the generation context on its own. The
expander works to an explicit token budget and stops, keeping the highest-ranked
material.

**A parent inherits its best child's rank.** Ordering is what survives the
downstream context budget (`ContextAssembler` truncates the tail), so a parent
promoted to the top of the list because it happens to be large would push out a
better-matching one.

**A leaf whose parent would not fit stays.** Dropping it entirely would lose
evidence that retrieval had already found — worse than the fragment.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.db.models.chunk import Chunk
from app.retrieval.schemas import RetrievedChunk

logger = get_logger("app.retrieval.expansion")


@dataclass
class ExpansionResult:
    """What expansion did, in terms the experiment record can carry."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    expanded: int = 0
    kept_as_leaf: int = 0
    parents_deduplicated: int = 0
    dropped_for_budget: int = 0
    tokens_before: int = 0
    tokens_after: int = 0

    @property
    def was_noop(self) -> bool:
        return self.expanded == 0


class ParentExpander:
    """Replaces retrieved leaves with the section chunks that contain them."""

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    async def expand(
        self,
        chunks: list[RetrievedChunk],
        session: AsyncSession,
        budget_tokens: int | None = None,
    ) -> ExpansionResult:
        """Expand retrieved leaves to their parents, within a token budget."""
        result = ExpansionResult(
            chunks=list(chunks),
            tokens_before=sum(c.token_count for c in chunks),
        )
        result.tokens_after = result.tokens_before

        if not chunks or not self.settings.ENABLE_PARENT_EXPANSION:
            return result

        budget = budget_tokens or self.settings.PARENT_EXPANSION_BUDGET_TOKENS
        parent_ids = {
            c.metadata["parent_chunk_id"]
            for c in chunks
            if c.metadata.get("parent_chunk_id")
        }
        if not parent_ids:
            # Nothing to expand into: the corpus was chunked by a strategy that
            # emits no hierarchy. Not an error — `fixed` is still a supported
            # configuration and Stage 5.3 compares against it.
            result.kept_as_leaf = len(chunks)
            return result

        parents = await self._load_parents(session, parent_ids)

        expanded: list[RetrievedChunk] = []
        seen_parents: set[uuid.UUID] = set()
        used = 0

        # Rank order, best first: the budget truncates the tail, so what is kept
        # should be what matched best rather than what happened to be small.
        for chunk in sorted(chunks, key=lambda c: (c.rank, -c.score)):
            raw_parent_id = chunk.metadata.get("parent_chunk_id")
            parent = parents.get(uuid.UUID(str(raw_parent_id))) if raw_parent_id else None

            if parent is None:
                if used + chunk.token_count <= budget:
                    expanded.append(chunk)
                    used += chunk.token_count
                    result.kept_as_leaf += 1
                else:
                    result.dropped_for_budget += 1
                continue

            if parent.id in seen_parents:
                # Several leaves from one section collapse into a single parent —
                # the common case, and the reason expansion does not simply
                # multiply the context size.
                result.parents_deduplicated += 1
                continue

            promoted = self._as_retrieved(parent, chunk)
            if used + promoted.token_count > budget:
                # The parent will not fit. Keep the leaf rather than nothing: it
                # is evidence retrieval already found.
                if used + chunk.token_count <= budget:
                    expanded.append(chunk)
                    used += chunk.token_count
                    result.kept_as_leaf += 1
                else:
                    result.dropped_for_budget += 1
                continue

            seen_parents.add(parent.id)
            expanded.append(promoted)
            used += promoted.token_count
            result.expanded += 1

        result.chunks = expanded
        result.tokens_after = used

        logger.info(
            "parent_expansion_complete",
            retrieved=len(chunks),
            returned=len(expanded),
            expanded=result.expanded,
            kept_as_leaf=result.kept_as_leaf,
            deduplicated=result.parents_deduplicated,
            dropped_for_budget=result.dropped_for_budget,
            tokens_before=result.tokens_before,
            tokens_after=result.tokens_after,
            budget=budget,
        )
        return result

    @staticmethod
    async def _load_parents(
        session: AsyncSession,
        parent_ids: set[object],
    ) -> dict[uuid.UUID, Chunk]:
        """Fetch every parent in one query rather than one per hit."""
        ids = {uuid.UUID(str(pid)) for pid in parent_ids if pid}
        if not ids:
            return {}
        rows = await session.execute(select(Chunk).where(Chunk.id.in_(ids)))
        return {chunk.id: chunk for chunk in rows.scalars().all()}

    @staticmethod
    def _as_retrieved(parent: Chunk, child: RetrievedChunk) -> RetrievedChunk:
        """Present a parent section as a retrieval hit, inheriting its child's rank.

        Score and rank come from the child because the parent was never scored —
        it is being included on the strength of what matched inside it, and
        inventing a score for it would corrupt the ordering the context budget
        depends on.
        """
        return RetrievedChunk(
            chunk_id=parent.id,
            document_id=parent.document_id,
            version_id=parent.version_id,
            content=parent.content,
            score=child.score,
            channel=f"{child.channel}+parent",
            rank=child.rank,
            chunk_index=parent.chunk_index,
            chunk_type=parent.chunk_type,
            token_count=parent.token_count,
            page_number=parent.primary_page_number,
            page_span=list(parent.page_span or []),
            section_path=list(parent.section_path or []),
            element_ids=list(parent.element_ids or []),
            bounding_box=parent.bounding_box,
            document_title=child.document_title,
            version_number=child.version_number,
            metadata={
                **child.metadata,
                "expanded_from_chunk_id": str(child.chunk_id),
                "parent_chunk_id": None,
            },
        )


_expander: ParentExpander | None = None


def get_parent_expander() -> ParentExpander:
    """Return the singleton ParentExpander."""
    global _expander
    if _expander is None:
        _expander = ParentExpander()
    return _expander
