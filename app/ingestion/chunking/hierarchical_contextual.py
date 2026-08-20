"""Hierarchical chunking with provenance prefixes (Task 5.3, ADR-006).

The Stage 5 sweep tested ``hierarchical`` and ``contextual`` as alternatives.
They are not alternatives: the prefix is applied to a chunk's *text*, while the
hierarchy is a *link between* chunks. Nothing prevents having both.

The sweep suggested it was worth building. ``hierarchical`` came last on
retrieval and ``contextual`` first, but a controlled comparison showed the gap
was not about chunk size — hierarchical's ~228-token leaves scored 0.280 while
contextual's ~211-token chunks scored 0.488. What hierarchical lacked was the
prefix, not smaller chunks.

This strategy exists so that claim can be tested rather than assumed, and so
ADR-006's locked decision (hierarchical) and the measured winner (the prefix)
can potentially be satisfied at once instead of traded off.

Composed from both rather than reimplemented, for the same reason
``contextual`` composes ``structure_aware``: if it split differently from
``hierarchical`` or prefixed differently from ``contextual``, a comparison
against either would be measuring an incidental difference.
"""

from __future__ import annotations

from app.db.models.element import Element
from app.ingestion.chunking.base import ChunkCandidate, ChunkingContext
from app.ingestion.chunking.contextual import apply_prefix
from app.ingestion.chunking.hierarchical import HierarchicalChunker
from app.ingestion.chunking.provenance import reserve_tokens


class HierarchicalContextualChunker:
    """Section/leaf hierarchy, with every chunk carrying its own provenance."""

    strategy_name = "hierarchical_contextual"

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 64,
        chunking_version: str = "hierarchical_contextual-v1",
        section_budget_multiplier: int = 4,
    ) -> None:
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        if chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must not be negative")
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens, "
                f"got overlap={chunk_overlap_tokens} size={chunk_size_tokens}"
            )

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.chunking_version = chunking_version
        self.section_budget_multiplier = section_budget_multiplier

    def chunk(
        self,
        elements: list[Element],
        context: ChunkingContext | None = None,
    ) -> list[ChunkCandidate]:
        """Build the hierarchy against a reduced budget, then prefix every chunk."""
        title = (context.document_title if context else "") or ""
        metadata = dict(context.metadata) if context and context.metadata else {}

        # Leaves *and* section chunks are prefixed, so both budgets are reduced.
        # Reserving only from the leaf budget would let section chunks drift over
        # the ceiling that Task 5.2's parent expansion budgets against.
        reserve = reserve_tokens(title, elements, metadata)
        inner = HierarchicalChunker(
            chunk_size_tokens=max(1, self.chunk_size_tokens - reserve),
            chunk_overlap_tokens=min(
                self.chunk_overlap_tokens, max(0, self.chunk_size_tokens - reserve - 1)
            ),
            chunking_version=self.chunking_version,
            section_budget_multiplier=self.section_budget_multiplier,
        )

        candidates = inner.chunk(elements, context)
        apply_prefix(candidates, title, metadata)
        return candidates
