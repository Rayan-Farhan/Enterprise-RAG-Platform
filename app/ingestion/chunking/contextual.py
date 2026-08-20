"""Contextual chunking — chunks that carry their own provenance (Task 5.1).

A chunk lifted out of a 90-page policy manual loses everything that told a
reader which policy it belonged to. "Employees are entitled to 96 work hours"
embeds almost identically whether it came from the staff handbook or the faculty
handbook, and the dev split contains questions that turn on exactly that
distinction — the corpus carries three documents restating the same policies.

So each chunk's text is prefixed with the document title and its section path
before embedding. The prefix is part of the embedded content by design: it is
what makes the vector encode "sick leave, in the staff handbook" rather than
"sick leave". Measured at +0.13 recall@5 over the fixed baseline (Task 5.3).

The cost is real and was weighed in Task 5.3, not hand-waved:

* the prefix consumes tokens from the chunk budget, so less body text fits;
* every chunk in a section shares an identical prefix, which pulls their vectors
  together and can blunt discrimination *within* a section while sharpening it
  between documents.
"""

from __future__ import annotations

from app.db.models.element import Element
from app.ingestion.chunking.base import (
    ChunkCandidate,
    ChunkingContext,
    estimate_tokens,
)
from app.ingestion.chunking.provenance import (
    PREFIX_SEPARATOR,
    build_prefix,
    reserve_tokens,
)
from app.ingestion.chunking.structure_aware import StructureAwareChunker


class ContextualChunker:
    """Structure-aware chunking, with document and section provenance prefixed.

    Deliberately composed from :class:`StructureAwareChunker` rather than
    reimplementing the walk: the two must differ *only* in the prefix, or Task
    5.3 cannot attribute a metric change to the prefix rather than to some
    incidental difference in how they split.
    """

    strategy_name = "contextual"

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 64,
        chunking_version: str = "contextual-v1",
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

    def chunk(
        self,
        elements: list[Element],
        context: ChunkingContext | None = None,
    ) -> list[ChunkCandidate]:
        """Chunk structurally, then prefix each chunk with its provenance."""
        title = (context.document_title if context else "") or ""
        metadata = dict(context.metadata) if context and context.metadata else {}

        # The body is chunked to a reduced budget so that body plus prefix still
        # fits the nominal size — see provenance.reserve_tokens for why.
        reserve = reserve_tokens(title, elements, metadata)
        inner = StructureAwareChunker(
            chunk_size_tokens=max(1, self.chunk_size_tokens - reserve),
            chunk_overlap_tokens=min(
                self.chunk_overlap_tokens, max(0, self.chunk_size_tokens - reserve - 1)
            ),
            chunking_version=self.chunking_version,
        )
        candidates = inner.chunk(elements, context)
        apply_prefix(candidates, title, metadata)
        return candidates


def apply_prefix(
    candidates: list[ChunkCandidate],
    title: str,
    metadata: dict[str, object],
) -> None:
    """Prefix each candidate in place and refresh its token count.

    Refreshing matters: a recorded count that excludes the prefix understates
    what is actually embedded, and every budget downstream reads that number.
    """
    for candidate in candidates:
        prefix = build_prefix(title, candidate.section_path, metadata)
        if not prefix:
            continue
        candidate.content = f"{prefix}{PREFIX_SEPARATOR}{candidate.content}"
        candidate.token_count = estimate_tokens(candidate.content)
