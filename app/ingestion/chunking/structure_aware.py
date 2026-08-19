"""Structure-aware chunking — splits on document structure (Task 5.1, ADR-006).

The fixed-size baseline packs text until a token budget is reached and then
cuts, so a chunk routinely begins mid-policy and ends mid-sentence of the next
one. This strategy never packs across a heading boundary: a chunk belongs to
exactly one section, and within a section it breaks on element boundaries.

The cost is variance — sections are not uniformly sized, so chunks are not
either. Whether that trade helps is Task 5.3's question, not an assumption made
here.
"""

from __future__ import annotations

from app.db.models.element import Element
from app.ingestion.chunking.base import (
    ChunkCandidate,
    ChunkingContext,
    ChunkType,
    estimate_tokens,
    union_bounding_box,
)
from app.ingestion.chunking.segmentation import Block, Section, segment
from app.ingestion.chunking.splitting import split_block


class StructureAwareChunker:
    """Packs elements into chunks that never span a heading boundary."""

    strategy_name = "structure_aware"

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 64,
        chunking_version: str = "structure_aware-v1",
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
        """Convert ordered canonical elements into structure-aligned chunks."""
        candidates: list[ChunkCandidate] = []
        for section in segment(elements):
            self._chunk_section(section, candidates)
        return candidates

    # ---- internals -----------------------------------------------------

    def _chunk_section(self, section: Section, candidates: list[ChunkCandidate]) -> None:
        """Emit chunks for one section, flushing at the token budget."""
        buffer: list[tuple[Block, str, int]] = []

        def flush() -> None:
            if buffer:
                candidates.append(self._materialise(buffer, section, len(candidates)))
                buffer.clear()

        for block in section.blocks:
            # A table is a coherent unit and is never merged with prose or split
            # mid-row, even when it exceeds the budget (ADR-006).
            if block.is_table:
                flush()
                candidates.append(
                    self._materialise([(block, block.text, block.tokens)], section, len(candidates))
                )
                continue

            for text, tokens in split_block(block.text, self.chunk_size_tokens):
                if buffer and sum(t for _, _, t in buffer) + tokens > self.chunk_size_tokens:
                    flush()
                buffer.append((block, text, tokens))

        flush()

    @staticmethod
    def _materialise(
        buffer: list[tuple[Block, str, int]],
        section: Section,
        chunk_index: int,
    ) -> ChunkCandidate:
        content = "\n".join(text for _, text, _ in buffer).strip()
        blocks = [block for block, _, _ in buffer]

        element_ids = list(dict.fromkeys(block.element.element_id for block in blocks))
        pages = sorted({block.element.page_number for block in blocks})
        boxes = [block.element.bounding_box for block in blocks if block.element.bounding_box]

        types = {block.chunk_type for block in blocks}
        chunk_type = next(iter(types)) if len(types) == 1 else ChunkType.MIXED

        return ChunkCandidate(
            chunk_index=chunk_index,
            content=content,
            element_ids=element_ids,
            section_path=list(section.section_path),
            primary_page_number=pages[0],
            page_span=pages,
            token_count=estimate_tokens(content),
            chunk_type=chunk_type,
            bounding_box=union_bounding_box([b for b in boxes if b]),
        )
