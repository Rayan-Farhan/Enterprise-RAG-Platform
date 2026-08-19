"""Hierarchical chunking — Document / Section / Chunk (Task 5.1, ADR-006).

The structure ADR-006 locks. Two kinds of chunk come out of one pass:

* **leaf chunks** — small, indexed for recall, the unit retrieval matches on;
* **section chunks** — the whole section as one coherent unit, which Task 5.2
  expands a matched leaf into so the generator sees the surrounding policy
  rather than the fragment that happened to match.

Both are emitted. Emitting only leaves would leave the parent unavailable at
generation time; emitting only sections would give retrieval nothing precise to
match. Each leaf records ``parent_index``, the position of its section chunk in
the same list, and Task 5.2 persists those links.

Section chunks are budgeted. A 40-page section is not a useful retrieval unit
and would exhaust the generation context on its own, so an oversized section is
represented by its opening, explicitly marked as abridged.
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


class HierarchicalChunker:
    """Emits section chunks and the leaf chunks that hang from them."""

    strategy_name = "hierarchical"

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 64,
        chunking_version: str = "hierarchical-v1",
        leaf_size_tokens: int | None = None,
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
        if section_budget_multiplier < 1:
            raise ValueError("section_budget_multiplier must be at least 1")

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self.chunking_version = chunking_version
        # Leaves are deliberately smaller than the nominal chunk size: they exist
        # to be matched precisely, and the parent supplies the context.
        self.leaf_size_tokens = leaf_size_tokens or max(1, chunk_size_tokens // 2)
        self.section_budget_tokens = chunk_size_tokens * section_budget_multiplier

    def chunk(
        self,
        elements: list[Element],
        context: ChunkingContext | None = None,
    ) -> list[ChunkCandidate]:
        """Convert ordered canonical elements into a section/leaf hierarchy."""
        candidates: list[ChunkCandidate] = []

        for section in segment(elements):
            parent_index = len(candidates)
            candidates.append(self._section_candidate(section, parent_index))
            self._leaf_candidates(section, parent_index, candidates)

        return candidates

    # ---- internals -----------------------------------------------------

    def _section_candidate(self, section: Section, chunk_index: int) -> ChunkCandidate:
        """The whole section as one chunk, abridged if it exceeds its budget."""
        full = "\n".join(block.text for block in section.blocks).strip()
        content = (
            self._abridge(section, full)
            if estimate_tokens(full) > self.section_budget_tokens
            else full
        )

        pages = sorted({block.element.page_number for block in section.blocks})
        boxes = [b.element.bounding_box for b in section.blocks if b.element.bounding_box]

        return ChunkCandidate(
            chunk_index=chunk_index,
            content=content,
            element_ids=[block.element.element_id for block in section.blocks],
            section_path=list(section.section_path),
            primary_page_number=pages[0],
            page_span=pages,
            token_count=estimate_tokens(content),
            chunk_type=ChunkType.SECTION,
            bounding_box=union_bounding_box([b for b in boxes if b]),
        )

    def _abridge(self, section: Section, full: str) -> str:
        """Represent an oversized section by its opening, marked as abridged.

        Marked rather than silently cut: a section chunk that looks complete and
        is not would let the generator answer "the policy does not say" from an
        artefact of chunking.
        """
        kept: list[str] = []
        total = 0
        for block in section.blocks:
            if total + block.tokens > self.section_budget_tokens:
                break
            kept.append(block.text)
            total += block.tokens

        if not kept:
            # One block larger than the entire section budget; take its head.
            kept = [split_block(full, self.section_budget_tokens)[0][0]]

        title = section.title or "this section"
        return "\n".join(kept).strip() + f"\n\n[... {title} continues beyond this excerpt ...]"

    def _leaf_candidates(
        self,
        section: Section,
        parent_index: int,
        candidates: list[ChunkCandidate],
    ) -> None:
        """Emit the small, precisely-matchable chunks under one section."""
        buffer: list[tuple[Block, str, int]] = []

        def flush() -> None:
            if buffer:
                candidates.append(self._leaf(buffer, section, len(candidates), parent_index))
                buffer.clear()

        for block in section.blocks:
            # A table is a coherent unit, never merged with prose or split
            # mid-row, even when it exceeds the budget (ADR-006).
            if block.is_table:
                flush()
                candidates.append(
                    self._leaf(
                        [(block, block.text, block.tokens)],
                        section,
                        len(candidates),
                        parent_index,
                    )
                )
                continue

            for text, tokens in split_block(block.text, self.leaf_size_tokens):
                if buffer and sum(t for _, _, t in buffer) + tokens > self.leaf_size_tokens:
                    flush()
                buffer.append((block, text, tokens))

        flush()

    @staticmethod
    def _leaf(
        buffer: list[tuple[Block, str, int]],
        section: Section,
        chunk_index: int,
        parent_index: int,
    ) -> ChunkCandidate:
        content = "\n".join(text for _, text, _ in buffer).strip()
        blocks = [block for block, _, _ in buffer]

        pages = sorted({block.element.page_number for block in blocks})
        boxes = [block.element.bounding_box for block in blocks if block.element.bounding_box]
        types = {block.chunk_type for block in blocks}

        return ChunkCandidate(
            chunk_index=chunk_index,
            content=content,
            element_ids=list(dict.fromkeys(block.element.element_id for block in blocks)),
            section_path=list(section.section_path),
            primary_page_number=pages[0],
            page_span=pages,
            token_count=estimate_tokens(content),
            chunk_type=next(iter(types)) if len(types) == 1 else ChunkType.MIXED,
            bounding_box=union_bounding_box([b for b in boxes if b]),
            parent_index=parent_index,
        )
