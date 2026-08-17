"""Fixed-size token chunking with overlap — the Stage 3 baseline (Task 3.1).

This is deliberately the naive strategy. It exists so Stage 5's structure-aware,
hierarchical, and contextual strategies have a measured baseline to beat, and it
is retained behind config afterwards rather than deleted (Stage 5 exit gate).
"""

from __future__ import annotations

import re

from app.db.models.element import Element
from app.ingestion.chunking.base import (
    ChunkCandidate,
    ChunkType,
    estimate_tokens,
    union_bounding_box,
)

# Sentence-ish boundary: keeps fixed-size splits from landing mid-sentence when
# a cheap boundary is available nearby.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+|\n+")

_TABLE_TYPES = frozenset({"table"})
_FIGURE_TYPES = frozenset({"figure", "image"})


class FixedSizeChunker:
    """Packs canonical elements into fixed token-budget chunks with overlap.

    Boilerplate elements are excluded (Stage 2 flags them; chunking is where that
    flag pays off). Tables are emitted as standalone chunks rather than packed
    with prose, because splitting a table mid-row destroys it and ADR-006 forbids
    that even for the baseline strategy.
    """

    strategy_name = "fixed"

    # A heading is a label, not a paragraph. Parsers do misclassify body text as
    # headings (measured on the real HR corpus: 137 of 194 "level 3 headings" were
    # body paragraphs, one of them 3309 characters). Trusting that classification
    # blindly moves real policy text out of chunk content and into `section_path`,
    # where it is never retrieved. Anything longer than this is treated as body
    # text so the content still gets indexed.
    MAX_HEADING_TOKENS = 25
    MAX_SECTION_PATH_DEPTH = 6

    def __init__(
        self,
        chunk_size_tokens: int = 512,
        chunk_overlap_tokens: int = 64,
        chunking_version: str = "fixed-v1",
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

    def chunk(self, elements: list[Element]) -> list[ChunkCandidate]:
        """Convert ordered canonical elements into fixed-size chunk candidates."""
        ordered = sorted(
            (el for el in elements if not el.is_boilerplate),
            key=lambda el: el.sequence_index,
        )

        candidates: list[ChunkCandidate] = []
        section_path: list[str] = []
        buffer: list[_Piece] = []

        for element in ordered:
            element_type = (element.element_type or "").lower()

            if element_type == "heading" and self._is_plausible_heading(element):
                # A heading closes the current chunk and updates the running
                # ancestry so every subsequent chunk carries its section path.
                self._flush(buffer, section_path, candidates)
                buffer = []
                section_path = self._advance_section_path(section_path, element)
                continue

            if element_type in _TABLE_TYPES:
                self._flush(buffer, section_path, candidates)
                buffer = []
                table_chunk = self._table_candidate(element, section_path, len(candidates))
                if table_chunk is not None:
                    candidates.append(table_chunk)
                continue

            text = (element.text_content or "").strip()
            if not text:
                continue

            for piece in self._split_element(element, text):
                if buffer and self._buffer_tokens(buffer) + piece.tokens > self.chunk_size_tokens:
                    self._flush(buffer, section_path, candidates)
                    buffer = self._carry_overlap(buffer)
                buffer.append(piece)

        self._flush(buffer, section_path, candidates)
        return candidates

    # ---- internals -----------------------------------------------------

    def _split_element(self, element: Element, text: str) -> list[_Piece]:
        """Break an element into pieces no larger than the chunk budget."""
        element_type = (element.element_type or "").lower()
        if estimate_tokens(text) <= self.chunk_size_tokens:
            return [_Piece(element, text, estimate_tokens(text), element_type)]

        pieces: list[_Piece] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in (s for s in _SENTENCE_SPLIT_RE.split(text) if s and s.strip()):
            sentence_tokens = estimate_tokens(sentence)
            if current and current_tokens + sentence_tokens > self.chunk_size_tokens:
                pieces.append(_Piece(element, " ".join(current), current_tokens, element_type))
                current = []
                current_tokens = 0
            current.append(sentence.strip())
            current_tokens += sentence_tokens

        if current:
            pieces.append(_Piece(element, " ".join(current), current_tokens, element_type))

        return pieces

    def _carry_overlap(self, buffer: list[_Piece]) -> list[_Piece]:
        """Return the tail of a flushed buffer to prepend to the next chunk.

        The overlap budget is a hard cap. An earlier version admitted the first
        piece unconditionally, so a single large trailing piece could be carried
        whole — producing chunks ~50% over the size budget on the real corpus
        (measured: 783 tokens against a 512 budget).
        """
        if self.chunk_overlap_tokens == 0:
            return []

        carried: list[_Piece] = []
        total = 0
        for piece in reversed(buffer):
            if total + piece.tokens > self.chunk_overlap_tokens:
                break
            carried.insert(0, piece)
            total += piece.tokens
        return carried

    def _flush(
        self,
        buffer: list[_Piece],
        section_path: list[str],
        candidates: list[ChunkCandidate],
    ) -> None:
        """Materialise the buffered pieces as a chunk candidate."""
        if not buffer:
            return

        content = "\n".join(piece.text for piece in buffer).strip()
        if not content:
            return

        # Preserve first-appearance order while de-duplicating: one element can
        # contribute several pieces, and overlap can re-introduce an element.
        element_ids = list(dict.fromkeys(piece.element.element_id for piece in buffer))
        pages = sorted({piece.element.page_number for piece in buffer})
        boxes = [piece.element.bounding_box for piece in buffer if piece.element.bounding_box]

        candidates.append(
            ChunkCandidate(
                chunk_index=len(candidates),
                content=content,
                element_ids=element_ids,
                section_path=list(section_path),
                primary_page_number=pages[0],
                page_span=pages,
                token_count=estimate_tokens(content),
                chunk_type=self._infer_chunk_type(buffer),
                bounding_box=union_bounding_box([b for b in boxes if b]),
            )
        )

    def _table_candidate(
        self,
        element: Element,
        section_path: list[str],
        chunk_index: int,
    ) -> ChunkCandidate | None:
        """Emit a table as one coherent chunk, never split mid-row (ADR-006)."""
        table_data = element.table_data or {}
        content = str(table_data.get("markdown") or "").strip()
        if not content:
            content = (element.text_content or "").strip()
        if not content:
            return None

        return ChunkCandidate(
            chunk_index=chunk_index,
            content=content,
            element_ids=[element.element_id],
            section_path=list(section_path),
            primary_page_number=element.page_number,
            page_span=[element.page_number],
            token_count=estimate_tokens(content),
            chunk_type=ChunkType.TABLE,
            bounding_box=union_bounding_box(
                [element.bounding_box] if element.bounding_box else []
            ),
        )

    @classmethod
    def _is_plausible_heading(cls, element: Element) -> bool:
        """Reject 'headings' that are really body text (see MAX_HEADING_TOKENS)."""
        text = (element.text_content or "").strip()
        if not text:
            return False
        return estimate_tokens(text) <= cls.MAX_HEADING_TOKENS

    @classmethod
    def _advance_section_path(cls, section_path: list[str], heading: Element) -> list[str]:
        """Update heading ancestry using the heading level recorded at ingestion."""
        title = (heading.text_content or "").strip()
        if not title:
            return section_path

        level_raw = (heading.extra_metadata or {}).get("heading_level")
        try:
            level = int(level_raw) if level_raw is not None else len(section_path) + 1
        except (TypeError, ValueError):
            level = len(section_path) + 1
        level = max(1, min(level, cls.MAX_SECTION_PATH_DEPTH))

        # Truncate to the parent depth and append. A level that skips ancestors
        # (H1 -> H3, common in real documents) is clamped to the next available
        # depth rather than padded with empty strings, which would put meaningless
        # "" segments into every citation's section label.
        truncated = section_path[: level - 1]
        return [*truncated, title]

    @staticmethod
    def _infer_chunk_type(buffer: list[_Piece]) -> ChunkType:
        """Classify a chunk from the element types that composed it."""
        types = {piece.element_type for piece in buffer}
        if types == {"paragraph"}:
            return ChunkType.PARAGRAPH
        if types == {"list"}:
            return ChunkType.LIST
        if types and types <= _FIGURE_TYPES:
            return ChunkType.FIGURE
        return ChunkType.MIXED

    @staticmethod
    def _buffer_tokens(buffer: list[_Piece]) -> int:
        return sum(piece.tokens for piece in buffer)


class _Piece:
    """A unit of text from one element that fits within the chunk budget."""

    __slots__ = ("element", "element_type", "text", "tokens")

    def __init__(self, element: Element, text: str, tokens: int, element_type: str) -> None:
        self.element = element
        self.text = text
        self.tokens = tokens
        self.element_type = element_type
