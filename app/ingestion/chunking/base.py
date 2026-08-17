"""Chunking protocol, deterministic identity, and token estimation (ADR-006, ADR-036, Task 3.1)."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from app.db.models.element import Element

# Stable namespace for all deterministic chunk identifiers. Changing this value
# re-identifies every chunk in the corpus, so it is a constant, never config.
CHUNK_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

_WORD_RE = re.compile(r"\w+|[^\w\s]")


class ChunkType(StrEnum):
    """Chunk composition types locked by ADR-006."""

    PARAGRAPH = "paragraph"
    SECTION = "section"
    TABLE = "table"
    FIGURE = "figure"
    LIST = "list"
    MIXED = "mixed"


def estimate_tokens(text: str) -> int:
    """Estimate a token count without a network-dependent tokenizer.

    Chunk boundaries must be reproducible offline and in CI, so this uses a
    deterministic lexical approximation (words plus punctuation, scaled for
    subword splitting) rather than a downloaded BPE vocabulary. Stage 5 replaces
    this with the real tokenizer of the locked embedding model once chunk
    parameters are benchmark-driven.
    """
    if not text:
        return 0
    lexemes = _WORD_RE.findall(text)
    # Empirically ~1.3 subword tokens per lexeme for English policy prose.
    return max(1, round(len(lexemes) * 1.3))


def compute_chunk_id(
    version_id: uuid.UUID,
    element_ids: list[str],
    chunk_index: int,
    chunking_version: str,
) -> uuid.UUID:
    """Derive a chunk's deterministic UUIDv5 identity (ADR-036).

    The inputs are exactly the four fields Task 3.1 specifies. Element IDs keep
    their given order because reading order is semantically meaningful — two
    chunks over the same elements in a different order are different chunks.
    """
    seed = "|".join(
        [
            str(version_id),
            ",".join(element_ids),
            str(chunk_index),
            chunking_version,
        ]
    )
    return uuid.uuid5(CHUNK_ID_NAMESPACE, seed)


def compute_point_id(chunk_id: uuid.UUID, embedding_version: str) -> str:
    """Derive the deterministic Qdrant point ID for a chunk's vector.

    Qdrant point IDs must be UUIDs or unsigned integers, so this returns a
    UUIDv5 string. Including ``embedding_version`` means re-embedding under a
    new model writes new points instead of silently overwriting vectors from a
    different model — required for the Stage 5/6 comparisons.
    """
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{chunk_id}|{embedding_version}"))


@dataclass
class ChunkCandidate:
    """A chunk produced by a strategy, before persistence."""

    chunk_index: int
    content: str
    element_ids: list[str]
    section_path: list[str]
    primary_page_number: int
    page_span: list[int]
    token_count: int
    chunk_type: ChunkType = ChunkType.MIXED
    bounding_box: dict[str, Any] | None = field(default=None)

    def identity(self, version_id: uuid.UUID, chunking_version: str) -> uuid.UUID:
        """Deterministic identity of this candidate under a given version."""
        return compute_chunk_id(
            version_id=version_id,
            element_ids=self.element_ids,
            chunk_index=self.chunk_index,
            chunking_version=chunking_version,
        )


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Common interface so Stage 5 can swap strategies by configuration alone."""

    strategy_name: str
    chunking_version: str

    def chunk(self, elements: list[Element]) -> list[ChunkCandidate]:
        """Convert ordered canonical elements into chunk candidates."""
        ...


def union_bounding_box(boxes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Compute the union of element bounding boxes that share a page.

    Boxes from different pages are not unionable into a single rectangle, so the
    union is taken over the page contributing the most boxes and tagged with
    that page number. Stage 9 needs per-element boxes for crop citations and
    reads those from ``Element`` directly rather than from this summary.
    """
    usable = [b for b in boxes if b and all(k in b for k in ("x0", "y0", "x1", "y1"))]
    if not usable:
        return None

    by_page: dict[int, list[dict[str, Any]]] = {}
    for box in usable:
        by_page.setdefault(int(box.get("page_number", 0)), []).append(box)

    page_number, page_boxes = max(by_page.items(), key=lambda item: len(item[1]))
    return {
        "x0": min(float(b["x0"]) for b in page_boxes),
        "y0": min(float(b["y0"]) for b in page_boxes),
        "x1": max(float(b["x1"]) for b in page_boxes),
        "y1": max(float(b["y1"]) for b in page_boxes),
        "page_number": page_number,
        "unit": str(page_boxes[0].get("unit", "pt")),
    }
