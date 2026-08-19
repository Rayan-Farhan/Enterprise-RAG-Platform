"""Shared document segmentation for the Stage 5 strategies (Task 5.1, ADR-006).

``structure_aware``, ``hierarchical``, and ``contextual`` all need the same
first step: walk the elements in reading order and recover the heading tree the
parser flattened. Only what they *do* with that tree differs. Keeping the walk
in one place means a fix to heading detection lands in all three at once, rather
than in whichever one someone remembered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.db.models.element import Element
from app.ingestion.chunking.base import ChunkType, estimate_tokens

#: A heading is a label, not a paragraph. The real HR corpus has parser output
#: where 137 of 194 "level 3 headings" were body paragraphs, one of them 3309
#: characters. Treating those as headings moves real policy text out of chunk
#: content and into ``section_path``, where nothing ever retrieves it.
MAX_HEADING_TOKENS = 25

#: Depth cap for heading ancestry, so a pathological document cannot produce
#: section paths too long to render in a citation.
MAX_SECTION_PATH_DEPTH = 6

TABLE_TYPES = frozenset({"table"})
FIGURE_TYPES = frozenset({"figure", "image"})
LIST_TYPES = frozenset({"list"})


def is_plausible_heading(element: Element) -> bool:
    """True when an element classified as a heading actually reads like one."""
    if (element.element_type or "").lower() != "heading":
        return False
    text = (element.text_content or "").strip()
    return bool(text) and estimate_tokens(text) <= MAX_HEADING_TOKENS


def advance_section_path(section_path: list[str], heading: Element) -> list[str]:
    """Update heading ancestry using the level recorded at ingestion.

    A level that skips ancestors (H1 -> H3, common in real documents) is clamped
    to the next available depth rather than padded with empty strings, which
    would put meaningless "" segments into every citation's section label.
    """
    title = (heading.text_content or "").strip()
    if not title:
        return section_path

    raw = (heading.extra_metadata or {}).get("heading_level")
    try:
        level = int(raw) if raw is not None else len(section_path) + 1
    except (TypeError, ValueError):
        level = len(section_path) + 1
    level = max(1, min(level, MAX_SECTION_PATH_DEPTH))

    return [*section_path[: level - 1], title]


def element_chunk_type(element: Element) -> ChunkType:
    """Classify a single element for chunk typing."""
    element_type = (element.element_type or "").lower()
    if element_type in TABLE_TYPES:
        return ChunkType.TABLE
    if element_type in FIGURE_TYPES:
        return ChunkType.FIGURE
    if element_type in LIST_TYPES:
        return ChunkType.LIST
    if element_type == "paragraph":
        return ChunkType.PARAGRAPH
    return ChunkType.MIXED


def combined_chunk_type(elements: list[Element]) -> ChunkType:
    """Classify a chunk from the element types that composed it."""
    types = {element_chunk_type(el) for el in elements}
    if len(types) == 1:
        return next(iter(types))
    return ChunkType.MIXED


def table_content(element: Element) -> str:
    """Best available text for a table element, markdown preferred."""
    table_data = element.table_data or {}
    content = str(table_data.get("markdown") or "").strip()
    return content or (element.text_content or "").strip()


@dataclass
class Block:
    """One body element under a heading path, with its text already resolved."""

    element: Element
    text: str
    tokens: int
    chunk_type: ChunkType
    section_path: list[str]

    @property
    def is_table(self) -> bool:
        return self.chunk_type is ChunkType.TABLE


@dataclass
class Section:
    """A run of body blocks sharing one heading path.

    Sections are the unit ``structure_aware`` refuses to pack across and the
    parent ``hierarchical`` hangs its leaves from. A document with no headings
    at all yields exactly one section with an empty path, so every strategy has
    a well-defined answer for unstructured input rather than a special case.
    """

    section_path: list[str]
    blocks: list[Block] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return sum(block.tokens for block in self.blocks)

    @property
    def elements(self) -> list[Element]:
        return [block.element for block in self.blocks]

    @property
    def title(self) -> str:
        return self.section_path[-1] if self.section_path else ""


def segment(elements: list[Element]) -> list[Section]:
    """Group ordered elements into sections under their heading ancestry.

    Boilerplate is excluded here rather than by each strategy: Stage 2 flags it
    and chunking is where that flag pays off. Empty-text elements are dropped
    for the same reason — they would otherwise create chunks with no content but
    real element IDs, which inflates recall denominators in Stage 4's metrics.
    """
    ordered = sorted(
        (el for el in elements if not el.is_boilerplate),
        key=lambda el: el.sequence_index,
    )

    sections: list[Section] = []
    section_path: list[str] = []
    current = Section(section_path=[])

    for element in ordered:
        if is_plausible_heading(element):
            if current.blocks:
                sections.append(current)
            section_path = advance_section_path(section_path, element)
            current = Section(section_path=list(section_path))
            continue

        chunk_type = element_chunk_type(element)
        text = (
            table_content(element)
            if chunk_type is ChunkType.TABLE
            else (element.text_content or "").strip()
        )
        if not text:
            continue

        current.blocks.append(
            Block(
                element=element,
                text=text,
                tokens=estimate_tokens(text),
                chunk_type=chunk_type,
                section_path=list(section_path),
            )
        )

    if current.blocks:
        sections.append(current)
    return sections
