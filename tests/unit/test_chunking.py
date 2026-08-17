"""Unit tests for baseline chunking and deterministic chunk identity (Task 3.1)."""

from __future__ import annotations

import uuid

import pytest

from app.db.models.element import Element
from app.ingestion.chunking.base import (
    ChunkType,
    compute_chunk_id,
    compute_point_id,
    estimate_tokens,
    union_bounding_box,
)
from app.ingestion.chunking.fixed_size import FixedSizeChunker

VERSION_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")


def make_element(
    element_id: str,
    text: str,
    element_type: str = "paragraph",
    page_number: int = 1,
    sequence_index: int = 0,
    is_boilerplate: bool = False,
    heading_level: int | None = None,
    bounding_box: dict | None = None,
    table_data: dict | None = None,
) -> Element:
    """Build an in-memory Element without touching the database."""
    return Element(
        id=uuid.uuid4(),
        version_id=VERSION_ID,
        page_id=uuid.uuid4(),
        page_number=page_number,
        element_id=element_id,
        element_type=element_type,
        sequence_index=sequence_index,
        text_content=text,
        content_hash="0" * 64,
        is_boilerplate=is_boilerplate,
        bounding_box=bounding_box,
        table_data=table_data,
        extra_metadata={"heading_level": heading_level} if heading_level else {},
    )


class TestDeterministicIdentity:
    def test_same_inputs_produce_same_chunk_id(self) -> None:
        a = compute_chunk_id(VERSION_ID, ["e1", "e2"], 0, "fixed-v1")
        b = compute_chunk_id(VERSION_ID, ["e1", "e2"], 0, "fixed-v1")
        assert a == b

    @pytest.mark.parametrize(
        "element_ids,chunk_index,chunking_version",
        [
            (["e1", "e3"], 0, "fixed-v1"),
            (["e2", "e1"], 0, "fixed-v1"),  # order is significant
            (["e1", "e2"], 1, "fixed-v1"),
            (["e1", "e2"], 0, "fixed-v2"),
        ],
    )
    def test_each_input_dimension_changes_the_id(
        self, element_ids: list[str], chunk_index: int, chunking_version: str
    ) -> None:
        baseline = compute_chunk_id(VERSION_ID, ["e1", "e2"], 0, "fixed-v1")
        variant = compute_chunk_id(VERSION_ID, element_ids, chunk_index, chunking_version)
        assert variant != baseline

    def test_different_version_produces_different_id(self) -> None:
        other_version = uuid.UUID("22222222-2222-2222-2222-222222222222")
        assert compute_chunk_id(VERSION_ID, ["e1"], 0, "fixed-v1") != compute_chunk_id(
            other_version, ["e1"], 0, "fixed-v1"
        )

    def test_point_id_is_deterministic_and_embedding_version_scoped(self) -> None:
        chunk_id = compute_chunk_id(VERSION_ID, ["e1"], 0, "fixed-v1")
        first = compute_point_id(chunk_id, "jina-v3")
        assert first == compute_point_id(chunk_id, "jina-v3")
        assert first != compute_point_id(chunk_id, "jina-v4")
        # Qdrant requires a UUID or unsigned int point ID.
        uuid.UUID(first)


class TestTokenEstimation:
    def test_empty_text_is_zero_tokens(self) -> None:
        assert estimate_tokens("") == 0

    def test_estimate_grows_with_length(self) -> None:
        short = estimate_tokens("Annual leave is 21 days.")
        long = estimate_tokens("Annual leave is 21 days. " * 20)
        assert 0 < short < long

    def test_estimate_is_deterministic(self) -> None:
        text = "Employees on probation accrue leave at 1.75 days per month."
        assert estimate_tokens(text) == estimate_tokens(text)


class TestFixedSizeChunker:
    def test_rejects_overlap_at_or_above_size(self) -> None:
        with pytest.raises(ValueError, match="smaller than chunk_size_tokens"):
            FixedSizeChunker(chunk_size_tokens=100, chunk_overlap_tokens=100)

    def test_chunks_carry_full_ancestry(self) -> None:
        elements = [
            make_element("h1", "Leave Policy", "heading", sequence_index=0, heading_level=1),
            make_element("h2", "Annual Leave", "heading", sequence_index=1, heading_level=2),
            make_element("p1", "Employees receive 21 days of annual leave.", sequence_index=2),
        ]
        chunks = FixedSizeChunker(chunk_size_tokens=200, chunk_overlap_tokens=0).chunk(elements)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.section_path == ["Leave Policy", "Annual Leave"]
        assert chunk.element_ids == ["p1"]
        assert chunk.primary_page_number == 1
        assert chunk.page_span == [1]
        assert chunk.token_count > 0

    def test_heading_hierarchy_truncates_on_sibling(self) -> None:
        elements = [
            make_element("h1", "Leave", "heading", sequence_index=0, heading_level=1),
            make_element("h2", "Annual", "heading", sequence_index=1, heading_level=2),
            make_element("p1", "Annual leave text here for the chunk.", sequence_index=2),
            make_element("h3", "Sick", "heading", sequence_index=3, heading_level=2),
            make_element("p2", "Sick leave text here for the chunk.", sequence_index=4),
        ]
        chunks = FixedSizeChunker(chunk_size_tokens=200, chunk_overlap_tokens=0).chunk(elements)

        assert [c.section_path for c in chunks] == [
            ["Leave", "Annual"],
            ["Leave", "Sick"],
        ]

    def test_boilerplate_elements_are_excluded(self) -> None:
        elements = [
            make_element("f1", "Confidential - Page 1 of 900", is_boilerplate=True),
            make_element("p1", "Real policy content that should survive.", sequence_index=1),
        ]
        chunks = FixedSizeChunker(chunk_size_tokens=200, chunk_overlap_tokens=0).chunk(elements)

        assert len(chunks) == 1
        assert "Confidential" not in chunks[0].content
        assert chunks[0].element_ids == ["p1"]

    def test_table_becomes_its_own_chunk_and_is_never_split(self) -> None:
        markdown = "| Grade | Days |\n|---|---|\n" + "".join(
            f"| G{i} | {i} |\n" for i in range(200)
        )
        elements = [
            make_element("p1", "Leave entitlement by grade:", sequence_index=0),
            make_element(
                "t1",
                "table",
                element_type="table",
                sequence_index=1,
                table_data={"markdown": markdown},
            ),
            make_element("p2", "Grades above G10 are negotiated individually.", sequence_index=2),
        ]
        chunks = FixedSizeChunker(chunk_size_tokens=100, chunk_overlap_tokens=0).chunk(elements)

        table_chunks = [c for c in chunks if c.chunk_type is ChunkType.TABLE]
        assert len(table_chunks) == 1
        # Whole table in one chunk despite far exceeding the token budget.
        assert table_chunks[0].content == markdown.strip()
        assert table_chunks[0].element_ids == ["t1"]

    def test_long_text_splits_into_multiple_chunks_with_overlap(self) -> None:
        sentences = " ".join(f"Policy clause number {i} applies to all staff." for i in range(80))
        elements = [make_element("p1", sentences, sequence_index=0)]
        chunker = FixedSizeChunker(chunk_size_tokens=100, chunk_overlap_tokens=30)
        chunks = chunker.chunk(elements)

        assert len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
        for chunk in chunks:
            assert chunk.token_count <= 200  # budget plus overlap headroom

    def test_rechunking_produces_identical_ids(self) -> None:
        elements = [
            make_element("h1", "Leave Policy", "heading", sequence_index=0, heading_level=1),
            make_element("p1", "Employees receive 21 days annual leave.", sequence_index=1),
            make_element("p2", "Carry-forward is capped at 5 days.", sequence_index=2),
        ]
        chunker = FixedSizeChunker(chunk_size_tokens=64, chunk_overlap_tokens=8)

        first = [c.identity(VERSION_ID, "fixed-v1") for c in chunker.chunk(elements)]
        second = [c.identity(VERSION_ID, "fixed-v1") for c in chunker.chunk(list(elements))]

        assert first and first == second

    def test_body_text_misclassified_as_heading_is_still_indexed(self) -> None:
        """Regression: real parsers emit body paragraphs typed as headings.

        Measured on the HR corpus: 137 of 194 'level 3 headings' were body text,
        one of 3309 characters. Trusting the type moved that content into
        `section_path`, where it is never retrieved.
        """
        body = (
            "Overtime for non-exempt employees is any time worked in excess of 40 hours "
            "in a seven-day work cycle. Overtime does not result until after 40 hours "
            "have been exceeded in a work cycle, and must be approved in advance."
        )
        elements = [
            make_element("h1", "Administration", "heading", sequence_index=0, heading_level=1),
            make_element("h2", body, "heading", sequence_index=1, heading_level=3),
        ]
        chunks = FixedSizeChunker(chunk_size_tokens=200, chunk_overlap_tokens=0).chunk(elements)

        assert len(chunks) == 1
        # The content is in the chunk body, not hidden in the ancestry.
        assert "40 hours" in chunks[0].content
        assert chunks[0].section_path == ["Administration"]
        assert chunks[0].element_ids == ["h2"]

    def test_short_headings_are_still_treated_as_headings(self) -> None:
        elements = [
            make_element("h1", "Leave Policy", "heading", sequence_index=0, heading_level=1),
            make_element("p1", "Employees receive 21 days of annual leave.", sequence_index=1),
        ]
        chunks = FixedSizeChunker(chunk_size_tokens=200, chunk_overlap_tokens=0).chunk(elements)

        assert chunks[0].section_path == ["Leave Policy"]
        assert "Leave Policy" not in chunks[0].content

    def test_skipped_heading_levels_produce_no_empty_path_segments(self) -> None:
        """H1 -> H3 is common in real documents and must not yield '' segments."""
        elements = [
            make_element("h1", "Benefits", "heading", sequence_index=0, heading_level=1),
            make_element("h2", "Dental Cover", "heading", sequence_index=1, heading_level=3),
            make_element("p1", "Dental cover includes two annual check-ups.", sequence_index=2),
        ]
        chunks = FixedSizeChunker(chunk_size_tokens=200, chunk_overlap_tokens=0).chunk(elements)

        assert chunks[0].section_path == ["Benefits", "Dental Cover"]
        assert "" not in chunks[0].section_path

    def test_section_path_depth_is_bounded(self) -> None:
        elements = [
            make_element(f"h{i}", f"Level {i}", "heading", sequence_index=i, heading_level=i)
            for i in range(1, 12)
        ]
        elements.append(make_element("p1", "Some policy body text here.", sequence_index=20))
        chunks = FixedSizeChunker(chunk_size_tokens=200, chunk_overlap_tokens=0).chunk(elements)

        assert len(chunks[0].section_path) <= FixedSizeChunker.MAX_SECTION_PATH_DEPTH

    def test_overlap_never_pushes_a_chunk_past_the_size_budget(self) -> None:
        """Regression: a single oversized trailing piece was carried whole."""
        # Long sentences, each near the budget, so the tail piece is large.
        sentences = " ".join(
            f"Clause {i} states that all employees must comply with the applicable "
            f"provisions of this policy without exception in all circumstances."
            for i in range(40)
        )
        elements = [make_element("p1", sentences, sequence_index=0)]
        size, overlap = 100, 40
        chunks = FixedSizeChunker(chunk_size_tokens=size, chunk_overlap_tokens=overlap).chunk(
            elements
        )

        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.token_count <= size + overlap, (
                f"chunk {chunk.chunk_index} has {chunk.token_count} tokens, "
                f"budget is {size} + {overlap} overlap"
            )

    def test_empty_and_whitespace_elements_produce_no_chunks(self) -> None:
        elements = [
            make_element("p1", "", sequence_index=0),
            make_element("p2", "   \n  ", sequence_index=1),
        ]
        assert FixedSizeChunker().chunk(elements) == []

    def test_elements_are_processed_in_sequence_order_not_list_order(self) -> None:
        elements = [
            make_element("p2", "Second paragraph content.", sequence_index=2),
            make_element("p1", "First paragraph content.", sequence_index=1),
        ]
        chunks = FixedSizeChunker(chunk_size_tokens=200, chunk_overlap_tokens=0).chunk(elements)
        assert chunks[0].element_ids == ["p1", "p2"]


class TestUnionBoundingBox:
    def test_returns_none_without_usable_boxes(self) -> None:
        assert union_bounding_box([]) is None
        assert union_bounding_box([{"x0": 1}]) is None

    def test_unions_boxes_on_the_dominant_page(self) -> None:
        boxes = [
            {"x0": 10, "y0": 10, "x1": 50, "y1": 20, "page_number": 1, "unit": "pt"},
            {"x0": 5, "y0": 30, "x1": 60, "y1": 40, "page_number": 1, "unit": "pt"},
            {"x0": 0, "y0": 0, "x1": 500, "y1": 500, "page_number": 2, "unit": "pt"},
        ]
        result = union_bounding_box(boxes)
        assert result == {
            "x0": 5.0,
            "y0": 10.0,
            "x1": 60.0,
            "y1": 40.0,
            "page_number": 1,
            "unit": "pt",
        }
