"""The four chunking strategies behind one interface (Task 5.1, ADR-006).

Every strategy is held to the same contract — swappable by configuration, intact
heading ancestry, deterministic identity, tables never split mid-row — because
Task 5.3 compares them against each other and a difference in any of those would
show up as a metric difference that has nothing to do with chunking quality.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.core.config import AppSettings
from app.db.models.element import Element
from app.ingestion.chunking.base import (
    ChunkCandidate,
    ChunkingContext,
    ChunkingStrategy,
    ChunkType,
    estimate_tokens,
)
from app.ingestion.chunking.contextual import ContextualChunker
from app.ingestion.chunking.fixed_size import FixedSizeChunker
from app.ingestion.chunking.hierarchical import HierarchicalChunker
from app.ingestion.chunking.hierarchical_contextual import HierarchicalContextualChunker
from app.ingestion.chunking.segmentation import segment
from app.ingestion.chunking.service import STRATEGIES, build_strategy
from app.ingestion.chunking.structure_aware import StructureAwareChunker

VERSION_ID = uuid.UUID("5e68ca81-3d8b-41ea-a7cd-d0a6aa73dd91")

ALL_STRATEGIES = (
    FixedSizeChunker,
    StructureAwareChunker,
    HierarchicalChunker,
    ContextualChunker,
    HierarchicalContextualChunker,
)


def element(
    element_id: str,
    text: str,
    element_type: str = "paragraph",
    sequence_index: int = 0,
    page_number: int = 1,
    heading_level: int | None = None,
    is_boilerplate: bool = False,
    table_data: dict[str, Any] | None = None,
) -> Element:
    return Element(
        id=uuid.uuid4(),
        version_id=VERSION_ID,
        page_id=uuid.uuid4(),
        page_number=page_number,
        element_id=element_id,
        element_type=element_type,
        sequence_index=sequence_index,
        text_content=text,
        content_hash="x" * 64,
        is_boilerplate=is_boilerplate,
        table_data=table_data,
        extra_metadata={"heading_level": heading_level} if heading_level else {},
    )


def handbook() -> list[Element]:
    """A small document with two sections, a table, and a boilerplate footer."""
    return [
        element("h1", "Paid and Unpaid Time Off Work", "heading", 0, 1, heading_level=1),
        element("h2", "Sick Leave", "heading", 1, 1, heading_level=2),
        element(
            "p1",
            "Full-time regular employees are entitled to 96 work hours of sick leave each year.",
            "paragraph",
            2,
            1,
        ),
        element(
            "p2",
            "Sick leave may be used for up to 6 weeks of maternity leave.",
            "paragraph",
            3,
            2,
        ),
        element("h3", "Annual Leave", "heading", 4, 2, heading_level=2),
        element(
            "p3",
            "Annual leave accrues for each hour worked, to 80 work hours per year.",
            "paragraph",
            5,
            2,
        ),
        element(
            "t1",
            "accrual table",
            "table",
            6,
            2,
            table_data={"markdown": "| Service | Days |\n| --- | --- |\n| 2 years | 12 |"},
        ),
        element("f1", "Page 2 of 56", "paragraph", 7, 2, is_boilerplate=True),
    ]


def build(cls: type, **kwargs: Any) -> Any:
    return cls(chunk_size_tokens=64, chunk_overlap_tokens=8, **kwargs)


class TestInterfaceConformance:
    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_every_strategy_satisfies_the_protocol(self, cls: type) -> None:
        assert isinstance(build(cls), ChunkingStrategy)

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_every_strategy_accepts_a_context(self, cls: type) -> None:
        # Swappable by configuration alone means one call signature, including
        # for the strategies that ignore the context.
        chunks = build(cls).chunk(handbook(), ChunkingContext(document_title="Staff Handbook"))
        assert chunks

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_every_strategy_names_itself_and_its_version(self, cls: type) -> None:
        strategy = build(cls)
        assert strategy.strategy_name
        assert strategy.chunking_version.startswith(strategy.strategy_name)

    def test_the_registry_covers_exactly_the_configurable_names(self) -> None:
        configurable = set(
            AppSettings.model_fields["CHUNKING_STRATEGY"].annotation.__args__  # type: ignore[union-attr]
        )
        assert set(STRATEGIES) == configurable

    @pytest.mark.parametrize(
        "name",
        ["fixed", "structure_aware", "hierarchical", "contextual", "hierarchical_contextual"],
    )
    def test_build_strategy_constructs_the_configured_one(self, name: str) -> None:
        settings = AppSettings(CHUNKING_STRATEGY=name, CHUNKING_VERSION=f"{name}-v1")

        strategy = build_strategy(settings)

        assert strategy.strategy_name == name
        assert strategy.chunking_version == f"{name}-v1"


class TestSharedContract:
    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_boilerplate_is_excluded(self, cls: type) -> None:
        chunks = build(cls).chunk(handbook())
        assert all("f1" not in c.element_ids for c in chunks)

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_tables_are_emitted_whole_and_never_split(self, cls: type) -> None:
        # The contract is that no chunk holds a *partial* table, not that only
        # one chunk mentions it: hierarchical legitimately carries the table in
        # both the leaf and the section chunk that contains it.
        chunks = build(cls).chunk(handbook())
        referencing = [c for c in chunks if "t1" in c.element_ids]

        assert referencing
        for chunk in referencing:
            assert "| 2 years | 12 |" in chunk.content

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_a_table_is_never_merged_into_a_prose_only_chunk(self, cls: type) -> None:
        chunks = build(cls).chunk(handbook())
        leaves = [c for c in chunks if c.chunk_type is not ChunkType.SECTION]
        table_leaves = [c for c in leaves if "t1" in c.element_ids]

        assert len(table_leaves) == 1
        assert table_leaves[0].chunk_type is ChunkType.TABLE

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_chunk_ids_are_deterministic(self, cls: type) -> None:
        first = build(cls).chunk(handbook())
        second = build(cls).chunk(handbook())

        ids_a = [c.identity(VERSION_ID, "x-v1") for c in first]
        ids_b = [c.identity(VERSION_ID, "x-v1") for c in second]

        assert ids_a == ids_b
        assert len(set(ids_a)) == len(ids_a), "chunk identities must be unique within a version"

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_every_chunk_carries_content_and_elements(self, cls: type) -> None:
        for chunk in build(cls).chunk(handbook()):
            assert chunk.content.strip()
            assert chunk.element_ids
            assert chunk.token_count > 0
            assert chunk.primary_page_number in chunk.page_span

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_headings_become_ancestry_not_content(self, cls: type) -> None:
        # A heading is a label. Leaving it as body text would index "Sick Leave"
        # as a chunk of its own; dropping it entirely would lose the path.
        chunks = build(cls).chunk(handbook())
        paths = {tuple(c.section_path) for c in chunks}

        assert ("Paid and Unpaid Time Off Work", "Sick Leave") in paths
        assert ("Paid and Unpaid Time Off Work", "Annual Leave") in paths

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_an_empty_document_yields_no_chunks(self, cls: type) -> None:
        assert build(cls).chunk([]) == []

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_a_document_with_no_headings_still_chunks(self, cls: type) -> None:
        elements = [element("p1", "Policy text without any heading at all.", "paragraph", 0, 1)]

        chunks = build(cls).chunk(elements)

        assert chunks
        assert all(c.section_path == [] for c in chunks)

    @pytest.mark.parametrize("cls", ALL_STRATEGIES)
    def test_rejects_an_overlap_at_or_above_the_chunk_size(self, cls: type) -> None:
        with pytest.raises(ValueError, match="chunk_overlap_tokens"):
            cls(chunk_size_tokens=64, chunk_overlap_tokens=64)


class TestStructureAware:
    def test_a_chunk_never_spans_two_sections(self) -> None:
        # The defect this strategy exists to remove: the fixed baseline packs
        # until a budget is hit, so one chunk can straddle two policies.
        chunks = build(StructureAwareChunker).chunk(handbook())

        for chunk in chunks:
            owners = {
                tuple(block.section_path)
                for section in segment(handbook())
                for block in section.blocks
                if block.element.element_id in chunk.element_ids
            }
            assert len(owners) == 1

    def test_a_large_section_is_split_within_its_own_path(self) -> None:
        body = " ".join(f"Sentence number {i} of the policy." for i in range(80))
        elements = [
            element("h1", "Overtime", "heading", 0, 1, heading_level=1),
            element("p1", body, "paragraph", 1, 1),
        ]

        chunks = build(StructureAwareChunker).chunk(elements)

        assert len(chunks) > 1
        assert all(c.section_path == ["Overtime"] for c in chunks)


class TestHierarchical:
    def test_emits_a_section_chunk_and_leaves_that_point_at_it(self) -> None:
        chunks = build(HierarchicalChunker).chunk(handbook())

        sections = [c for c in chunks if c.chunk_type is ChunkType.SECTION]
        leaves = [c for c in chunks if c.parent_index is not None]

        assert sections
        assert leaves
        for leaf in leaves:
            parent = chunks[leaf.parent_index]  # type: ignore[index]
            assert parent.chunk_type is ChunkType.SECTION
            assert parent.section_path == leaf.section_path

    def test_a_leaf_is_smaller_than_the_nominal_chunk_size(self) -> None:
        # Leaves exist to be matched precisely; the parent supplies context.
        chunker = build(HierarchicalChunker)
        assert chunker.leaf_size_tokens < chunker.chunk_size_tokens

    def test_every_leaf_element_appears_in_its_parent(self) -> None:
        chunks = build(HierarchicalChunker).chunk(handbook())

        for leaf in (c for c in chunks if c.parent_index is not None):
            parent = chunks[leaf.parent_index]  # type: ignore[index]
            assert set(leaf.element_ids) <= set(parent.element_ids)

    def test_an_oversized_section_is_abridged_and_says_so(self) -> None:
        # Silently truncating would let the generator answer "the policy does not
        # say" from an artefact of chunking.
        body = " ".join(f"Clause {i} of a very long section." for i in range(400))
        elements = [
            element("h1", "Benefits", "heading", 0, 1, heading_level=1),
            element("p1", body, "paragraph", 1, 1),
        ]

        chunks = build(HierarchicalChunker).chunk(elements)
        section = next(c for c in chunks if c.chunk_type is ChunkType.SECTION)

        assert "continues beyond this excerpt" in section.content
        assert section.token_count <= build(HierarchicalChunker).section_budget_tokens * 1.2

    def test_rejects_a_section_budget_multiplier_below_one(self) -> None:
        with pytest.raises(ValueError, match="section_budget_multiplier"):
            HierarchicalChunker(
                chunk_size_tokens=64, chunk_overlap_tokens=8, section_budget_multiplier=0
            )


class TestContextual:
    def test_prefixes_the_document_title_and_section_path(self) -> None:
        chunks = build(ContextualChunker).chunk(
            handbook(), ChunkingContext(document_title="Staff Handbook")
        )
        body = next(c for c in chunks if "p1" in c.element_ids)

        assert body.content.startswith("Document: Staff Handbook | Section:")
        assert "Sick Leave" in body.content.split("\n\n")[0]

    def test_the_prefix_distinguishes_documents_that_state_the_same_policy(self) -> None:
        # The corpus carries three documents restating the same rules; without a
        # prefix their chunks embed almost identically.
        staff = build(ContextualChunker).chunk(
            handbook(), ChunkingContext(document_title="Staff Handbook")
        )
        faculty = build(ContextualChunker).chunk(
            handbook(), ChunkingContext(document_title="Faculty Handbook")
        )

        assert staff[0].content != faculty[0].content

    def test_omits_parts_the_document_does_not_have(self) -> None:
        chunks = build(ContextualChunker).chunk(handbook(), ChunkingContext())

        assert not any(c.content.startswith("Document:") for c in chunks)
        assert not any("None" in c.content.split("\n\n")[0] for c in chunks)

    def test_metadata_fields_are_included_when_present(self) -> None:
        chunks = build(ContextualChunker).chunk(
            handbook(),
            ChunkingContext(
                document_title="Staff Handbook",
                metadata={"effective_date": "2026-02-04", "version_label": "v1"},
            ),
        )

        assert "Effective Date: 2026-02-04" in chunks[0].content
        assert "Version Label: v1" in chunks[0].content

    def test_token_count_reflects_the_prefix(self) -> None:
        # Otherwise the recorded count understates what is actually embedded.
        chunks = build(ContextualChunker).chunk(
            handbook(), ChunkingContext(document_title="Staff Handbook")
        )

        for chunk in chunks:
            assert chunk.token_count == estimate_tokens(chunk.content)

    def test_the_prefix_is_reserved_from_the_body_budget(self) -> None:
        # Measured regression: without reserving, the prefix pushed 147 of 826
        # chunks over budget on the real corpus, so the strategy was effectively
        # being compared at a larger chunk size than the others — crediting the
        # prefix for gains that actually came from carrying more text.
        body = " ".join(f"Sentence number {i} of the leave policy." for i in range(200))
        elements = [
            element("h1", "Paid and Unpaid Time Off Work", "heading", 0, 1, heading_level=1),
            element("h2", "Sick Leave and Related Absence Provisions", "heading", 1, 1, heading_level=2),
            element("p1", body, "paragraph", 2, 1),
        ]
        context = ChunkingContext(
            document_title="University Employee Policy Manual and Handbook",
            metadata={"effective_date": "2026-02-18", "version_label": "v1"},
        )

        chunks = ContextualChunker(chunk_size_tokens=128, chunk_overlap_tokens=8).chunk(
            elements, context
        )
        prose = [c for c in chunks if c.chunk_type is not ChunkType.TABLE]

        assert prose
        for chunk in prose:
            assert chunk.token_count <= 128, (
                f"prefix pushed a chunk to {chunk.token_count} tokens against a 128 budget"
            )

    def test_a_longer_prefix_leaves_less_room_for_body_text(self) -> None:
        body = " ".join(f"Sentence number {i} of the leave policy." for i in range(200))
        elements = [
            element("h1", "Leave", "heading", 0, 1, heading_level=1),
            element("p1", body, "paragraph", 1, 1),
        ]

        short = ContextualChunker(chunk_size_tokens=128, chunk_overlap_tokens=8).chunk(
            elements, ChunkingContext(document_title="X")
        )
        long = ContextualChunker(chunk_size_tokens=128, chunk_overlap_tokens=8).chunk(
            elements,
            ChunkingContext(
                document_title="University Employee Policy Manual and Handbook",
                metadata={"effective_date": "2026-02-18", "version_label": "v1"},
            ),
        )

        # More prefix means less body per chunk, hence more chunks for the same text.
        assert len(long) >= len(short)

    def test_splits_identically_to_structure_aware(self) -> None:
        # The two must differ only in the prefix, or Task 5.3 cannot attribute a
        # metric change to the prefix rather than to a different split.
        plain = build(StructureAwareChunker).chunk(handbook())
        prefixed = build(ContextualChunker).chunk(
            handbook(), ChunkingContext(document_title="Staff Handbook")
        )

        assert [c.element_ids for c in plain] == [c.element_ids for c in prefixed]


class TestHierarchicalContextual:
    """The combined strategy: hierarchy and provenance are orthogonal (Task 5.3)."""

    def test_it_keeps_the_hierarchy(self) -> None:
        chunks = build(HierarchicalContextualChunker).chunk(
            handbook(), ChunkingContext(document_title="Staff Handbook")
        )

        sections = [c for c in chunks if c.chunk_type is ChunkType.SECTION]
        leaves = [c for c in chunks if c.parent_index is not None]

        assert sections
        assert leaves
        for leaf in leaves:
            assert chunks[leaf.parent_index].chunk_type is ChunkType.SECTION  # type: ignore[index]

    def test_it_prefixes_every_chunk_including_sections(self) -> None:
        # Section chunks are what parent expansion hands the generator, so an
        # unprefixed section would lose provenance exactly where it matters most.
        chunks = build(HierarchicalContextualChunker).chunk(
            handbook(), ChunkingContext(document_title="Staff Handbook")
        )

        prefixed = [c for c in chunks if c.content.startswith("Document: Staff Handbook")]
        assert len(prefixed) == len(chunks)

    def test_the_prefix_is_reserved_from_both_budgets(self) -> None:
        body = " ".join(f"Sentence number {i} of the leave policy." for i in range(200))
        elements = [
            element("h1", "Paid and Unpaid Time Off Work", "heading", 0, 1, heading_level=1),
            element("p1", body, "paragraph", 1, 1),
        ]
        context = ChunkingContext(document_title="University Employee Policy Manual")

        chunker = HierarchicalContextualChunker(chunk_size_tokens=128, chunk_overlap_tokens=8)
        chunks = chunker.chunk(elements, context)
        leaves = [c for c in chunks if c.chunk_type is not ChunkType.SECTION]

        assert leaves
        for leaf in leaves:
            assert leaf.token_count <= 128

    def test_it_is_distinguishable_from_both_parents(self) -> None:
        # Same corpus, three strategies: identity must not collide, or Stage 5's
        # comparison silently measures a mixture.
        ctx = ChunkingContext(document_title="Staff Handbook")
        combined = build(HierarchicalContextualChunker).chunk(handbook(), ctx)
        hierarchical = build(HierarchicalChunker).chunk(handbook(), ctx)
        contextual = build(ContextualChunker).chunk(handbook(), ctx)

        ids = {
            "combined": {c.identity(VERSION_ID, "hierarchical_contextual-v1") for c in combined},
            "hierarchical": {c.identity(VERSION_ID, "hierarchical-v1") for c in hierarchical},
            "contextual": {c.identity(VERSION_ID, "contextual-v1") for c in contextual},
        }
        assert not (ids["combined"] & ids["hierarchical"])
        assert not (ids["combined"] & ids["contextual"])


class TestStrategyIsolation:
    def test_two_strategies_cannot_share_a_chunking_version(self) -> None:
        # Chunk IDs do not include the strategy name, so a shared version string
        # would let two strategies overwrite each other's rows and make Stage 5's
        # comparison measure a mixture of both.
        with pytest.raises(ValueError, match="CHUNKING_VERSION must start with"):
            AppSettings(CHUNKING_STRATEGY="hierarchical", CHUNKING_VERSION="fixed-v2")

    def test_a_strategy_name_that_prefixes_another_cannot_borrow_its_version(self) -> None:
        # "hierarchical_contextual-s256-o32" also starts with "hierarchical".
        with pytest.raises(ValueError, match="CHUNKING_VERSION must start with"):
            AppSettings(
                CHUNKING_STRATEGY="hierarchical",
                CHUNKING_VERSION="hierarchical_contextual-s256-o32",
            )

    def test_the_same_elements_under_two_strategies_get_different_ids(self) -> None:
        structure = build(StructureAwareChunker).chunk(handbook())
        hierarchical = build(HierarchicalChunker).chunk(handbook())

        structure_ids = {c.identity(VERSION_ID, "structure_aware-v1") for c in structure}
        hierarchical_ids = {c.identity(VERSION_ID, "hierarchical-v1") for c in hierarchical}

        assert not (structure_ids & hierarchical_ids)


class TestFixedBaselineRetained:
    def test_the_baseline_is_still_available_for_comparison(self) -> None:
        # Stage 5's exit gate requires losing strategies stay behind config.
        assert "fixed" in STRATEGIES
        assert build(FixedSizeChunker).chunk(handbook())

    def test_the_baseline_embeds_no_provenance(self) -> None:
        chunks: list[ChunkCandidate] = build(FixedSizeChunker).chunk(
            handbook(), ChunkingContext(document_title="Staff Handbook")
        )

        assert not any("Staff Handbook" in c.content for c in chunks)
