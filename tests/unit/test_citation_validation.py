"""Unit tests for citation validation and support classification (Task 3.5, ADR-025/026)."""

from __future__ import annotations

import uuid

import pytest

from app.core.config import AppSettings
from app.generation.citation import CitationValidator, SupportState
from app.generation.context import AssembledContext, ContextAssembler
from app.retrieval.schemas import RetrievedChunk

DOC_ID = uuid.UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
VER_ID = uuid.UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")


def make_chunk(content: str, index: int = 0, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=DOC_ID,
        version_id=VER_ID,
        content=content,
        score=score,
        chunk_index=index,
        page_number=index + 1,
        element_ids=[f"e{index}"],
        document_title="Staff Handbook",
        version_number=1,
    )


@pytest.fixture
def context() -> AssembledContext:
    """A context supplying exactly two evidence blocks: markers [1] and [2]."""
    settings = AppSettings(APP_ENV="testing")
    return ContextAssembler(settings).assemble(
        query="How many annual leave days do I get?",
        chunks=[
            make_chunk("Employees receive 21 days of annual leave per year.", 0, 0.95),
            make_chunk("Carry-forward of unused leave is capped at 5 days.", 1, 0.80),
        ],
    )


@pytest.fixture
def empty_context() -> AssembledContext:
    return ContextAssembler(AppSettings(APP_ENV="testing")).assemble_abstention("q")


@pytest.fixture
def validator() -> CitationValidator:
    return CitationValidator()


class TestValidCitations:
    def test_grounded_answer_keeps_its_citations(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "Employees receive 21 days of annual leave [1].\n\nSUPPORT: grounded"
        result = validator.validate(raw, context)

        assert result.support is SupportState.GROUNDED
        assert [c.marker for c in result.citations] == ["1"]
        assert result.fabricated_markers == []
        assert result.is_valid
        assert "SUPPORT:" not in result.answer

    def test_multiple_markers_all_resolve(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = (
            "You receive 21 days of annual leave [1]. Unused days carry forward "
            "up to a cap of 5 days [2].\n\nSUPPORT: grounded"
        )
        result = validator.validate(raw, context)
        assert {c.marker for c in result.citations} == {"1", "2"}
        assert result.support is SupportState.GROUNDED

    def test_grouped_marker_syntax_is_expanded(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "Leave rules are defined together [1, 2] for all staff.\n\nSUPPORT: grounded"
        result = validator.validate(raw, context)
        assert {c.marker for c in result.citations} == {"1", "2"}

    def test_citations_resolve_to_real_elements(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "Annual leave is 21 days [1].\n\nSUPPORT: grounded"
        result = validator.validate(raw, context)

        supplied = {c.chunk_id for c in context.citations}
        for citation in result.citations:
            assert citation.chunk_id in supplied
            assert citation.element_ids
            assert citation.page_number >= 1


class TestFabricatedCitations:
    def test_fabricated_marker_is_detected_and_stripped(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = (
            "Annual leave is 21 days [1]. Housing allowance is also provided [7].\n\n"
            "SUPPORT: grounded"
        )
        result = validator.validate(raw, context)

        assert result.fabricated_markers == ["7"]
        assert "[7]" not in result.answer
        assert "[1]" in result.answer
        assert [c.marker for c in result.citations] == ["1"]
        # Fabrication downgrades support; it can never remain "grounded".
        assert result.support is SupportState.PARTIAL
        assert not result.is_valid

    def test_only_fabricated_citations_rejects_the_answer(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "The company provides housing to all employees [9].\n\nSUPPORT: grounded"
        result = validator.validate(raw, context)

        assert result.rejected
        assert result.rejection_reason is not None
        assert result.citations == []
        assert result.support is SupportState.INSUFFICIENT

    def test_answer_with_evidence_but_no_markers_is_rejected(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "Employees receive 21 days of annual leave per year.\n\nSUPPORT: grounded"
        result = validator.validate(raw, context)

        assert result.rejected
        assert result.citations == []
        assert result.support is SupportState.INSUFFICIENT

    def test_honest_abstention_with_evidence_present_is_not_rejected(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        """Declaring `insufficient` is the desired behaviour, not a failure."""
        raw = (
            "The retrieved policy text does not state whether housing is provided.\n\n"
            "SUPPORT: insufficient"
        )
        result = validator.validate(raw, context)

        assert not result.rejected
        assert result.support is SupportState.INSUFFICIENT
        assert result.declared_support is SupportState.INSUFFICIENT

    def test_grouped_marker_keeps_valid_and_drops_fabricated(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "Leave entitlements are described here [1, 8].\n\nSUPPORT: grounded"
        result = validator.validate(raw, context)

        assert result.fabricated_markers == ["8"]
        assert "[1]" in result.answer
        assert "8" not in result.answer.split("SUPPORT")[0].replace("[1]", "")
        assert [c.marker for c in result.citations] == ["1"]

    def test_fabricated_markers_are_reported_in_numeric_order(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "Claims [12] and [3] and [1] appear here.\n\nSUPPORT: grounded"
        result = validator.validate(raw, context)
        assert result.fabricated_markers == ["3", "12"]


class TestSupportState:
    def test_no_evidence_always_yields_insufficient(
        self, validator: CitationValidator, empty_context: AssembledContext
    ) -> None:
        raw = "The knowledge base has no answer to this.\n\nSUPPORT: insufficient"
        result = validator.validate(raw, empty_context)
        assert result.support is SupportState.INSUFFICIENT
        assert result.citations == []
        assert not result.rejected

    def test_model_cannot_upgrade_its_own_support_claim(
        self, validator: CitationValidator, empty_context: AssembledContext
    ) -> None:
        # No evidence supplied, yet the model claims to be grounded.
        raw = "Annual leave is 30 days [1].\n\nSUPPORT: grounded"
        result = validator.validate(raw, empty_context)
        assert result.support is SupportState.INSUFFICIENT

    def test_model_may_downgrade_its_own_support_claim(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "Leave is 21 days [1], but the carry-forward rule is unclear.\n\nSUPPORT: partial"
        result = validator.validate(raw, context)
        assert result.support is SupportState.PARTIAL

    def test_uncited_factual_sentence_downgrades_to_partial(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = (
            "Employees receive 21 days of annual leave [1]. "
            "Employees are additionally granted a housing stipend every quarter.\n\n"
            "SUPPORT: grounded"
        )
        result = validator.validate(raw, context)

        assert result.support is SupportState.PARTIAL
        assert len(result.uncited_sentences) == 1
        assert "housing stipend" in result.uncited_sentences[0]

    def test_missing_support_line_is_computed(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        result = validator.validate("Annual leave is 21 days [1].", context)
        assert result.support is SupportState.GROUNDED

    def test_short_fragments_do_not_count_as_uncited_claims(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "21 days. Employees receive 21 days of annual leave [1].\n\nSUPPORT: grounded"
        result = validator.validate(raw, context)
        assert result.uncited_sentences == []
        assert result.support is SupportState.GROUNDED

    def test_markdown_scaffolding_is_not_an_uncited_claim(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = (
            "## Annual leave summary that runs long enough to pass the length filter\n"
            "- A bullet item that is also long enough to exceed the length threshold\n"
            "Employees receive 21 days of annual leave [1].\n\nSUPPORT: grounded"
        )
        result = validator.validate(raw, context)
        assert result.support is SupportState.GROUNDED


class TestSupportLineExtraction:
    @pytest.mark.parametrize("declared", ["grounded", "GROUNDED", "Grounded"])
    def test_support_line_is_case_insensitive(
        self, validator: CitationValidator, context: AssembledContext, declared: str
    ) -> None:
        result = validator.validate(f"Leave is 21 days [1].\n\nSUPPORT: {declared}", context)
        assert result.support is SupportState.GROUNDED

    def test_fenced_support_line_is_removed(
        self, validator: CitationValidator, context: AssembledContext
    ) -> None:
        raw = "Leave is 21 days [1].\n\n```\nSUPPORT: grounded\n```"
        result = validator.validate(raw, context)
        assert "SUPPORT" not in result.answer
        assert "```" not in result.answer
        assert result.support is SupportState.GROUNDED
