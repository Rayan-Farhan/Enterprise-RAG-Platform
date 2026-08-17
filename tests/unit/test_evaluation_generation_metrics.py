"""Layer 1 deterministic generation metrics (Task 4.3)."""

from __future__ import annotations

import uuid

import pytest

from app.evaluation.metrics.generation import normalize_text, score_deterministic
from app.evaluation.schemas import (
    DatasetSplit,
    Difficulty,
    ExpectedEvidence,
    GoldenQuestion,
    QuestionType,
)
from app.generation.citation import SupportState
from app.generation.service import AnswerResult
from app.retrieval.schemas import Citation

DOCUMENT_ID = uuid.uuid4()
OTHER_DOCUMENT_ID = uuid.uuid4()
VERSION_ID = uuid.uuid4()


def citation(**overrides: object) -> Citation:
    payload: dict[str, object] = {
        "marker": "[1]",
        "document_id": DOCUMENT_ID,
        "version_id": VERSION_ID,
        "chunk_id": uuid.uuid4(),
        "page_number": 14,
        "section_path": ["Leave Policy", "Annual Leave"],
        "element_ids": ["el-0001"],
    }
    payload.update(overrides)
    return Citation(**payload)  # type: ignore[arg-type]


def question(**overrides: object) -> GoldenQuestion:
    payload: dict[str, object] = {
        "question_id": "dev-factual-0001",
        "question": "How much annual leave is accrued?",
        "question_type": QuestionType.FACTUAL,
        "difficulty": Difficulty.EASY,
        "split": DatasetSplit.DEV,
        "expected_evidence": [
            ExpectedEvidence(
                document_id=DOCUMENT_ID,
                version_id=VERSION_ID,
                element_ids=["el-0001"],
                page_numbers=[14],
                section_path=["Leave Policy", "Annual Leave"],
            )
        ],
        "acceptable_answer": "20 days per year.",
        "required_citations": 1,
    }
    payload.update(overrides)
    return GoldenQuestion(**payload)  # type: ignore[arg-type]


def answer(**overrides: object) -> AnswerResult:
    payload: dict[str, object] = {
        "query": "How much annual leave is accrued?",
        "answer": "Full-time employees accrue 20 days per year [1].",
        "support": SupportState.GROUNDED,
        "citations": [citation()],
    }
    payload.update(overrides)
    return AnswerResult(**payload)  # type: ignore[arg-type]


class TestNormalizeText:
    def test_repairs_the_extraction_defects_that_would_fake_a_failure(self) -> None:
        # The parser emits U+FFFD for smart quotes and PDFs mix dash characters.
        # Without repair, an exact-match check would score an extraction defect
        # as a generation failure.
        assert normalize_text("one‑and‑one�half   the  RATE") == normalize_text(
            "one-and-one'half the rate"
        )

    def test_collapses_whitespace_and_casefolds(self) -> None:
        assert normalize_text("  Three\n\tMonths ") == "three months"


class TestProvenanceMatching:
    def test_all_matches_when_the_citation_points_at_the_expected_element(self) -> None:
        scores = score_deterministic(question(), answer())
        assert scores.document_match == 1.0
        assert scores.element_match == 1.0
        assert scores.page_match == 1.0
        assert scores.section_match == 1.0

    def test_right_document_wrong_element_separates_the_two_signals(self) -> None:
        # The whole point of scoring these independently: a citation can land in
        # the correct document and still point at the wrong passage.
        scores = score_deterministic(
            question(), answer(citations=[citation(element_ids=["el-9999"], page_number=41)])
        )
        assert scores.document_match == 1.0
        assert scores.element_match == 0.0
        assert scores.page_match == 0.0

    def test_wrong_document_scores_zero(self) -> None:
        scores = score_deterministic(
            question(), answer(citations=[citation(document_id=OTHER_DOCUMENT_ID)])
        )
        assert scores.document_match == 0.0

    def test_section_match_ignores_case_and_spacing(self) -> None:
        scores = score_deterministic(
            question(), answer(citations=[citation(section_path=["  ANNUAL   LEAVE "])])
        )
        assert scores.section_match == 1.0

    def test_no_citations_scores_every_provenance_check_zero(self) -> None:
        scores = score_deterministic(question(), answer(citations=[]))
        assert scores.document_match == 0.0
        assert scores.element_match == 0.0
        assert scores.required_citations_met == 0.0


class TestAbstention:
    def test_correct_abstention_on_a_negative_question(self) -> None:
        negative = question(
            question_id="dev-negative-0001",
            question_type=QuestionType.NEGATIVE_UNSUPPORTED,
            expected_evidence=[],
            required_citations=0,
        )
        scores = score_deterministic(
            negative,
            answer(answer="The knowledge base does not cover this.", abstained=True, citations=[]),
        )
        assert scores.abstention_correct == 1.0

    def test_answering_a_negative_question_is_scored_wrong(self) -> None:
        negative = question(
            question_id="dev-negative-0001",
            question_type=QuestionType.NEGATIVE_UNSUPPORTED,
            expected_evidence=[],
            required_citations=0,
        )
        scores = score_deterministic(negative, answer(abstained=False))
        assert scores.abstention_correct == 0.0

    def test_abstaining_on_an_answerable_question_is_scored_wrong(self) -> None:
        scores = score_deterministic(question(), answer(abstained=True, citations=[]))
        assert scores.abstention_correct == 0.0
        assert scores.answered_when_expected == 0.0

    def test_provenance_is_neutral_not_zero_for_abstention_cases(self) -> None:
        # Scoring an abstention 0.0 on document_match would make a dataset with
        # more good negative questions look like a worse system.
        negative = question(
            question_id="dev-adversarial-0001",
            question_type=QuestionType.ADVERSARIAL,
            expected_evidence=[],
            required_citations=0,
        )
        scores = score_deterministic(negative, answer(abstained=True, citations=[]))
        assert scores.document_match == 1.0
        assert scores.element_match == 1.0


class TestExactMatchAndIntegrity:
    def test_exact_match_is_none_when_nothing_was_required(self) -> None:
        # None, not 0.0 — a question with no exact-match expectation must not
        # count as having failed one.
        assert score_deterministic(question(), answer()).exact_match is None
        assert "exact_match" not in score_deterministic(question(), answer()).as_metrics()

    def test_exact_match_requires_every_substring(self) -> None:
        strict = question(must_contain=["20 days", "per year"])
        assert score_deterministic(strict, answer()).exact_match == 1.0
        assert (
            score_deterministic(strict, answer(answer="Employees accrue 20 days [1].")).exact_match
            == 0.0
        )

    def test_fabricated_markers_break_citation_integrity(self) -> None:
        scores = score_deterministic(question(), answer(fabricated_markers=["[7]"]))
        assert scores.citation_integrity == 0.0
        assert scores.fabricated_citation_count == 1

    def test_required_citation_count_is_enforced(self) -> None:
        strict = question(required_citations=2)
        assert score_deterministic(strict, answer()).required_citations_met == 0.0
        assert (
            score_deterministic(
                strict, answer(citations=[citation(), citation(marker="[2]")])
            ).required_citations_met
            == 1.0
        )

    def test_metrics_mapping_is_all_floats(self) -> None:
        metrics = score_deterministic(question(), answer()).as_metrics()
        assert all(isinstance(value, float) for value in metrics.values())
        assert metrics["citation_count"] == pytest.approx(1.0)
