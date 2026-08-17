"""Layer 1 — deterministic generation metrics (Task 4.3, master §52).

These checks never call a model. They answer the questions that have an
objectively correct answer: did the citation point at the right document, the
right page, the right section? Did the system abstain when it should have?

Master §51 is explicit that no single automatic evaluator is trusted. This layer
exists so that the LLM judge in :mod:`app.evaluation.judge` is never the only
thing standing between a hallucination and a green run — a judge that drifts
cannot make these numbers move.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

from app.evaluation.schemas import GoldenQuestion
from app.generation.service import AnswerResult
from app.retrieval.schemas import Citation


@dataclass(frozen=True)
class DeterministicScores:
    """Per-question Layer 1 outcome. Every field is 0.0/1.0 unless noted."""

    question_id: str

    abstention_correct: float
    answered_when_expected: float

    citation_count: int
    required_citations_met: float
    fabricated_citation_count: int
    citation_integrity: float

    document_match: float
    page_match: float
    section_match: float
    element_match: float

    exact_match: float | None
    rejected: float

    def as_metrics(self) -> dict[str, float]:
        """Flatten to the metric mapping the runner aggregates."""
        metrics = {
            "abstention_correct": self.abstention_correct,
            "answered_when_expected": self.answered_when_expected,
            "required_citations_met": self.required_citations_met,
            "citation_integrity": self.citation_integrity,
            "document_match": self.document_match,
            "page_match": self.page_match,
            "section_match": self.section_match,
            "element_match": self.element_match,
            "rejected": self.rejected,
            "citation_count": float(self.citation_count),
            "fabricated_citation_count": float(self.fabricated_citation_count),
        }
        if self.exact_match is not None:
            metrics["exact_match"] = self.exact_match
        return metrics


def normalize_text(text: str) -> str:
    """Casefold, collapse whitespace, and normalise the punctuation extraction mangles.

    The parser emits U+FFFD where smart quotes should be, and PDFs mix hyphens
    with en dashes and non-breaking spaces. Without this, a substring check for
    ``one and one-half`` fails against text that is, to a reader, identical —
    scoring an extraction defect as a generation failure.
    """
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("�", "'")
    for dash in ("‐", "‑", "‒", "–", "—", "−"):
        normalized = normalized.replace(dash, "-")
    for quote in ("‘", "’", "‛", "′"):
        normalized = normalized.replace(quote, "'")
    for quote in ("“", "”", "„", "″"):
        normalized = normalized.replace(quote, '"')
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().casefold()


def _section_tokens(section_path: Sequence[str]) -> set[str]:
    return {normalize_text(part) for part in section_path if part and part.strip()}


def score_deterministic(question: GoldenQuestion, result: AnswerResult) -> DeterministicScores:
    """Compute every Layer 1 check for one answered question."""
    citations: list[Citation] = list(result.citations)
    fabricated = len(result.fabricated_markers)

    # --- Abstention -------------------------------------------------------
    if question.is_abstention_case:
        abstention_correct = 1.0 if result.abstained else 0.0
        answered_when_expected = 1.0  # not applicable; neutral so it never drags the mean
    else:
        abstention_correct = 1.0 if not result.abstained else 0.0
        answered_when_expected = 1.0 if not result.abstained else 0.0

    # --- Citation integrity ----------------------------------------------
    # Fabrication is structurally impossible (Task 3.5), so a non-zero count here
    # is a regression in the validator itself, not a model quality signal.
    citation_integrity = 1.0 if fabricated == 0 else 0.0
    required_met = 1.0 if len(citations) >= question.required_citations else 0.0

    # --- Provenance matching ---------------------------------------------
    if question.is_abstention_case:
        # An abstention has nothing to match against; neutral rather than zero,
        # for the same reason retrieval metrics are omitted for these questions.
        document_match = page_match = section_match = element_match = 1.0
    elif not citations:
        document_match = page_match = section_match = element_match = 0.0
    else:
        expected_documents = question.expected_document_ids()
        expected_elements = question.expected_element_ids()
        expected_pages = {p for ev in question.expected_evidence for p in ev.page_numbers}
        expected_sections = {
            token for ev in question.expected_evidence for token in _section_tokens(ev.section_path)
        }

        document_match = 1.0 if any(c.document_id in expected_documents for c in citations) else 0.0
        element_match = (
            1.0 if any(set(c.element_ids) & expected_elements for c in citations) else 0.0
        )
        page_match = (
            1.0
            if not expected_pages or any(c.page_number in expected_pages for c in citations)
            else 0.0
        )
        section_match = (
            1.0
            if not expected_sections
            or any(_section_tokens(c.section_path) & expected_sections for c in citations)
            else 0.0
        )

    # --- Exact match ------------------------------------------------------
    # None, not 0.0, when the question declares no required substrings: a
    # question that never had an exact-match expectation must not be counted as
    # having failed one.
    exact_match: float | None = None
    if question.must_contain:
        answer_normalized = normalize_text(result.answer)
        exact_match = (
            1.0
            if all(normalize_text(needle) in answer_normalized for needle in question.must_contain)
            else 0.0
        )

    return DeterministicScores(
        question_id=question.question_id,
        abstention_correct=abstention_correct,
        answered_when_expected=answered_when_expected,
        citation_count=len(citations),
        required_citations_met=required_met,
        fabricated_citation_count=fabricated,
        citation_integrity=citation_integrity,
        document_match=document_match,
        page_match=page_match,
        section_match=section_match,
        element_match=element_match,
        exact_match=exact_match,
        rejected=1.0 if result.rejected else 0.0,
    )
