"""Golden dataset and experiment data contracts (Task 4.1, ADR-028/029).

The dataset is the measuring instrument for every stage from 5 onward, so its
schema is strict on purpose. A question whose ``expected_evidence`` does not
resolve to real elements silently deflates recall for every future experiment
and is indistinguishable from a genuine retrieval failure — which is why
:mod:`app.evaluation.dataset` refuses to load one.
"""

from __future__ import annotations

import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class QuestionType(StrEnum):
    """The ten master §49 question types the dataset must represent.

    Several are expected to fail at Stage 4 — that is the point. ``MULTIMODAL``
    is unanswerable until Stage 9, ``TEMPORAL`` and ``CONFLICTING_VERSIONS``
    until Stage 10's versioned retrieval, ``MULTI_HOP`` until Stage 10's
    decomposition. Recording them now means the improvement is measured rather
    than asserted.
    """

    FACTUAL = "factual"
    EXACT_RETRIEVAL = "exact_retrieval"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"
    NEGATIVE_UNSUPPORTED = "negative_unsupported"
    TEMPORAL = "temporal"
    CONFLICTING_VERSIONS = "conflicting_versions"
    CALCULATION = "calculation"
    MULTIMODAL = "multimodal"
    ADVERSARIAL = "adversarial"


class Difficulty(StrEnum):
    """Author-assigned difficulty, used to slice metrics rather than to weight them."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DatasetSplit(StrEnum):
    """Dataset splits with distinct, non-interchangeable purposes.

    ``TEST`` is locked: it is opened exactly once, at Stage 14. Tuning against it
    would make the final readiness number a measurement of the tuning rather than
    of the system, so the loader requires an explicit unlock to read it.
    """

    DEV = "dev"
    VALIDATION = "validation"
    TEST = "test"


#: Question types that require no supporting evidence — the system is expected to
#: abstain rather than retrieve. Their ``expected_evidence`` must be empty, and
#: retrieval metrics are undefined for them (they are scored on abstention).
ABSTENTION_TYPES: frozenset[QuestionType] = frozenset(
    {QuestionType.NEGATIVE_UNSUPPORTED, QuestionType.ADVERSARIAL}
)


class ExpectedEvidence(BaseModel):
    """A pointer to the canonical evidence that answers a question (ADR-005).

    Evidence is recorded at element granularity rather than chunk granularity
    because chunk IDs are derived from ``CHUNKING_VERSION`` (ADR-036). Stage 5
    re-chunks the entire corpus; a dataset keyed on chunk IDs would be
    invalidated by the very experiment it exists to measure. Elements are stable
    across re-chunking, so recall can be computed by mapping retrieved chunks
    back to their source elements.
    """

    document_id: uuid.UUID
    version_id: uuid.UUID
    element_ids: list[str] = Field(
        min_length=1,
        description="Canonical element_ids (not chunk IDs) that contain the answer",
    )
    page_numbers: list[int] = Field(default_factory=list)
    section_path: list[str] = Field(default_factory=list)
    document_title: str | None = None
    quote: str | None = Field(
        default=None,
        description="Verbatim excerpt from the element, used by Layer 1 exact-match checks",
    )


class GoldenQuestion(BaseModel):
    """One curated question with its expected evidence and answer (master §48)."""

    model_config = {"extra": "forbid"}

    question_id: str = Field(min_length=1, description="Stable ID, e.g. dev-factual-0007")
    question: str = Field(min_length=5)
    question_type: QuestionType
    difficulty: Difficulty = Difficulty.MEDIUM
    split: DatasetSplit

    expected_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    acceptable_answer: str = Field(
        min_length=1,
        description="A reference answer; judged for equivalence, never string-compared",
    )
    acceptable_answer_variants: list[str] = Field(default_factory=list)
    required_citations: int = Field(
        default=0,
        ge=0,
        description="Minimum number of resolvable citations a correct answer must carry",
    )
    must_contain: list[str] = Field(
        default_factory=list,
        description="Substrings a correct answer must contain (exact-retrieval checks)",
    )
    must_abstain: bool = Field(
        default=False,
        description="True when the only correct behaviour is refusing to answer",
    )

    # Provenance of the question itself, so a bad generator run is traceable.
    source: str = Field(default="hand", description="hand | llm-drafted | imported")
    author_model: str | None = None
    notes: str | None = None
    expected_to_fail_until_stage: int | None = Field(
        default=None,
        description="Roadmap stage that is expected to make this question pass",
    )

    @field_validator("question_id")
    @classmethod
    def _no_whitespace_in_id(cls, v: str) -> str:
        if v != v.strip() or " " in v:
            raise ValueError(f"question_id must not contain whitespace: {v!r}")
        return v

    @model_validator(mode="after")
    def _check_evidence_consistency(self) -> GoldenQuestion:
        """Keep evidence expectations and abstention expectations from contradicting.

        Both directions are real authoring mistakes. A negative question carrying
        evidence would count a correct abstention as a recall miss; an answerable
        question with no evidence would count every retrieval as perfect.
        """
        is_abstention_type = self.question_type in ABSTENTION_TYPES

        if is_abstention_type and self.expected_evidence:
            raise ValueError(
                f"{self.question_id}: {self.question_type} questions must have no "
                f"expected_evidence — they are scored on abstention, not recall"
            )
        if self.must_abstain and self.expected_evidence:
            raise ValueError(
                f"{self.question_id}: must_abstain is set but expected_evidence is present"
            )
        if self.must_abstain and self.required_citations:
            raise ValueError(
                f"{self.question_id}: an abstention cannot require {self.required_citations} "
                f"citations"
            )
        if not is_abstention_type and not self.must_abstain and not self.expected_evidence:
            raise ValueError(
                f"{self.question_id}: answerable questions must declare expected_evidence"
            )
        return self

    @property
    def is_abstention_case(self) -> bool:
        """True when the correct behaviour is to refuse rather than answer."""
        return self.must_abstain or self.question_type in ABSTENTION_TYPES

    def expected_element_ids(self) -> set[str]:
        """Every element ID that counts as relevant for this question."""
        return {eid for ev in self.expected_evidence for eid in ev.element_ids}

    def expected_document_ids(self) -> set[uuid.UUID]:
        """Every document that contains part of the expected evidence."""
        return {ev.document_id for ev in self.expected_evidence}


class DatasetStats(BaseModel):
    """Composition of a loaded split, reported by the validator and the README."""

    split: DatasetSplit
    version: str
    total: int
    by_type: dict[str, int] = Field(default_factory=dict)
    by_difficulty: dict[str, int] = Field(default_factory=dict)
    by_source: dict[str, int] = Field(default_factory=dict)
    abstention_cases: int = 0
    documents_covered: int = 0
    missing_types: list[str] = Field(default_factory=list)


class EvidenceResolutionIssue(BaseModel):
    """One unresolvable evidence pointer found during validation."""

    question_id: str
    reason: str
    detail: dict[str, Any] = Field(default_factory=dict)


class DatasetValidationReport(BaseModel):
    """Outcome of validating a split against the live corpus."""

    stats: DatasetStats
    issues: list[EvidenceResolutionIssue] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Valid only when every pointer resolves and every type is represented."""
        return not self.issues and not self.stats.missing_types
