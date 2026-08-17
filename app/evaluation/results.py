"""Experiment result contracts (Task 4.4, ADR-029).

An experiment record has one job: make a number from three months ago
interpretable. That means the configuration, the dataset version, and every
model, prompt and prompt hash involved are stored *with* the metrics, not
alongside them in a commit message. Stage 14 compares against
``experiment-001-baseline`` directly, and by then nobody will remember what
``RETRIEVAL_MIN_SCORE`` was set to.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from app.evaluation.schemas import DatasetSplit, Difficulty, QuestionType


def _now() -> datetime:
    return datetime.now(UTC)


class QuestionResult(BaseModel):
    """Everything one question produced in one experiment run."""

    question_id: str
    question: str
    question_type: QuestionType
    difficulty: Difficulty

    answer: str = ""
    abstained: bool = False
    rejected: bool = False
    support: str | None = None

    retrieved_chunk_ids: list[uuid.UUID] = Field(default_factory=list)
    retrieved_element_ids: list[str] = Field(default_factory=list)
    context_element_ids: list[str] = Field(default_factory=list)
    citation_markers: list[str] = Field(default_factory=list)
    cited_element_ids: list[str] = Field(default_factory=list)

    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    deterministic_metrics: dict[str, float] = Field(default_factory=dict)
    judge_metrics: dict[str, float] = Field(default_factory=dict)
    judge_raw_scores: dict[str, float] = Field(default_factory=dict)
    judge_stdev: dict[str, float] = Field(default_factory=dict)
    judge_reasoning: dict[str, str] = Field(default_factory=dict)
    judge_errors: list[str] = Field(default_factory=list)

    latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    token_counts: dict[str, int] = Field(default_factory=dict)
    evidence_tokens: int = Field(
        default=0,
        description="Context tokens sent to the generator; replayed when a run is resumed",
    )

    error: str | None = Field(
        default=None,
        description="Set when the pipeline raised; the question counts toward failure_rate",
    )
    expected_to_fail_until_stage: int | None = None

    evaluated_at: datetime = Field(
        default_factory=_now,
        description="When this question was evaluated; differs across a resumed multi-day run",
    )

    @property
    def failed(self) -> bool:
        return self.error is not None

    @property
    def failed_on_quota(self) -> bool:
        """True when the failure was a provider quota or rate limit, not the system.

        These results are never checkpointed: the question was not measured, it
        was refused, and recording it would bake an infrastructure limit into the
        experiment as if it were pipeline behaviour.
        """
        if self.error is None:
            return False
        haystack = self.error.lower()
        return "ratelimit" in haystack.replace(" ", "") or "429" in haystack

    def all_metrics(self) -> dict[str, float]:
        """Every metric this question contributed, in one mapping."""
        return {**self.retrieval_metrics, **self.deterministic_metrics, **self.judge_metrics}


class ExperimentRun(BaseModel):
    """One named configuration measured against one dataset split."""

    run_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    name: str = Field(description="Experiment name, e.g. experiment-001-baseline")
    description: str = ""
    started_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = None

    dataset_split: DatasetSplit
    dataset_version: str
    dataset_size: int = 0

    # Reproducibility snapshot — read back verbatim by `make eval-diff`.
    git_commit: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    embedding_version: str | None = None
    chunking_version: str | None = None
    generator_provider: str | None = None
    generator_model: str | None = None
    generator_model_version: str | None = None
    judge_provider: str | None = None
    judge_model: str | None = None
    judge_model_version: str | None = None
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)

    metrics: dict[str, float] = Field(default_factory=dict)
    metrics_by_type: dict[str, dict[str, float]] = Field(default_factory=dict)
    system_metrics: dict[str, float] = Field(default_factory=dict)
    results: list[QuestionResult] = Field(default_factory=list)

    notes: str = ""

    @property
    def evaluation_days(self) -> list[str]:
        """The distinct UTC dates on which this run's questions were evaluated.

        More than one means the run was resumed across a quota boundary. That is
        expected on the free tiers and it is not free of consequence: a provider
        can change its served model between days, so the list is recorded and
        reported rather than left for someone to infer from timestamps.
        """
        return sorted({result.evaluated_at.date().isoformat() for result in self.results})

    @property
    def spans_multiple_days(self) -> bool:
        return len(self.evaluation_days) > 1

    @property
    def duration_seconds(self) -> float:
        if self.completed_at is None:
            return 0.0
        return (self.completed_at - self.started_at).total_seconds()

    def summary_line(self) -> str:
        """A one-line digest for CLI output."""
        recall = self.metrics.get("recall@5", 0.0)
        faithfulness = self.metrics.get("judge_faithfulness", 0.0)
        return (
            f"{self.name} [{self.dataset_split.value}/{self.dataset_version}] "
            f"n={self.dataset_size} recall@5={recall:.3f} "
            f"judge_faithfulness={faithfulness:.3f} "
            f"failure_rate={self.system_metrics.get('failure_rate', 0.0):.3f}"
        )


class HumanVerdict(BaseModel):
    """Layer 3 — a reviewer's judgement, stored alongside the automatic ones.

    Kept separate from the judge's scores rather than overwriting them: the point
    of Layer 3 is to be able to ask *where the automatic evaluator disagrees with
    a human*, and an overwrite destroys exactly that comparison.
    """

    run_id: uuid.UUID
    question_id: str
    reviewer: str = Field(min_length=1)
    reviewed_at: datetime = Field(default_factory=_now)

    is_correct: bool | None = None
    is_faithful: bool | None = None
    citations_are_correct: bool | None = None
    abstention_was_right: bool | None = None
    severity: str | None = Field(default=None, description="none | minor | major | unacceptable")
    comment: str = ""

    def as_scores(self) -> dict[str, float]:
        """Flatten the answered fields to ``human_*`` metrics; skip unanswered ones."""
        fields = {
            "human_correct": self.is_correct,
            "human_faithful": self.is_faithful,
            "human_citations_correct": self.citations_are_correct,
            "human_abstention_right": self.abstention_was_right,
        }
        return {name: float(value) for name, value in fields.items() if value is not None}
