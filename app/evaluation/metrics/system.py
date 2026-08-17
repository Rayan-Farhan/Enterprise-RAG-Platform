"""System metrics: latency percentiles, token usage, failure rate (Task 4.4).

Quality metrics alone cannot settle a Stage 6 decision. Reranking and multi-hop
buy accuracy with latency and tokens, and Stage 11 has to hold a budget, so the
cost side of every experiment is recorded with the same rigour as the accuracy
side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile over an unsorted sequence.

    Nearest-rank rather than interpolated: every reported percentile is then an
    actually-observed latency, which is what makes "p95 was 9.4s" a statement
    someone can go and reproduce.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(-(-fraction * len(ordered) // 1))))
    return ordered[rank - 1]


@dataclass
class SystemMetrics:
    """Cost and reliability profile of one experiment run."""

    total_questions: int = 0
    failures: int = 0

    latency_ms: list[float] = field(default_factory=list)
    retrieval_latency_ms: list[float] = field(default_factory=list)
    generation_latency_ms: list[float] = field(default_factory=list)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    evidence_tokens: int = 0

    def record(
        self,
        total_ms: float,
        retrieval_ms: float,
        generation_ms: float,
        token_counts: dict[str, int],
        evidence_tokens: int,
    ) -> None:
        """Accumulate one successful question."""
        self.latency_ms.append(total_ms)
        self.retrieval_latency_ms.append(retrieval_ms)
        self.generation_latency_ms.append(generation_ms)
        self.prompt_tokens += int(token_counts.get("prompt_tokens", 0))
        self.completion_tokens += int(token_counts.get("completion_tokens", 0))
        self.total_tokens += int(token_counts.get("total_tokens", 0))
        self.evidence_tokens += evidence_tokens

    def record_failure(self) -> None:
        """Accumulate one question the pipeline could not answer at all."""
        self.failures += 1

    def as_metrics(self) -> dict[str, float]:
        """Flatten to the metric mapping stored on the experiment run."""
        answered = len(self.latency_ms)
        return {
            "latency_p50_ms": percentile(self.latency_ms, 0.50),
            "latency_p95_ms": percentile(self.latency_ms, 0.95),
            "latency_p99_ms": percentile(self.latency_ms, 0.99),
            "retrieval_latency_p95_ms": percentile(self.retrieval_latency_ms, 0.95),
            "generation_latency_p95_ms": percentile(self.generation_latency_ms, 0.95),
            "failure_rate": (self.failures / self.total_questions if self.total_questions else 0.0),
            "avg_prompt_tokens": self.prompt_tokens / answered if answered else 0.0,
            "avg_completion_tokens": self.completion_tokens / answered if answered else 0.0,
            "avg_total_tokens": self.total_tokens / answered if answered else 0.0,
            "avg_evidence_tokens": self.evidence_tokens / answered if answered else 0.0,
        }
