"""Experiment comparison and the CI regression gate (Tasks 4.4, 4.6).

The roadmap's rule for every stage from 5 onward is that nothing merges on
intuition. That rule needs a tool that answers "what actually changed?" at the
resolution decisions are made at — per metric *and* per question type, because
the aggregate number routinely hides two opposite movements cancelling out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from app.evaluation.results import ExperimentRun


class Direction(StrEnum):
    """Which way a metric is allowed to move."""

    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


#: Metrics where a rise is a regression. Everything not listed defaults to
#: higher-is-better, so a metric added in a later stage is never silently
#: mis-signed in the wrong direction — it just needs adding here if it is a cost.
_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "failure_rate",
        "fabricated_citation_count",
        "rejected",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "retrieval_latency_p95_ms",
        "generation_latency_p95_ms",
        "avg_prompt_tokens",
        "avg_completion_tokens",
        "avg_total_tokens",
        "avg_evidence_tokens",
    }
)

#: The metrics the CI gate protects. Deliberately short: gating on everything
#: makes the gate flaky and trains people to ignore it, which is worse than
#: having no gate at all. These are the ones whose regression means the system
#: got meaningfully worse.
GATED_METRICS: tuple[str, ...] = (
    "recall@5",
    "mrr",
    "context_precision",
    "abstention_correct",
    "citation_integrity",
    "document_match",
)


def direction(metric: str) -> Direction:
    """Return whether a metric should rise or fall."""
    return Direction.LOWER_IS_BETTER if metric in _LOWER_IS_BETTER else Direction.HIGHER_IS_BETTER


@dataclass(frozen=True)
class MetricDelta:
    """One metric's movement between two runs."""

    metric: str
    baseline: float | None
    candidate: float | None
    direction: Direction

    @property
    def delta(self) -> float | None:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline

    @property
    def improvement(self) -> float | None:
        """Signed so that positive always means better, whichever way the metric runs."""
        if self.delta is None:
            return None
        return self.delta if self.direction is Direction.HIGHER_IS_BETTER else -self.delta

    def is_regression(self, tolerance: float) -> bool:
        """True when the metric moved the wrong way by more than the tolerance."""
        improvement = self.improvement
        return improvement is not None and improvement < -tolerance


@dataclass
class ExperimentDiff:
    """Full comparison of two runs."""

    baseline_name: str
    candidate_name: str
    overall: list[MetricDelta] = field(default_factory=list)
    by_type: dict[str, list[MetricDelta]] = field(default_factory=dict)
    system: list[MetricDelta] = field(default_factory=list)
    per_question_regressions: list[str] = field(default_factory=list)
    comparability_warnings: list[str] = field(default_factory=list)

    def regressions(self, tolerance: float) -> list[MetricDelta]:
        """Every overall metric that moved the wrong way beyond the tolerance."""
        return [d for d in self.overall if d.is_regression(tolerance)]


def _deltas(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> list[MetricDelta]:
    names = sorted(set(baseline) | set(candidate))
    return [
        MetricDelta(
            metric=name,
            baseline=baseline.get(name),
            candidate=candidate.get(name),
            direction=direction(name),
        )
        for name in names
    ]


def _comparability_warnings(baseline: ExperimentRun, candidate: ExperimentRun) -> list[str]:
    """Flag differences that make the two runs not strictly comparable.

    These are warnings rather than errors on purpose: comparing across a
    deliberate embedding change is exactly what Stage 6 does. What must never
    happen is comparing across one *by accident* and reading the difference as a
    win for whatever else changed.
    """
    warnings: list[str] = []

    if baseline.dataset_split != candidate.dataset_split:
        warnings.append(
            f"different splits: {baseline.dataset_split.value} vs {candidate.dataset_split.value}"
        )
    if baseline.dataset_version != candidate.dataset_version:
        warnings.append(
            f"different dataset versions: {baseline.dataset_version} vs {candidate.dataset_version}"
        )
    if baseline.dataset_size != candidate.dataset_size:
        warnings.append(
            f"different question counts: {baseline.dataset_size} vs {candidate.dataset_size}"
        )
    if baseline.embedding_version != candidate.embedding_version:
        warnings.append(
            f"different embedding versions: {baseline.embedding_version} vs "
            f"{candidate.embedding_version}"
        )
    if baseline.chunking_version != candidate.chunking_version:
        warnings.append(
            f"different chunking versions: {baseline.chunking_version} vs "
            f"{candidate.chunking_version}"
        )
    if baseline.judge_model != candidate.judge_model:
        warnings.append(
            f"judged by different models: {baseline.judge_model} vs {candidate.judge_model} "
            f"— judge_* metrics are not comparable"
        )
    if baseline.prompt_hashes and baseline.prompt_hashes != candidate.prompt_hashes:
        changed = sorted(
            key
            for key in set(baseline.prompt_hashes) | set(candidate.prompt_hashes)
            if baseline.prompt_hashes.get(key) != candidate.prompt_hashes.get(key)
        )
        warnings.append(f"prompt content changed for: {', '.join(changed)}")

    return warnings


def _per_question_regressions(
    baseline: ExperimentRun,
    candidate: ExperimentRun,
    tolerance: float,
) -> list[str]:
    """Question IDs that got materially worse, so the diff points at examples."""
    baseline_by_id = {r.question_id: r for r in baseline.results}
    regressed: list[str] = []

    for result in candidate.results:
        before = baseline_by_id.get(result.question_id)
        if before is None:
            continue

        if result.failed and not before.failed:
            regressed.append(result.question_id)
            continue

        before_metrics = before.all_metrics()
        after_metrics = result.all_metrics()
        for metric in GATED_METRICS:
            if metric in before_metrics and metric in after_metrics:
                delta = MetricDelta(
                    metric=metric,
                    baseline=before_metrics[metric],
                    candidate=after_metrics[metric],
                    direction=direction(metric),
                )
                if delta.is_regression(tolerance):
                    regressed.append(result.question_id)
                    break

    return sorted(set(regressed))


def compare(
    baseline: ExperimentRun,
    candidate: ExperimentRun,
    tolerance: float = 0.0,
) -> ExperimentDiff:
    """Produce a per-metric, per-question-type comparison of two runs."""
    diff = ExperimentDiff(
        baseline_name=baseline.name,
        candidate_name=candidate.name,
        overall=_deltas(baseline.metrics, candidate.metrics),
        system=_deltas(baseline.system_metrics, candidate.system_metrics),
        comparability_warnings=_comparability_warnings(baseline, candidate),
        per_question_regressions=_per_question_regressions(baseline, candidate, tolerance),
    )

    for question_type in sorted(set(baseline.metrics_by_type) | set(candidate.metrics_by_type)):
        diff.by_type[question_type] = _deltas(
            baseline.metrics_by_type.get(question_type, {}),
            candidate.metrics_by_type.get(question_type, {}),
        )

    return diff


@dataclass(frozen=True)
class GateResult:
    """Outcome of the CI regression gate."""

    passed: bool
    failures: list[MetricDelta]
    missing: list[str]
    tolerance: float

    def report(self) -> str:
        """Human-readable gate verdict for CI logs."""
        if self.passed and not self.missing:
            return f"Regression gate PASSED (tolerance {self.tolerance:.3f})."

        lines: list[str] = [f"Regression gate FAILED (tolerance {self.tolerance:.3f})."]
        for delta in self.failures:
            lines.append(
                f"  {delta.metric}: {delta.baseline:.4f} -> {delta.candidate:.4f} "
                f"({delta.improvement:+.4f}, {delta.direction.value}-is-better)"
            )
        for metric in self.missing:
            lines.append(f"  {metric}: absent from the candidate run — cannot be verified")
        return "\n".join(lines)


def gate(
    baseline: ExperimentRun,
    candidate: ExperimentRun,
    tolerance: float,
    metrics: Sequence[str] = GATED_METRICS,
) -> GateResult:
    """Fail when a gated metric regressed beyond the tolerance.

    A gated metric present in the baseline but missing from the candidate is a
    failure, not a pass. Otherwise the easiest way to get a green gate would be
    to stop computing the metric.
    """
    failures: list[MetricDelta] = []
    missing: list[str] = []

    for metric in metrics:
        if metric not in baseline.metrics:
            continue
        if metric not in candidate.metrics:
            missing.append(metric)
            continue

        delta = MetricDelta(
            metric=metric,
            baseline=baseline.metrics[metric],
            candidate=candidate.metrics[metric],
            direction=direction(metric),
        )
        if delta.is_regression(tolerance):
            failures.append(delta)

    return GateResult(
        passed=not failures and not missing,
        failures=failures,
        missing=missing,
        tolerance=tolerance,
    )


def format_diff(diff: ExperimentDiff, tolerance: float = 0.0, show_types: bool = True) -> str:
    """Render a diff as an aligned text table for the CLI."""
    lines: list[str] = [
        f"{diff.baseline_name}  ->  {diff.candidate_name}",
        "=" * 78,
    ]

    if diff.comparability_warnings:
        lines.append("")
        lines.append("COMPARABILITY WARNINGS")
        lines.extend(f"  ! {warning}" for warning in diff.comparability_warnings)

    def table(title: str, deltas: Sequence[MetricDelta]) -> list[str]:
        if not deltas:
            return []
        rows = [
            "",
            title,
            "-" * 78,
            f"{'metric':<34}{'baseline':>12}{'candidate':>12}{'change':>14}",
        ]
        for delta in deltas:
            baseline = "—" if delta.baseline is None else f"{delta.baseline:.4f}"
            candidate = "—" if delta.candidate is None else f"{delta.candidate:.4f}"
            if delta.improvement is None:
                change = "—"
            else:
                marker = (
                    "REGRESSION"
                    if delta.is_regression(tolerance)
                    else ("better" if delta.improvement > 0 else "")
                )
                change = f"{delta.delta:+.4f} {marker}".rstrip()
            rows.append(f"{delta.metric:<34}{baseline:>12}{candidate:>12}{change:>14}")
        return rows

    lines.extend(table("OVERALL", diff.overall))
    lines.extend(table("SYSTEM", diff.system))

    if show_types:
        for question_type, deltas in diff.by_type.items():
            moved = [d for d in deltas if d.delta is None or abs(d.delta) > 1e-9]
            lines.extend(table(f"BY TYPE — {question_type}", moved))

    if diff.per_question_regressions:
        lines.append("")
        lines.append(f"QUESTIONS THAT REGRESSED ({len(diff.per_question_regressions)})")
        lines.extend(f"  {qid}" for qid in diff.per_question_regressions)

    return "\n".join(lines)
