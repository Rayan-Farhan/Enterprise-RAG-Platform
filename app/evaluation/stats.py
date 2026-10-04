"""Paired significance for experiment comparisons (results verification, Phase 4).

Two runs over the same golden questions are compared question by question:
each question's metric value in run A is paired with its value in run B, and
the uncertainty of the mean difference comes from resampling questions with
replacement (a paired bootstrap). Pairing matters: questions differ far more
in difficulty than configurations do, and an unpaired comparison would bury a
real 0.05 gain under that spread.

Fixed choices, so every interval in the docs is reproducible from this code:

* ``DEFAULT_RESAMPLES`` = 10,000 resamples and ``DEFAULT_SEED`` = 2026.
* A 95% percentile interval.
* Pairing by ``question_id`` over the intersection of questions that report
  the metric in both runs. Failed questions are excluded, and ``n`` is
  reported, because a comparison over 79 questions is not one over 82.
* "Significant" means the interval excludes zero.

An exact two-sided sign test over the questions that changed is reported
alongside as a cross-check for small n (34 answerable on validation). It
ignores magnitudes and asks only whether improvements outnumber regressions.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 2026
DEFAULT_CONFIDENCE = 0.95


@dataclass(frozen=True)
class PairedComparison:
    """Run B minus run A on one metric, paired by question."""

    metric: str
    n: int
    mean_a: float
    mean_b: float
    diff: float
    ci_low: float
    ci_high: float
    improved: int
    regressed: int
    sign_test_p: float
    resamples: int
    seed: int
    confidence: float

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0

    def summary(self) -> str:
        sig = "significant" if self.significant else "not significant"
        return (
            f"{self.metric}: {self.mean_a:.3f} -> {self.mean_b:.3f}, "
            f"diff {self.diff:+.3f} [{self.ci_low:+.3f}, {self.ci_high:+.3f}] "
            f"n={self.n}, {sig}; sign test {self.improved}+/{self.regressed}- "
            f"p={self.sign_test_p:.3f}"
        )


def question_values(run: Mapping[str, Any], metric: str) -> dict[str, float]:
    """Per-question values of ``metric`` from a run record (raw JSON or model dump).

    Failed questions and questions that do not report the metric are left out,
    which is the runner's own aggregation rule.
    """
    values: dict[str, float] = {}
    for result in run.get("results") or []:
        if result.get("error") is not None:
            continue
        merged = {
            **(result.get("retrieval_metrics") or {}),
            **(result.get("deterministic_metrics") or {}),
            **(result.get("judge_metrics") or {}),
        }
        value = merged.get(metric)
        if value is not None:
            values[str(result["question_id"])] = float(value)
    return values


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already sorted list."""
    if not sorted_values:
        return math.nan
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def sign_test(improved: int, regressed: int) -> float:
    """Exact two-sided binomial sign test p-value (ties excluded)."""
    total = improved + regressed
    if total == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = sum(math.comb(total, k) for k in range(tail + 1)) / 2**total
    return float(min(1.0, 2 * probability))


def paired_bootstrap(
    a: Mapping[str, float],
    b: Mapping[str, float],
    metric: str = "",
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> PairedComparison:
    """Mean of (b - a) over shared questions with a percentile bootstrap interval."""
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError(f"no shared questions report {metric or 'the metric'}")
    diffs = [b[q] - a[q] for q in shared]
    n = len(diffs)

    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2

    improved = sum(1 for d in diffs if d > 0)
    regressed = sum(1 for d in diffs if d < 0)
    return PairedComparison(
        metric=metric,
        n=n,
        mean_a=sum(a[q] for q in shared) / n,
        mean_b=sum(b[q] for q in shared) / n,
        diff=sum(diffs) / n,
        ci_low=_percentile(means, tail),
        ci_high=_percentile(means, 1 - tail),
        improved=improved,
        regressed=regressed,
        sign_test_p=sign_test(improved, regressed),
        resamples=resamples,
        seed=seed,
        confidence=confidence,
    )


def compare_runs(
    run_a: Mapping[str, Any],
    run_b: Mapping[str, Any],
    metric: str,
    **options: Any,
) -> PairedComparison:
    """Paired bootstrap of one metric between two run records."""
    return paired_bootstrap(
        question_values(run_a, metric), question_values(run_b, metric), metric=metric, **options
    )
