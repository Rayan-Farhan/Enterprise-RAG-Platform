"""Paired bootstrap and sign test (results verification, Phase 4)."""

from __future__ import annotations

from typing import Any

import pytest

from app.evaluation.stats import compare_runs, paired_bootstrap, question_values, sign_test


def test_identical_runs_have_a_zero_width_interval() -> None:
    values = {f"q{i}": float(i % 2) for i in range(40)}
    result = paired_bootstrap(values, values, metric="recall@10")
    assert result.diff == 0.0
    assert (result.ci_low, result.ci_high) == (0.0, 0.0)
    assert not result.significant
    assert result.sign_test_p == 1.0


def test_a_consistent_gain_is_significant_and_reproducible() -> None:
    a = {f"q{i}": 0.0 for i in range(50)}
    b = {f"q{i}": 1.0 if i < 30 else 0.0 for i in range(50)}
    first = paired_bootstrap(a, b, metric="hit_rate@1")
    again = paired_bootstrap(a, b, metric="hit_rate@1")
    assert first == again  # fixed seed
    assert first.diff == pytest.approx(0.6)
    assert first.ci_low > 0.4 and first.ci_high < 0.8
    assert first.significant
    assert (first.improved, first.regressed) == (30, 0)
    assert first.sign_test_p < 1e-6


def test_pairing_uses_only_shared_questions() -> None:
    a = {"q1": 0.0, "q2": 1.0, "q3": 1.0}
    b = {"q2": 1.0, "q3": 0.0, "q4": 1.0}
    result = paired_bootstrap(a, b)
    assert result.n == 2
    assert result.diff == pytest.approx(-0.5)


def test_no_shared_questions_is_an_error() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap({"q1": 1.0}, {"q2": 1.0}, metric="mrr")


def test_sign_test_matches_the_binomial() -> None:
    # 9 improvements, 1 regression: P = 2 * (C(10,0) + C(10,1)) / 2**10
    assert sign_test(9, 1) == pytest.approx(2 * 11 / 1024)
    assert sign_test(0, 0) == 1.0


def _run(values: dict[str, float | None], error: set[str] | None = None) -> dict[str, Any]:
    return {
        "results": [
            {
                "question_id": q,
                "error": "boom" if q in (error or set()) else None,
                "retrieval_metrics": {} if v is None else {"recall@10": v},
                "deterministic_metrics": {},
                "judge_metrics": {},
            }
            for q, v in values.items()
        ]
    }


def test_question_values_skip_failed_and_unreported_questions() -> None:
    run = _run({"q1": 1.0, "q2": None, "q3": 0.5}, error={"q3"})
    assert question_values(run, "recall@10") == {"q1": 1.0}


def test_compare_runs_reads_run_records() -> None:
    a = _run({"q1": 0.0, "q2": 0.0, "q3": 1.0})
    b = _run({"q1": 1.0, "q2": 0.0, "q3": 1.0})
    result = compare_runs(a, b, "recall@10", resamples=500)
    assert result.n == 3 and result.diff == pytest.approx(1 / 3)
