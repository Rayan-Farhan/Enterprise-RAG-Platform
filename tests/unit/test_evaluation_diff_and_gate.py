"""Experiment comparison and the CI regression gate (Tasks 4.4, 4.6)."""

from __future__ import annotations

from typing import Any

import pytest

from app.evaluation.diff import (
    Direction,
    compare,
    direction,
    format_diff,
    gate,
)
from app.evaluation.results import ExperimentRun, QuestionResult
from app.evaluation.schemas import DatasetSplit, Difficulty, QuestionType


def run(**overrides: Any) -> ExperimentRun:
    payload: dict[str, Any] = {
        "name": "experiment-001-baseline",
        "dataset_split": DatasetSplit.DEV,
        "dataset_version": "v1",
        "dataset_size": 100,
        "embedding_version": "jina-embeddings-v3",
        "chunking_version": "fixed-v2",
        "judge_model": "openai/gpt-oss-120b",
        "metrics": {"recall@5": 0.60, "mrr": 0.50, "abstention_correct": 0.90},
        "system_metrics": {"latency_p95_ms": 9000.0, "failure_rate": 0.0},
    }
    payload.update(overrides)
    return ExperimentRun(**payload)


def question_result(**overrides: Any) -> QuestionResult:
    payload: dict[str, Any] = {
        "question_id": "dev-factual-0001",
        "question": "How much annual leave?",
        "question_type": QuestionType.FACTUAL,
        "difficulty": Difficulty.EASY,
        "retrieval_metrics": {"recall@5": 1.0},
    }
    payload.update(overrides)
    return QuestionResult(**payload)


class TestDirection:
    def test_quality_metrics_should_rise(self) -> None:
        assert direction("recall@5") is Direction.HIGHER_IS_BETTER

    def test_cost_metrics_should_fall(self) -> None:
        assert direction("latency_p95_ms") is Direction.LOWER_IS_BETTER
        assert direction("failure_rate") is Direction.LOWER_IS_BETTER

    def test_unknown_metric_defaults_to_higher_is_better(self) -> None:
        # A metric added in a later stage is never silently mis-signed.
        assert direction("some_stage_9_metric") is Direction.HIGHER_IS_BETTER


class TestCompare:
    def test_improvement_is_signed_so_positive_always_means_better(self) -> None:
        candidate = run(
            name="experiment-002",
            metrics={"recall@5": 0.70, "mrr": 0.50, "abstention_correct": 0.90},
            system_metrics={"latency_p95_ms": 7000.0, "failure_rate": 0.0},
        )
        diff = compare(run(), candidate)

        recall = next(d for d in diff.overall if d.metric == "recall@5")
        latency = next(d for d in diff.system if d.metric == "latency_p95_ms")

        assert recall.improvement == pytest.approx(0.10)
        # Latency fell by 2000ms — a lower-is-better metric, so that is positive.
        assert latency.delta == pytest.approx(-2000.0)
        assert latency.improvement == pytest.approx(2000.0)

    def test_regression_respects_the_tolerance(self) -> None:
        candidate = run(name="experiment-002", metrics={"recall@5": 0.57})
        diff = compare(run(), candidate)

        recall = next(d for d in diff.overall if d.metric == "recall@5")
        assert recall.is_regression(tolerance=0.0)
        assert not recall.is_regression(tolerance=0.05)

    def test_metric_present_in_only_one_run_has_no_delta(self) -> None:
        diff = compare(run(), run(name="experiment-002", metrics={"recall@5": 0.6, "new": 1.0}))
        added = next(d for d in diff.overall if d.metric == "new")
        assert added.baseline is None
        assert added.delta is None

    def test_per_type_breakdown_is_produced(self) -> None:
        baseline = run(metrics_by_type={"factual": {"recall@5": 0.8}})
        candidate = run(
            name="experiment-002",
            metrics_by_type={"factual": {"recall@5": 0.6}, "multi_hop": {"recall@5": 0.2}},
        )
        diff = compare(baseline, candidate)

        assert set(diff.by_type) == {"factual", "multi_hop"}
        factual = next(d for d in diff.by_type["factual"] if d.metric == "recall@5")
        assert factual.improvement == pytest.approx(-0.2)


class TestComparabilityWarnings:
    def test_different_embedding_versions_are_flagged(self) -> None:
        diff = compare(run(), run(name="experiment-002", embedding_version="bge-m3"))
        assert any("embedding versions" in w for w in diff.comparability_warnings)

    def test_different_judge_models_invalidate_judge_metrics(self) -> None:
        diff = compare(run(), run(name="experiment-002", judge_model="gemini-3.6-flash"))
        assert any("judged by different models" in w for w in diff.comparability_warnings)

    def test_changed_prompt_hash_is_flagged(self) -> None:
        baseline = run(prompt_hashes={"answer_v1": "aaa"})
        candidate = run(name="experiment-002", prompt_hashes={"answer_v1": "bbb"})
        diff = compare(baseline, candidate)
        assert any("prompt content changed" in w for w in diff.comparability_warnings)

    def test_identical_runs_produce_no_warnings(self) -> None:
        assert compare(run(), run()).comparability_warnings == []


class TestPerQuestionRegressions:
    def test_names_the_questions_that_moved(self) -> None:
        baseline = run(
            results=[
                question_result(question_id="dev-a", retrieval_metrics={"recall@5": 1.0}),
                question_result(question_id="dev-b", retrieval_metrics={"recall@5": 1.0}),
            ]
        )
        candidate = run(
            name="experiment-002",
            results=[
                question_result(question_id="dev-a", retrieval_metrics={"recall@5": 0.0}),
                question_result(question_id="dev-b", retrieval_metrics={"recall@5": 1.0}),
            ],
        )
        diff = compare(baseline, candidate)
        assert diff.per_question_regressions == ["dev-a"]

    def test_a_newly_failing_question_counts_as_a_regression(self) -> None:
        baseline = run(results=[question_result(question_id="dev-a")])
        candidate = run(
            name="experiment-002",
            results=[question_result(question_id="dev-a", error="TimeoutError: provider")],
        )
        assert compare(baseline, candidate).per_question_regressions == ["dev-a"]


class TestGate:
    def test_passes_when_nothing_regressed(self) -> None:
        result = gate(run(), run(name="candidate"), tolerance=0.0)
        assert result.passed
        assert "PASSED" in result.report()

    def test_fails_when_a_gated_metric_drops_beyond_tolerance(self) -> None:
        # Deliberately degrading retrieval must fail CI (Task 4.6 exit criterion).
        degraded = run(name="candidate", metrics={"recall@5": 0.20, "mrr": 0.50})
        result = gate(run(), degraded, tolerance=0.05)

        assert not result.passed
        assert [d.metric for d in result.failures] == ["recall@5"]
        assert "REGRESSION" not in result.report()
        assert "recall@5" in result.report()

    def test_tolerance_absorbs_small_movement(self) -> None:
        noisy = run(
            name="candidate",
            metrics={"recall@5": 0.57, "mrr": 0.50, "abstention_correct": 0.90},
        )
        assert gate(run(), noisy, tolerance=0.05).passed

    def test_dropping_a_gated_metric_is_a_failure_not_a_pass(self) -> None:
        # Otherwise the easiest route to a green gate is to stop computing it.
        result = gate(
            run(),
            run(name="candidate", metrics={"mrr": 0.50, "abstention_correct": 0.90}),
            tolerance=0.05,
        )
        assert not result.passed
        assert result.missing == ["recall@5"]
        assert "cannot be verified" in result.report()

    def test_metric_absent_from_the_baseline_is_not_gated(self) -> None:
        baseline = run(metrics={"recall@5": 0.6})
        candidate = run(name="candidate", metrics={"recall@5": 0.6})
        assert gate(baseline, candidate, tolerance=0.0).passed


class TestFormatting:
    def test_renders_warnings_metrics_and_regressed_questions(self) -> None:
        baseline = run(results=[question_result(question_id="dev-a")])
        candidate = run(
            name="experiment-002",
            embedding_version="bge-m3",
            metrics={"recall@5": 0.30, "mrr": 0.50, "abstention_correct": 0.90},
            results=[
                question_result(question_id="dev-a", retrieval_metrics={"recall@5": 0.0})
            ],
        )
        rendered = format_diff(compare(baseline, candidate), tolerance=0.05)

        assert "experiment-001-baseline  ->  experiment-002" in rendered
        assert "COMPARABILITY WARNINGS" in rendered
        assert "REGRESSION" in rendered
        assert "dev-a" in rendered
