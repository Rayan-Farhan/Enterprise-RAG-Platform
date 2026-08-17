"""Resumable multi-day experiment runs (Task 4.5).

The free-tier providers meter by the day, so a 100-question run cannot finish in
one sitting. These tests pin the two behaviours that make a multi-day run
trustworthy: nothing already paid for is lost, and nothing refused by a quota is
recorded as if the pipeline had answered it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.evaluation import storage
from app.evaluation.results import ExperimentRun, QuestionResult
from app.evaluation.runner import QUOTA_ABORT_THRESHOLD, ExperimentRunner, QuotaExhausted
from app.evaluation.schemas import (
    DatasetSplit,
    Difficulty,
    ExpectedEvidence,
    GoldenQuestion,
    QuestionType,
)

RATE_LIMIT_ERROR = (
    "ModelProviderException: Provider 'groq' failed: groq rate limit reached (HTTP 429): "
    '{"error":{"message":"Rate limit reached ... tokens per day (TPD): Limit 200000"}}'
)


def result(question_id: str, **overrides: Any) -> QuestionResult:
    payload: dict[str, Any] = {
        "question_id": question_id,
        "question": "How much annual leave does a staff employee accrue?",
        "question_type": QuestionType.FACTUAL,
        "difficulty": Difficulty.EASY,
        "retrieval_metrics": {"recall@5": 1.0},
        "latency_ms": 9000.0,
        "token_counts": {"total_tokens": 2400},
        "evidence_tokens": 1600,
    }
    payload.update(overrides)
    return QuestionResult(**payload)


def golden(question_id: str, question_type: QuestionType = QuestionType.FACTUAL) -> GoldenQuestion:
    return GoldenQuestion(
        question_id=question_id,
        question="How much annual leave does a staff employee accrue?",
        question_type=question_type,
        split=DatasetSplit.DEV,
        acceptable_answer="10 days up to two years of service.",
        expected_evidence=[
            ExpectedEvidence(
                document_id=uuid.uuid4(),
                version_id=uuid.uuid4(),
                element_ids=["docling_elem_25_123"],
            )
        ],
    )


class TestCheckpointStore:
    def test_a_result_round_trips(self, tmp_path: Path) -> None:
        storage.append_checkpoint("experiment-001-baseline", result("dev-factual-001"), tmp_path)

        loaded = storage.load_checkpoint("experiment-001-baseline", tmp_path)

        assert list(loaded) == ["dev-factual-001"]
        assert loaded["dev-factual-001"].retrieval_metrics == {"recall@5": 1.0}

    def test_quota_failures_are_never_checkpointed(self, tmp_path: Path) -> None:
        # The question was refused, not measured. Recording it would bake an
        # accounting limit into the experiment as pipeline behaviour, and the
        # resumed run would never retry it.
        storage.append_checkpoint(
            "experiment-001-baseline", result("dev-factual-001", error=RATE_LIMIT_ERROR), tmp_path
        )

        assert storage.load_checkpoint("experiment-001-baseline", tmp_path) == {}

    def test_genuine_failures_are_checkpointed(self, tmp_path: Path) -> None:
        storage.append_checkpoint(
            "experiment-001-baseline",
            result("dev-factual-001", error="ValueError: chunk had no text"),
            tmp_path,
        )

        loaded = storage.load_checkpoint("experiment-001-baseline", tmp_path)

        assert loaded["dev-factual-001"].failed
        assert not loaded["dev-factual-001"].failed_on_quota

    def test_a_truncated_trailing_line_is_dropped_not_fatal(self, tmp_path: Path) -> None:
        # What a process killed mid-write leaves behind. Losing one question
        # beats losing the day's work.
        storage.append_checkpoint("experiment-001-baseline", result("dev-factual-001"), tmp_path)
        path = storage.checkpoint_path("experiment-001-baseline", tmp_path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write('{"question_id": "dev-factual-002", "quest')

        loaded = storage.load_checkpoint("experiment-001-baseline", tmp_path)

        assert list(loaded) == ["dev-factual-001"]

    def test_a_re_evaluated_question_overwrites_the_earlier_entry(self, tmp_path: Path) -> None:
        storage.append_checkpoint("experiment-001-baseline", result("dev-factual-001"), tmp_path)
        storage.append_checkpoint(
            "experiment-001-baseline",
            result("dev-factual-001", retrieval_metrics={"recall@5": 0.0}),
            tmp_path,
        )

        loaded = storage.load_checkpoint("experiment-001-baseline", tmp_path)

        assert loaded["dev-factual-001"].retrieval_metrics == {"recall@5": 0.0}

    def test_checkpoints_do_not_appear_as_committed_results(self, tmp_path: Path) -> None:
        storage.append_checkpoint("experiment-001-baseline", result("dev-factual-001"), tmp_path)

        assert storage.list_run_files(tmp_path) == []

    def test_clearing_removes_the_file(self, tmp_path: Path) -> None:
        storage.append_checkpoint("experiment-001-baseline", result("dev-factual-001"), tmp_path)
        storage.clear_checkpoint("experiment-001-baseline", tmp_path)

        assert storage.load_checkpoint("experiment-001-baseline", tmp_path) == {}

    def test_clearing_an_absent_checkpoint_is_not_an_error(self, tmp_path: Path) -> None:
        storage.clear_checkpoint("never-ran", tmp_path)


class TestMultiDayReporting:
    def test_a_single_day_run_does_not_claim_to_span_days(self) -> None:
        now = datetime.now(UTC)
        run = ExperimentRun(
            name="experiment-001-baseline",
            dataset_split=DatasetSplit.DEV,
            dataset_version="v1",
            results=[result("a", evaluated_at=now), result("b", evaluated_at=now)],
        )

        assert not run.spans_multiple_days
        assert len(run.evaluation_days) == 1

    def test_a_resumed_run_reports_every_day_it_touched(self) -> None:
        now = datetime.now(UTC)
        run = ExperimentRun(
            name="experiment-001-baseline",
            dataset_split=DatasetSplit.DEV,
            dataset_version="v1",
            results=[
                result("a", evaluated_at=now - timedelta(days=2)),
                result("b", evaluated_at=now - timedelta(days=1)),
                result("c", evaluated_at=now),
            ],
        )

        assert run.spans_multiple_days
        assert len(run.evaluation_days) == 3
        assert run.evaluation_days == sorted(run.evaluation_days)


class _StubGeneration:
    """Answers a fixed number of questions, then refuses like an exhausted tier."""

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self.calls = 0

    async def answer(self, query: str, session: Any) -> Any:  # noqa: ANN401 - test double
        self.calls += 1
        if self.calls > self.budget:
            raise RuntimeError(
                "groq rate limit reached (HTTP 429): tokens per day (TPD): Limit 200000"
            )
        raise RuntimeError("ValueError: deliberate non-quota failure")


class _NullSessionFactory:
    def __call__(self) -> _NullSessionFactory:
        return self

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: object) -> bool:
        return False


class TestQuotaAbort:
    @pytest.mark.asyncio
    async def test_the_run_stops_instead_of_recording_the_rest_as_failures(self) -> None:
        # Grinding on would produce a run that reads as a catastrophic quality
        # regression and is really an accounting limit.
        settings = _settings(concurrency=1)
        generation = _StubGeneration(budget=3)
        runner = ExperimentRunner(
            generation_service=generation,  # type: ignore[arg-type]
            judge=_StubJudge(),  # type: ignore[arg-type]
            settings=settings,
        )
        questions = [golden(f"dev-factual-{i:03d}") for i in range(1, 21)]

        with pytest.raises(QuotaExhausted) as exc_info:
            await runner.run(
                name="experiment-001-baseline",
                questions=questions,
                session_factory=_NullSessionFactory(),
                split=DatasetSplit.DEV,
                dataset_version="v1",
                judge_enabled=False,
            )

        assert exc_info.value.completed == 3
        assert exc_info.value.total == 20
        # It stopped shortly after the budget ran out rather than attempting all 20.
        assert generation.calls <= 3 + QUOTA_ABORT_THRESHOLD + settings.EVAL_CONCURRENCY

    @pytest.mark.asyncio
    async def test_everything_paid_for_before_the_abort_is_handed_back(self) -> None:
        settings = _settings(concurrency=1)
        runner = ExperimentRunner(
            generation_service=_StubGeneration(budget=2),  # type: ignore[arg-type]
            judge=_StubJudge(),  # type: ignore[arg-type]
            settings=settings,
        )
        seen: list[QuestionResult] = []

        with pytest.raises(QuotaExhausted):
            await runner.run(
                name="experiment-001-baseline",
                questions=[golden(f"dev-factual-{i:03d}") for i in range(1, 11)],
                session_factory=_NullSessionFactory(),
                split=DatasetSplit.DEV,
                dataset_version="v1",
                judge_enabled=False,
                on_result=seen.append,
            )

        assert len(seen) == 2
        assert all(not r.failed_on_quota for r in seen)

    @pytest.mark.asyncio
    async def test_resumed_questions_are_not_re_evaluated(self) -> None:
        settings = _settings(concurrency=1)
        generation = _StubGeneration(budget=100)
        runner = ExperimentRunner(
            generation_service=generation,  # type: ignore[arg-type]
            judge=_StubJudge(),  # type: ignore[arg-type]
            settings=settings,
        )
        questions = [golden(f"dev-factual-{i:03d}") for i in range(1, 6)]
        already = {f"dev-factual-{i:03d}": result(f"dev-factual-{i:03d}") for i in range(1, 4)}

        run = await runner.run(
            name="experiment-001-baseline",
            questions=questions,
            session_factory=_NullSessionFactory(),
            split=DatasetSplit.DEV,
            dataset_version="v1",
            judge_enabled=False,
            completed=already,
        )

        assert generation.calls == 2
        assert len(run.results) == 5
        assert run.dataset_size == 5

    @pytest.mark.asyncio
    async def test_checkpointed_results_from_another_split_are_ignored(self) -> None:
        settings = _settings(concurrency=1)
        generation = _StubGeneration(budget=100)
        runner = ExperimentRunner(
            generation_service=generation,  # type: ignore[arg-type]
            judge=_StubJudge(),  # type: ignore[arg-type]
            settings=settings,
        )

        run = await runner.run(
            name="experiment-001-baseline",
            questions=[golden("dev-factual-001")],
            session_factory=_NullSessionFactory(),
            split=DatasetSplit.DEV,
            dataset_version="v1",
            judge_enabled=False,
            completed={"val-factual-001": result("val-factual-001")},
        )

        assert [r.question_id for r in run.results] == ["dev-factual-001"]


class _StubJudge:
    async def judge(self, *args: object, **kwargs: object) -> Any:  # noqa: ANN401 - test double
        raise AssertionError("the judge must not be called when judging is disabled")


def _settings(concurrency: int) -> Any:
    from app.core.config import AppSettings

    return AppSettings(EVAL_CONCURRENCY=concurrency, EVAL_JUDGE_ENABLED=False)
