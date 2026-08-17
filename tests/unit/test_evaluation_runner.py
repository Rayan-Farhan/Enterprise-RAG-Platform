"""Experiment runner behaviour (Task 4.4)."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

import pytest

from app.core.config import AppSettings
from app.evaluation.judge import JudgeVerdict
from app.evaluation.runner import ExperimentRunner, config_snapshot
from app.evaluation.schemas import (
    DatasetSplit,
    Difficulty,
    ExpectedEvidence,
    GoldenQuestion,
    QuestionType,
)
from app.generation.citation import SupportState
from app.generation.service import AnswerResult
from app.retrieval.schemas import Citation, RetrievedChunk

DOCUMENT_ID = uuid.uuid4()
VERSION_ID = uuid.uuid4()


@asynccontextmanager
async def _session() -> Any:
    yield object()


def session_factory() -> Any:
    return _session()


def chunk(element_ids: list[str], score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=DOCUMENT_ID,
        version_id=VERSION_ID,
        content="policy text",
        score=score,
        element_ids=element_ids,
    )


def golden(**overrides: Any) -> GoldenQuestion:
    payload: dict[str, Any] = {
        "question_id": "dev-factual-0001",
        "question": "How much annual leave is accrued?",
        "question_type": QuestionType.FACTUAL,
        "difficulty": Difficulty.EASY,
        "split": DatasetSplit.DEV,
        "expected_evidence": [
            ExpectedEvidence(
                document_id=DOCUMENT_ID, version_id=VERSION_ID, element_ids=["el-1"]
            )
        ],
        "acceptable_answer": "20 days.",
    }
    payload.update(overrides)
    return GoldenQuestion(**payload)


class FakeGeneration:
    """Returns a scripted AnswerResult, or raises for a designated question."""

    def __init__(
        self,
        result: AnswerResult | None = None,
        raise_for: set[str] | None = None,
    ) -> None:
        self.result = result
        self.raise_for = raise_for or set()
        self.queries: list[str] = []

    async def answer(self, query: str, session: Any, **kwargs: Any) -> AnswerResult:
        self.queries.append(query)
        if query in self.raise_for:
            raise TimeoutError("provider timed out")
        assert self.result is not None
        return self.result


class FakeJudge:
    def __init__(self, verdict: JudgeVerdict | None = None) -> None:
        self.verdict = verdict
        self.calls = 0

    async def judge(self, question: Any, result: Any, evidence_text: str) -> JudgeVerdict:
        self.calls += 1
        return self.verdict or JudgeVerdict(question_id=question.question_id)


def settings(**overrides: Any) -> AppSettings:
    base: dict[str, Any] = {"INFERENCE_PROFILE": "hosted", "EVAL_CONCURRENCY": 2}
    base.update(overrides)
    return AppSettings(**base)


def good_answer() -> AnswerResult:
    hit = chunk(["el-1"], score=0.62)
    miss = chunk(["el-9"], score=0.31)
    return AnswerResult(
        query="How much annual leave is accrued?",
        answer="Employees accrue 20 days per year [1].",
        support=SupportState.GROUNDED,
        citations=[
            Citation(
                marker="[1]",
                document_id=DOCUMENT_ID,
                version_id=VERSION_ID,
                chunk_id=hit.chunk_id,
                page_number=14,
                element_ids=["el-1"],
            )
        ],
        retrieved_chunk_ids=[hit.chunk_id, miss.chunk_id],
        retrieved_chunks=[hit, miss],
        context_chunk_ids=[hit.chunk_id],
        provider="gemini",
        model_name="gemini-3.6-flash",
        model_version="gemini-3.6-flash",
        prompt_versions={"answer": "answer_v1"},
        prompt_hashes={"answer": "abc123"},
        token_counts={"prompt_tokens": 900, "completion_tokens": 60, "total_tokens": 960},
        retrieval_latency_ms=210.0,
        generation_latency_ms=6400.0,
        total_latency_ms=6700.0,
        evidence_block="--- BEGIN EVIDENCE [1] ---\npolicy text\n--- END EVIDENCE [1] ---",
        evidence_tokens=420,
    )


class TestConfigSnapshot:
    def test_captures_what_can_move_a_metric_and_no_credentials(self) -> None:
        snapshot = config_snapshot(settings(GEMINI_API_KEY="secret-key"))
        assert snapshot["retrieval_min_score"] == 0.35
        assert snapshot["chunking_version"] == "fixed-v2"
        assert "secret-key" not in str(snapshot)


class TestRun:
    async def test_records_metrics_provenance_and_system_costs(self) -> None:
        runner = ExperimentRunner(
            generation_service=FakeGeneration(good_answer()),  # type: ignore[arg-type]
            judge=FakeJudge(),  # type: ignore[arg-type]
            settings=settings(),
        )

        run = await runner.run(
            name="experiment-001-baseline",
            questions=[golden()],
            session_factory=session_factory,
            split=DatasetSplit.DEV,
            dataset_version="v1",
        )

        assert run.dataset_size == 1
        assert run.generator_model == "gemini-3.6-flash"
        assert run.prompt_hashes == {"answer": "abc123"}
        # el-1 is at rank 1 of 2 retrieved chunks.
        assert run.metrics["recall@5"] == pytest.approx(1.0)
        assert run.metrics["mrr"] == pytest.approx(1.0)
        assert run.metrics["precision@5"] == pytest.approx(0.5)
        assert run.metrics["element_match"] == pytest.approx(1.0)
        assert run.system_metrics["failure_rate"] == 0.0
        assert run.system_metrics["avg_total_tokens"] == pytest.approx(960.0)
        assert run.completed_at is not None

    async def test_context_recall_reflects_what_survived_assembly(self) -> None:
        runner = ExperimentRunner(
            generation_service=FakeGeneration(good_answer()),  # type: ignore[arg-type]
            judge=FakeJudge(),  # type: ignore[arg-type]
            settings=settings(),
        )
        run = await runner.run(
            name="e",
            questions=[golden()],
            session_factory=session_factory,
            split=DatasetSplit.DEV,
            dataset_version="v1",
        )
        result = run.results[0]
        assert result.retrieved_element_ids == ["el-1", "el-9"]
        assert result.context_element_ids == ["el-1"]

    async def test_a_failing_question_is_recorded_not_dropped(self) -> None:
        # Dropping it would make a run that crashed on its hardest third look
        # like a run that scored well.
        runner = ExperimentRunner(
            generation_service=FakeGeneration(  # type: ignore[arg-type]
                good_answer(), raise_for={"trigger a provider failure"}
            ),
            judge=FakeJudge(),  # type: ignore[arg-type]
            settings=settings(),
        )

        run = await runner.run(
            name="e",
            questions=[golden(), golden(question_id="dev-factual-0002", question="trigger a provider failure")],
            session_factory=session_factory,
            split=DatasetSplit.DEV,
            dataset_version="v1",
        )

        failed = next(r for r in run.results if r.question_id == "dev-factual-0002")
        assert failed.failed
        assert "TimeoutError" in (failed.error or "")
        assert failed.retrieval_metrics == {}
        assert run.system_metrics["failure_rate"] == pytest.approx(0.5)
        # The failure contributes no metrics, so the successful question's 1.0
        # is not diluted into 0.5.
        assert run.metrics["recall@5"] == pytest.approx(1.0)

    async def test_per_type_breakdown_separates_question_types(self) -> None:
        runner = ExperimentRunner(
            generation_service=FakeGeneration(good_answer()),  # type: ignore[arg-type]
            judge=FakeJudge(),  # type: ignore[arg-type]
            settings=settings(),
        )
        run = await runner.run(
            name="e",
            questions=[
                golden(),
                golden(
                    question_id="dev-multihop-0001",
                    question_type=QuestionType.MULTI_HOP,
                    difficulty=Difficulty.HARD,
                ),
            ],
            session_factory=session_factory,
            split=DatasetSplit.DEV,
            dataset_version="v1",
        )
        assert set(run.metrics_by_type) == {"factual", "multi_hop"}

    async def test_judge_can_be_switched_off_for_a_fast_subset(self) -> None:
        judge = FakeJudge()
        runner = ExperimentRunner(
            generation_service=FakeGeneration(good_answer()),  # type: ignore[arg-type]
            judge=judge,  # type: ignore[arg-type]
            settings=settings(),
        )
        await runner.run(
            name="e",
            questions=[golden()],
            session_factory=session_factory,
            split=DatasetSplit.DEV,
            dataset_version="v1",
            judge_enabled=False,
        )
        assert judge.calls == 0

    async def test_judge_scores_are_merged_into_the_run_metrics(self) -> None:
        verdict = JudgeVerdict(
            question_id="dev-factual-0001",
            scores={"faithfulness": 1.0, "citation_correctness": 0.75},
            judge_provider="groq",
            judge_model="openai/gpt-oss-120b",
            prompt_versions={"judge_answer_v1": "judge_answer_v1"},
            prompt_hashes={"judge_answer_v1": "deadbeef"},
        )
        runner = ExperimentRunner(
            generation_service=FakeGeneration(good_answer()),  # type: ignore[arg-type]
            judge=FakeJudge(verdict),  # type: ignore[arg-type]
            settings=settings(),
        )
        run = await runner.run(
            name="e",
            questions=[golden()],
            session_factory=session_factory,
            split=DatasetSplit.DEV,
            dataset_version="v1",
        )
        assert run.metrics["judge_faithfulness"] == pytest.approx(1.0)
        assert run.judge_model == "openai/gpt-oss-120b"
        assert run.prompt_hashes["judge_answer_v1"] == "deadbeef"

    async def test_results_are_ordered_by_question_id(self) -> None:
        runner = ExperimentRunner(
            generation_service=FakeGeneration(good_answer()),  # type: ignore[arg-type]
            judge=FakeJudge(),  # type: ignore[arg-type]
            settings=settings(),
        )
        run = await runner.run(
            name="e",
            questions=[golden(question_id="dev-z"), golden(question_id="dev-a")],
            session_factory=session_factory,
            split=DatasetSplit.DEV,
            dataset_version="v1",
        )
        assert [r.question_id for r in run.results] == ["dev-a", "dev-z"]
