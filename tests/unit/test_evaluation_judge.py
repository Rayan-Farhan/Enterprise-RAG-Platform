"""Layer 2 LLM-as-judge behaviour (Task 4.3)."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from app.core.config import AppSettings
from app.evaluation.judge import (
    JudgeParseError,
    JudgeService,
    normalize_score,
    parse_judge_json,
)
from app.evaluation.schemas import (
    DatasetSplit,
    Difficulty,
    ExpectedEvidence,
    GoldenQuestion,
    QuestionType,
)
from app.generation.citation import SupportState
from app.generation.service import AnswerResult
from app.models.schemas import GenerationResult, ModelMetadata

DOCUMENT_ID = uuid.uuid4()
VERSION_ID = uuid.uuid4()


class RecordingGateway:
    """A gateway that returns scripted judge JSON and records how it was called."""

    def __init__(self, payloads: dict[str, Any] | list[str] | None = None) -> None:
        self.payloads = payloads
        self.calls: list[dict[str, Any]] = []

    async def generate(self, **kwargs: Any) -> GenerationResult:
        self.calls.append(kwargs)
        prompt_version = kwargs.get("prompt_version", "")

        if isinstance(self.payloads, list):
            text = self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]
        elif isinstance(self.payloads, dict):
            text = json.dumps(self.payloads.get(prompt_version, {}))
        else:
            text = json.dumps(
                {
                    "faithfulness": 5,
                    "groundedness": 4,
                    "answer_correctness": 5,
                    "relevance": 5,
                    "completeness": 4,
                    "citation_correctness": 4,
                    "citation_completeness": 3,
                    "abstention_accuracy": 5,
                    "reasoning": "supported by the cited passage",
                }
            )

        return GenerationResult(
            text=text,
            metadata=ModelMetadata(
                provider="groq",
                model_name="openai/gpt-oss-120b",
                model_version="gpt-oss-120b-2026",
                latency_ms=12.0,
            ),
        )

    async def embed(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    async def rerank(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError

    async def vision(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - unused
        raise NotImplementedError


def settings(**overrides: Any) -> AppSettings:
    base: dict[str, Any] = {
        "INFERENCE_PROFILE": "hosted",
        "EVAL_JUDGE_ENABLED": True,
        "EVAL_JUDGE_PROVIDER": "groq",
        "EVAL_JUDGE_SAMPLES": 1,
    }
    base.update(overrides)
    return AppSettings(**base)


def golden_question(**overrides: Any) -> GoldenQuestion:
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
        "acceptable_answer": "20 days per year.",
    }
    payload.update(overrides)
    return GoldenQuestion(**payload)


def answer_result(**overrides: Any) -> AnswerResult:
    payload: dict[str, Any] = {
        "query": "How much annual leave is accrued?",
        "answer": "Employees accrue 20 days per year [1].",
        "support": SupportState.GROUNDED,
    }
    payload.update(overrides)
    return AnswerResult(**payload)


class TestParsing:
    def test_plain_json(self) -> None:
        assert parse_judge_json('{"faithfulness": 4}') == {"faithfulness": 4}

    def test_markdown_fenced_json(self) -> None:
        # Models fence their output despite instructions not to; failing the
        # whole evaluation on formatting would be a poor trade.
        assert parse_judge_json('```json\n{"faithfulness": 4}\n```') == {"faithfulness": 4}

    def test_json_embedded_in_prose(self) -> None:
        text = 'Here is my assessment.\n{"faithfulness": 3}\nLet me know if you need more.'
        assert parse_judge_json(text) == {"faithfulness": 3}

    def test_unparseable_response_raises_rather_than_defaulting(self) -> None:
        # A neutral default would make a run in which the judge was broken look
        # like a run in which the system was mediocre.
        with pytest.raises(JudgeParseError):
            parse_judge_json("I am unable to score this.")

    def test_json_array_is_rejected(self) -> None:
        with pytest.raises(JudgeParseError, match="expected an object"):
            parse_judge_json("[1, 2, 3]")


class TestNormalizeScore:
    def test_maps_likert_onto_unit_interval(self) -> None:
        assert normalize_score(1) == 0.0
        assert normalize_score(3) == 0.5
        assert normalize_score(5) == 1.0

    def test_clamps_out_of_range_scores(self) -> None:
        assert normalize_score(9) == 1.0
        assert normalize_score(-2) == 0.0


class TestJudging:
    async def test_scores_every_dimension_and_records_provenance(self) -> None:
        gateway = RecordingGateway()
        judge = JudgeService(gateway=gateway, settings=settings())

        verdict = await judge.judge(golden_question(), answer_result(), "evidence text")

        assert verdict.scores["faithfulness"] == 1.0
        assert verdict.scores["completeness"] == pytest.approx(0.75)
        assert verdict.scores["citation_completeness"] == pytest.approx(0.5)
        assert verdict.scores["abstention_accuracy"] == 1.0
        assert verdict.judge_model == "openai/gpt-oss-120b"
        assert set(verdict.prompt_hashes) == {
            "judge_answer_v1",
            "judge_citation_v1",
            "judge_abstention_v1",
        }
        assert not verdict.errors

    async def test_judge_runs_on_the_configured_provider_not_the_generator(self) -> None:
        # A model scoring its own output has a documented self-preference bias,
        # and the dataset itself is partly LLM-drafted.
        gateway = RecordingGateway()
        judge = JudgeService(gateway=gateway, settings=settings(EVAL_JUDGE_PROVIDER="groq"))

        await judge.judge(golden_question(), answer_result(), "evidence")

        assert {call["provider"] for call in gateway.calls} == {"groq"}

    async def test_temperature_zero_and_prompt_version_are_passed_through(self) -> None:
        gateway = RecordingGateway()
        judge = JudgeService(gateway=gateway, settings=settings())

        await judge.judge(golden_question(), answer_result(), "evidence")

        assert {call["temperature"] for call in gateway.calls} == {0.0}
        assert {call["prompt_version"] for call in gateway.calls} == {
            "judge_answer_v1",
            "judge_citation_v1",
            "judge_abstention_v1",
        }

    async def test_unparseable_response_yields_no_score_and_an_error(self) -> None:
        gateway = RecordingGateway(payloads=["not json at all"])
        judge = JudgeService(gateway=gateway, settings=settings())

        verdict = await judge.judge(golden_question(), answer_result(), "evidence")

        assert verdict.scores == {}
        assert verdict.errors

    async def test_missing_dimension_is_recorded_rather_than_invented(self) -> None:
        gateway = RecordingGateway(
            payloads={
                "judge_answer_v1": {"faithfulness": 5, "reasoning": "ok"},
                "judge_citation_v1": {"citation_correctness": 4, "citation_completeness": 4},
                "judge_abstention_v1": {"abstention_accuracy": 5},
            }
        )
        judge = JudgeService(gateway=gateway, settings=settings())

        verdict = await judge.judge(golden_question(), answer_result(), "evidence")

        assert verdict.scores["faithfulness"] == 1.0
        assert "groundedness" not in verdict.scores
        assert any("groundedness" in error for error in verdict.errors)

    async def test_repeat_sampling_records_the_variance_band(self) -> None:
        # This is how the exit gate's "reproducible within a documented variance
        # band" is measured rather than asserted.
        payloads = [
            json.dumps({"faithfulness": 5, "groundedness": 5, "answer_correctness": 5, "relevance": 5, "completeness": 5}),
            json.dumps({"faithfulness": 3, "groundedness": 5, "answer_correctness": 5, "relevance": 5, "completeness": 5}),
        ]
        gateway = RecordingGateway(payloads=payloads)
        judge = JudgeService(gateway=gateway, settings=settings(EVAL_JUDGE_SAMPLES=2))

        verdict = await judge.judge(golden_question(), answer_result(), "evidence")

        assert verdict.raw_scores["faithfulness"] == pytest.approx(4.0)
        assert verdict.score_stdev["faithfulness"] > 0.0
        assert verdict.samples == 2

    async def test_stub_profile_is_refused(self) -> None:
        # Scoring canned text would produce numbers that look like evaluation
        # data without being it.
        gateway = RecordingGateway()
        judge = JudgeService(gateway=gateway, settings=settings(INFERENCE_PROFILE="stub"))

        verdict = await judge.judge(golden_question(), answer_result(), "evidence")

        assert verdict.scores == {}
        assert "judge_skipped_stub_profile" in verdict.errors
        assert gateway.calls == []

    async def test_disabled_judge_makes_no_calls(self) -> None:
        gateway = RecordingGateway()
        judge = JudgeService(gateway=gateway, settings=settings(EVAL_JUDGE_ENABLED=False))

        verdict = await judge.judge(golden_question(), answer_result(), "evidence")

        assert verdict.errors == ["judge_disabled"]
        assert gateway.calls == []
