"""The runner keeps the generator's draft so rejections can be diagnosed."""

from __future__ import annotations

from app.evaluation.results import QuestionResult
from app.evaluation.runner import ExperimentRunner
from app.evaluation.schemas import Difficulty, QuestionType
from app.generation.citation import SupportState
from app.generation.service import AnswerResult

REFUSAL = "I could not produce an answer that is verifiably supported by the HR knowledge base."


def blank_result() -> QuestionResult:
    return QuestionResult(
        question_id="dev-calculation-001",
        question="q",
        question_type=QuestionType.CALCULATION,
        difficulty=Difficulty.MEDIUM,
    )


class TestRecordAnswerKeepsTheDraft:
    """Experiment 003 rejected 15 answers whose evidence *was* retrieved.

    Only the replacement refusal was recorded, so nothing in the result file could
    say whether the model cited badly, cited nothing, or never answered at all.
    """

    def test_rejected_answer_records_draft_and_reason(self) -> None:
        answer = AnswerResult(
            query="q",
            answer=REFUSAL,
            support=SupportState.INSUFFICIENT,
            abstained=True,
            rejected=True,
            rejection_reason="Answer contains no citation resolving to supplied evidence",
            raw_answer="An employee accrues 15 days.\n\nSUPPORT: grounded",
            declared_support=SupportState.GROUNDED,
        )
        result = blank_result()

        ExperimentRunner._record_answer(result, answer)

        assert result.answer == REFUSAL
        assert result.raw_answer.startswith("An employee accrues 15 days.")
        assert result.declared_support == "grounded"
        assert result.rejection_reason is not None

    def test_draft_without_a_support_line_records_none(self) -> None:
        answer = AnswerResult(
            query="q", answer="a [1]", support=SupportState.GROUNDED, raw_answer="a [1]"
        )
        result = blank_result()

        ExperimentRunner._record_answer(result, answer)

        assert result.declared_support is None
        assert result.rejection_reason is None

    def test_results_written_before_the_fields_existed_still_load(self) -> None:
        legacy = blank_result().model_dump(
            exclude={"raw_answer", "declared_support", "rejection_reason"}
        )
        assert QuestionResult.model_validate(legacy).raw_answer == ""


class TestGeneratorMix:
    def test_run_reports_every_generator_that_answered(self) -> None:
        from app.evaluation.results import ExperimentRun
        from app.evaluation.schemas import DatasetSplit

        results = []
        for provider, model in [("gemini", "g"), ("groq", "o"), ("groq", "o"), (None, None)]:
            row = blank_result()
            row.generator_provider, row.generator_model = provider, model
            results.append(row)
        run = ExperimentRun(
            name="x", dataset_split=DatasetSplit.DEV, dataset_version="v1", results=results
        )

        assert run.generator_mix == {"gemini/g": 1, "groq/o": 2}
        assert run.mixes_generators

    def test_record_answer_copies_the_answering_provider(self) -> None:
        answer = AnswerResult(
            query="q",
            answer="a [1]",
            support=SupportState.GROUNDED,
            provider="groq",
            model_name="openai/gpt-oss-120b",
        )
        result = blank_result()

        ExperimentRunner._record_answer(result, answer)

        assert (result.generator_provider, result.generator_model) == (
            "groq",
            "openai/gpt-oss-120b",
        )
