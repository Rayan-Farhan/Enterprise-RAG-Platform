"""Layer 3 human review selection, export and import (Task 4.3)."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from app.core.exceptions import ValidationException
from app.evaluation.human_review import (
    agreement_report,
    export_review_batch,
    import_review_batch,
    review_priority,
    select_for_review,
)
from app.evaluation.results import ExperimentRun, QuestionResult
from app.evaluation.schemas import DatasetSplit, Difficulty, QuestionType


def result(**overrides: Any) -> QuestionResult:
    payload: dict[str, Any] = {
        "question_id": "dev-factual-0001",
        "question": "How much annual leave?",
        "question_type": QuestionType.FACTUAL,
        "difficulty": Difficulty.EASY,
        "answer": "20 days [1].",
        "deterministic_metrics": {"abstention_correct": 1.0, "element_match": 1.0},
        "judge_metrics": {"judge_faithfulness": 1.0, "judge_citation_correctness": 1.0},
    }
    payload.update(overrides)
    return QuestionResult(**payload)


def run(results: list[QuestionResult]) -> ExperimentRun:
    return ExperimentRun(
        name="experiment-001-baseline",
        dataset_split=DatasetSplit.DEV,
        dataset_version="v1",
        dataset_size=len(results),
        results=results,
    )


class TestPriority:
    def test_a_clean_result_scores_zero(self) -> None:
        assert review_priority(result()) == pytest.approx(0.0)

    def test_a_pipeline_failure_outranks_everything_else(self) -> None:
        assert review_priority(result(error="TimeoutError")) > review_priority(
            result(rejected=True)
        )

    def test_layer_disagreement_raises_priority(self) -> None:
        # The deterministic check says the citation points at the right element;
        # the judge says it does not support the claim. One of them is wrong and
        # only a person can say which.
        disagreement = result(
            deterministic_metrics={"abstention_correct": 1.0, "element_match": 1.0},
            judge_metrics={"judge_faithfulness": 1.0, "judge_citation_correctness": 0.0},
        )
        assert review_priority(disagreement) == pytest.approx(6.0)

    def test_answering_when_it_should_have_abstained_outranks_the_reverse(self) -> None:
        answered_wrongly = result(
            abstained=False, deterministic_metrics={"abstention_correct": 0.0}
        )
        refused_wrongly = result(
            abstained=True, deterministic_metrics={"abstention_correct": 0.0}
        )
        assert review_priority(answered_wrongly) > review_priority(refused_wrongly)

    def test_types_automatic_scoring_handles_worst_are_promoted(self) -> None:
        adversarial = result(question_type=QuestionType.ADVERSARIAL)
        assert review_priority(adversarial) > review_priority(result())

    def test_a_judge_error_leaves_a_hole_only_a_human_fills(self) -> None:
        assert review_priority(result(judge_errors=["parse failure"])) == pytest.approx(3.0)


class TestSelection:
    def test_highest_priority_first_then_stable_by_id(self) -> None:
        selected = select_for_review(
            run(
                [
                    result(question_id="dev-c"),
                    result(question_id="dev-a", error="boom"),
                    result(question_id="dev-b"),
                ]
            ),
            limit=3,
        )
        assert [r.question_id for r in selected] == ["dev-a", "dev-b", "dev-c"]

    def test_limit_is_honoured(self) -> None:
        results = [result(question_id=f"dev-{i:03d}") for i in range(10)]
        assert len(select_for_review(run(results), limit=4)) == 4


class TestExportImport:
    def test_round_trip(self, tmp_path: Path) -> None:
        experiment = run([result(question_id="dev-a", error="boom"), result()])
        path = export_review_batch(experiment, tmp_path / "batch.jsonl", limit=2)

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert rows[0]["question_id"] == "dev-a"
        assert rows[0]["reviewer"] == ""

        rows[0]["reviewer"] = "rayan"
        rows[0]["is_faithful"] = False
        rows[0]["citations_are_correct"] = True
        rows[0]["comment"] = "cites a real page but the wrong clause"
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

        verdicts = import_review_batch(path)
        assert len(verdicts) == 1
        assert verdicts[0].reviewer == "rayan"
        assert verdicts[0].is_faithful is False
        assert verdicts[0].run_id == experiment.run_id

    def test_unreviewed_rows_are_skipped_not_rejected(self, tmp_path: Path) -> None:
        # Reviewing in several sittings is the normal case; refusing the whole
        # file because two rows are blank would make that impossible.
        experiment = run([result(question_id="dev-a"), result(question_id="dev-b")])
        path = export_review_batch(experiment, tmp_path / "batch.jsonl", limit=2)

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["reviewer"] = "rayan"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        assert len(import_review_batch(path)) == 1

    def test_missing_file_is_reported_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationException, match="not found"):
            import_review_batch(tmp_path / "absent.jsonl")

    def test_malformed_line_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "batch.jsonl"
        path.write_text('{"reviewer": "rayan"\n', encoding="utf-8")
        with pytest.raises(ValidationException, match="not valid JSON"):
            import_review_batch(path)


class TestAgreement:
    def test_reports_where_the_judge_matched_the_human(self) -> None:
        experiment = run(
            [
                result(
                    question_id="dev-a",
                    judge_metrics={
                        "judge_faithfulness": 1.0,
                        "judge_citation_correctness": 1.0,
                    },
                ),
                result(
                    question_id="dev-b",
                    judge_metrics={
                        "judge_faithfulness": 0.25,
                        "judge_citation_correctness": 1.0,
                    },
                ),
            ]
        )
        verdicts = import_verdicts(
            experiment.run_id,
            [("dev-a", True, True), ("dev-b", True, False)],
        )

        report = agreement_report(experiment, verdicts)
        # dev-a: judge faithful, human faithful -> agree.
        # dev-b: judge unfaithful (0.25), human faithful -> disagree.
        assert report["agreement_faithfulness"] == pytest.approx(0.5)
        # Citations: judge says correct for both; human agrees on a, not on b.
        assert report["agreement_citation_correctness"] == pytest.approx(0.5)
        assert report["human_reviewed_cases"] == 2.0

    def test_verdicts_for_unknown_questions_are_ignored(self) -> None:
        experiment = run([result(question_id="dev-a")])
        verdicts = import_verdicts(experiment.run_id, [("dev-missing", True, True)])
        report = agreement_report(experiment, verdicts)
        assert "agreement_faithfulness" not in report


def import_verdicts(
    run_id: uuid.UUID,
    rows: list[tuple[str, bool, bool]],
) -> list[Any]:
    from app.evaluation.results import HumanVerdict

    return [
        HumanVerdict(
            run_id=run_id,
            question_id=question_id,
            reviewer="rayan",
            is_faithful=is_faithful,
            citations_are_correct=citations_correct,
        )
        for question_id, is_faithful, citations_correct in rows
    ]
