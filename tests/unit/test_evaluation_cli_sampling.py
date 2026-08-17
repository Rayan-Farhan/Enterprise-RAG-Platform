"""Comparable subset sampling for fast evaluation runs (Tasks 4.4, 4.6)."""

from __future__ import annotations

import uuid

from app.evaluation.cli import sample_per_type
from app.evaluation.schemas import (
    DatasetSplit,
    ExpectedEvidence,
    GoldenQuestion,
    QuestionType,
)

DOCUMENT_ID = uuid.uuid4()
VERSION_ID = uuid.uuid4()


def question(question_id: str, question_type: QuestionType) -> GoldenQuestion:
    abstains = question_type in {QuestionType.NEGATIVE_UNSUPPORTED, QuestionType.ADVERSARIAL}
    return GoldenQuestion(
        question_id=question_id,
        question="How much annual leave does a staff employee accrue?",
        question_type=question_type,
        split=DatasetSplit.DEV,
        acceptable_answer="10 days up to two years of service.",
        expected_evidence=[]
        if abstains
        else [
            ExpectedEvidence(
                document_id=DOCUMENT_ID,
                version_id=VERSION_ID,
                element_ids=["docling_elem_25_123"],
            )
        ],
    )


def split(per_type: int = 5) -> list[GoldenQuestion]:
    return [
        question(f"dev-{question_type.value}-{index:03d}", question_type)
        for question_type in QuestionType
        for index in range(1, per_type + 1)
    ]


class TestSamplePerType:
    def test_takes_n_of_every_type(self) -> None:
        sampled = sample_per_type(split(), per_type=2)

        counts: dict[str, int] = {}
        for item in sampled:
            counts[item.question_type.value] = counts.get(item.question_type.value, 0) + 1

        assert len(sampled) == 2 * len(QuestionType)
        assert set(counts.values()) == {2}

    def test_every_type_survives_where_a_plain_limit_would_drop_most(self) -> None:
        questions = split()
        # `--limit 20` on an ID-sorted split yields two types and no factual
        # questions at all — which is the reason this function exists.
        by_limit = {q.question_type for q in sorted(questions, key=lambda q: q.question_id)[:20]}
        by_type = {q.question_type for q in sample_per_type(questions, per_type=2)}

        assert len(by_limit) < len(QuestionType)
        assert by_type == set(QuestionType)

    def test_sampling_is_deterministic(self) -> None:
        questions = split()

        first = [q.question_id for q in sample_per_type(questions, per_type=3)]
        second = [q.question_id for q in sample_per_type(list(reversed(questions)), per_type=3)]

        assert first == second

    def test_a_type_with_fewer_questions_than_requested_contributes_all_of_them(self) -> None:
        questions = [
            question("dev-factual-001", QuestionType.FACTUAL),
            question("dev-factual-002", QuestionType.FACTUAL),
            question("dev-temporal-001", QuestionType.TEMPORAL),
        ]

        sampled = sample_per_type(questions, per_type=2)

        assert len(sampled) == 3

    def test_requesting_more_than_exists_returns_everything(self) -> None:
        questions = split(per_type=2)

        assert len(sample_per_type(questions, per_type=99)) == len(questions)
