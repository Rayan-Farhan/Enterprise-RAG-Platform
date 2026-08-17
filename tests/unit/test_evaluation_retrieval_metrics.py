"""Retrieval metrics against hand-computed fixtures (Task 4.2 exit criterion).

Every expected value below is written as the arithmetic that produces it rather
than as a decimal literal. A metric test whose expectations were produced by
running the code under test proves only that the code is deterministic.
"""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics.retrieval import (
    RetrievalCase,
    aggregate,
    compute_case_metrics,
    context_precision,
    context_recall,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def make_case(
    retrieved: list[set[str]],
    relevant: set[str],
    context: set[str] | None = None,
) -> RetrievalCase:
    return RetrievalCase(
        question_id="q-1",
        retrieved_element_sets=[frozenset(s) for s in retrieved],
        relevant_element_ids=frozenset(relevant),
        context_element_ids=frozenset(context or set()),
    )


@pytest.fixture
def worked_example() -> RetrievalCase:
    """Ranks 2 and 4 are relevant; three of four expected elements are retrieved.

        rank 1: {e9}        miss
        rank 2: {e1, e2}    hit
        rank 3: {e7}        miss
        rank 4: {e3}        hit
        rank 5: {e8}        miss

    Expected evidence is {e1, e2, e3, e4}; e4 is never retrieved. Only ranks 1-2
    survived context assembly.
    """
    return make_case(
        retrieved=[{"e9"}, {"e1", "e2"}, {"e7"}, {"e3"}, {"e8"}],
        relevant={"e1", "e2", "e3", "e4"},
        context={"e9", "e1", "e2"},
    )


class TestHitRate:
    def test_no_hit_in_top_1(self, worked_example: RetrievalCase) -> None:
        assert hit_rate_at_k(worked_example, 1) == 0.0

    def test_hit_appears_by_rank_2(self, worked_example: RetrievalCase) -> None:
        assert hit_rate_at_k(worked_example, 3) == 1.0
        assert hit_rate_at_k(worked_example, 5) == 1.0


class TestPrecision:
    def test_hand_computed(self, worked_example: RetrievalCase) -> None:
        assert precision_at_k(worked_example, 1) == 0 / 1
        assert precision_at_k(worked_example, 3) == pytest.approx(1 / 3)
        assert precision_at_k(worked_example, 5) == pytest.approx(2 / 5)

    def test_denominator_is_what_came_back_not_k(self) -> None:
        # Three chunks returned, all relevant: precision@10 is 1.0, not 0.3.
        # Dividing by K would punish a retriever for being selective.
        case = make_case([{"a"}, {"b"}, {"c"}], relevant={"a", "b", "c"})
        assert precision_at_k(case, 10) == 1.0


class TestRecall:
    def test_hand_computed(self, worked_example: RetrievalCase) -> None:
        assert recall_at_k(worked_example, 1) == 0 / 4
        assert recall_at_k(worked_example, 3) == pytest.approx(2 / 4)
        assert recall_at_k(worked_example, 5) == pytest.approx(3 / 4)

    def test_is_element_level_not_chunk_level(self) -> None:
        # One chunk carries one of three expected elements. Chunk-level recall
        # would say 1.0 and hide the two missing pieces of the answer.
        case = make_case([{"e1"}], relevant={"e1", "e2", "e3"})
        assert recall_at_k(case, 5) == pytest.approx(1 / 3)

    def test_duplicate_evidence_across_chunks_is_not_double_counted(self) -> None:
        case = make_case([{"e1"}, {"e1"}, {"e1"}], relevant={"e1", "e2"})
        assert recall_at_k(case, 3) == pytest.approx(1 / 2)


class TestReciprocalRank:
    def test_first_relevant_is_rank_2(self, worked_example: RetrievalCase) -> None:
        assert reciprocal_rank(worked_example) == pytest.approx(1 / 2)

    def test_zero_when_nothing_relevant(self) -> None:
        assert reciprocal_rank(make_case([{"x"}], relevant={"y"})) == 0.0


class TestNDCG:
    def test_hand_computed_at_5(self, worked_example: RetrievalCase) -> None:
        dcg = 1 / math.log2(2 + 1) + 1 / math.log2(4 + 1)
        idcg = 1 / math.log2(1 + 1) + 1 / math.log2(2 + 1)
        assert ndcg_at_k(worked_example, 5) == pytest.approx(dcg / idcg)

    def test_perfect_ranking_scores_one(self) -> None:
        case = make_case([{"e1"}, {"e2"}, {"x"}], relevant={"e1", "e2"})
        assert ndcg_at_k(case, 3) == pytest.approx(1.0)

    def test_zero_when_no_relevant_chunk_retrieved(self) -> None:
        assert ndcg_at_k(make_case([{"x"}, {"y"}], relevant={"e1"}), 5) == 0.0


class TestContextMetrics:
    def test_context_precision_is_average_precision(
        self, worked_example: RetrievalCase
    ) -> None:
        # hits at ranks 2 and 4 -> mean of (1/2, 2/4)
        assert context_precision(worked_example) == pytest.approx((1 / 2 + 2 / 4) / 2)

    def test_context_recall_measures_the_assembled_context_not_retrieval(
        self, worked_example: RetrievalCase
    ) -> None:
        # e3 was retrieved at rank 4 but dropped before assembly, so context
        # recall (2/4) is strictly below recall@5 (3/4). Reporting only recall
        # would blame the retriever for a loss the assembler caused.
        assert context_recall(worked_example) == pytest.approx(2 / 4)
        assert recall_at_k(worked_example, 5) == pytest.approx(3 / 4)


class TestCaseAggregation:
    def test_questions_without_judgements_report_nothing(self) -> None:
        # A correct abstention must not be scored 0.0 on recall; adding good
        # negative questions would otherwise look like a retrieval regression.
        negative = make_case([{"e1"}], relevant=set())
        assert compute_case_metrics(negative) == {}

    def test_metric_names_cover_every_k(self, worked_example: RetrievalCase) -> None:
        metrics = compute_case_metrics(worked_example, k_values=(1, 5))
        assert set(metrics) == {
            "hit_rate@1",
            "precision@1",
            "recall@1",
            "ndcg@1",
            "hit_rate@5",
            "precision@5",
            "recall@5",
            "ndcg@5",
            "mrr",
            "context_precision",
            "context_recall",
        }

    def test_aggregate_averages_per_metric_over_reporting_cases(self) -> None:
        # 'mrr' appears in both cases, 'recall@5' in only one: each is averaged
        # over the cases that reported it, not over the case count.
        aggregated = aggregate([{"mrr": 1.0, "recall@5": 0.5}, {"mrr": 0.0}])
        assert aggregated["mrr"] == pytest.approx(0.5)
        assert aggregated["recall@5"] == pytest.approx(0.5)

    def test_aggregate_of_nothing_is_empty(self) -> None:
        assert aggregate([]) == {}
