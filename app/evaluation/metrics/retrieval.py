"""Deterministic retrieval metrics (Task 4.2, ADR-028, master §50).

No LLM is involved anywhere in this module. These numbers must be reproducible
byte-for-byte across runs, because they are the ones Stages 5 and 6 are judged
on and a metric with variance cannot settle an argument about a 2% change.

**Relevance is judged at element granularity, not chunk granularity.** A chunk
counts as relevant when it contains at least one element the dataset marked as
expected evidence. Keying on chunk IDs instead would make the metrics
incomparable across Stage 5's re-chunking, since chunk identity is derived from
``CHUNKING_VERSION`` (ADR-036) — the experiment would change its own ruler.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

#: K values reported by default. Top-K is 8 in Stage 3, so K=10 shows headroom
#: and K=1/3/5 show whether the answer is actually reachable near the top.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True)
class RetrievalCase:
    """One question's retrieval outcome, reduced to what the metrics need.

    ``retrieved_element_sets`` is in rank order, best first: entry *i* holds the
    element IDs of the chunk returned at rank *i+1*.
    """

    question_id: str
    retrieved_element_sets: Sequence[frozenset[str]]
    relevant_element_ids: frozenset[str]
    context_element_ids: frozenset[str] = field(default_factory=frozenset)

    @property
    def has_relevance_judgements(self) -> bool:
        """False for negative/adversarial questions, where retrieval metrics are undefined."""
        return bool(self.relevant_element_ids)

    def relevance_flags(self) -> list[bool]:
        """Per-rank relevance: True when that chunk carried expected evidence."""
        return [bool(s & self.relevant_element_ids) for s in self.retrieved_element_sets]


def hit_rate_at_k(case: RetrievalCase, k: int) -> float:
    """1.0 when any of the top-K chunks carries expected evidence, else 0.0."""
    return 1.0 if any(case.relevance_flags()[:k]) else 0.0


def precision_at_k(case: RetrievalCase, k: int) -> float:
    """Fraction of the top-K *returned* chunks that carry expected evidence.

    The denominator is how many chunks actually came back, not K. Dividing by K
    would penalise a retriever that correctly returns three good chunks when only
    three clear the score threshold — punishing precision for being selective.
    """
    flags = case.relevance_flags()[:k]
    if not flags:
        return 0.0
    return sum(flags) / len(flags)


def recall_at_k(case: RetrievalCase, k: int) -> float:
    """Fraction of expected evidence *elements* covered by the top-K chunks.

    Element-level rather than chunk-level: a question whose answer spans three
    elements is only fully answerable when all three are retrievable, and a
    chunk-level recall of 1.0 could hide two of them being missing.
    """
    if not case.relevant_element_ids:
        return 0.0
    covered: set[str] = set()
    for element_ids in case.retrieved_element_sets[:k]:
        covered |= element_ids & case.relevant_element_ids
    return len(covered) / len(case.relevant_element_ids)


def reciprocal_rank(case: RetrievalCase) -> float:
    """1/rank of the first relevant chunk, or 0.0 when none is relevant."""
    for index, is_relevant in enumerate(case.relevance_flags(), start=1):
        if is_relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(case: RetrievalCase, k: int) -> float:
    """Binary-gain nDCG@K with log2 discount.

    The ideal ranking is defined over the relevant chunks *present in the
    retrieved list*, capped at K. The alternative — an ideal built from every
    relevant chunk in the corpus — is not computable here, because the dataset
    records relevant elements rather than an exhaustive list of the chunks that
    contain them, and that list changes whenever chunking changes. This
    definition therefore measures ranking quality given what retrieval returned,
    and is read alongside recall, which measures what it failed to return.
    """
    flags = case.relevance_flags()
    dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(flags[:k], start=1) if hit)

    ideal_hits = min(sum(flags), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def context_precision(case: RetrievalCase, k: int | None = None) -> float:
    """Rank-weighted precision: mean of precision@i over each relevant rank i.

    This is average precision. It rewards putting evidence high in the context
    window, which matters because :mod:`app.generation.context` truncates the
    tail of the evidence list against a token budget — a relevant chunk at rank
    12 may never reach the model at all.
    """
    flags = case.relevance_flags() if k is None else case.relevance_flags()[:k]
    if not any(flags):
        return 0.0

    running_hits = 0
    precision_sum = 0.0
    for rank, is_relevant in enumerate(flags, start=1):
        if is_relevant:
            running_hits += 1
            precision_sum += running_hits / rank
    return precision_sum / running_hits


def context_recall(case: RetrievalCase) -> float:
    """Fraction of expected evidence elements present in the *assembled context*.

    Deliberately distinct from ``recall@k``: retrieval can succeed and assembly
    can still drop the evidence against the token budget. Reporting only recall
    would attribute that loss to the retriever and send Stage 6 chasing the wrong
    component.
    """
    if not case.relevant_element_ids:
        return 0.0
    covered = case.context_element_ids & case.relevant_element_ids
    return len(covered) / len(case.relevant_element_ids)


def compute_case_metrics(
    case: RetrievalCase,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Every retrieval metric for one question, keyed by metric name.

    Returns an empty mapping for questions with no relevance judgements
    (negative and adversarial types). Scoring them as 0.0 would be wrong in the
    other direction: it would drag the corpus-level averages down in proportion
    to how many correct-abstention cases the dataset contains, so adding good
    negative questions would look like a retrieval regression.
    """
    if not case.has_relevance_judgements:
        return {}

    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(case, k)
        metrics[f"precision@{k}"] = precision_at_k(case, k)
        metrics[f"recall@{k}"] = recall_at_k(case, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(case, k)

    metrics["mrr"] = reciprocal_rank(case)
    metrics["context_precision"] = context_precision(case)
    metrics["context_recall"] = context_recall(case)
    return metrics


def aggregate(
    per_case: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Macro-average each metric over the cases that reported it.

    Averaging per metric rather than requiring a uniform key set means a metric
    is never diluted by cases that could not produce it.
    """
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for case_metrics in per_case:
        for name, value in case_metrics.items():
            totals[name] = totals.get(name, 0.0) + value
            counts[name] = counts.get(name, 0) + 1

    return {name: totals[name] / counts[name] for name in sorted(totals)}
