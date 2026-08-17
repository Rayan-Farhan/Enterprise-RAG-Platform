"""Layer 3 — human review export and import (Task 4.3, master §51-52).

Master §51 says not to rely on a single automatic evaluator. That is only
actionable if there is a route for a human verdict to enter the record, and if
the cases put in front of the human are the ones worth their time.

Reviewing a random sample would spend most of that time confirming easy factual
questions the system already gets right. :func:`select_for_review` instead ranks
by how *informative* a human verdict would be: cases where the automatic layers
disagree with each other come first, because one of them is wrong and only a
person can say which.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.core.exceptions import ValidationException
from app.core.logging import get_logger
from app.evaluation.results import ExperimentRun, HumanVerdict, QuestionResult
from app.evaluation.schemas import QuestionType

logger = get_logger("app.evaluation.human_review")

#: Types whose correctness a deterministic check cannot settle. Ambiguous
#: questions have no single right answer, and adversarial ones are judged on
#: behaviour under attack, which is precisely what an automatic judge is worst at.
ALWAYS_REVIEW_TYPES: frozenset[QuestionType] = frozenset(
    {QuestionType.AMBIGUOUS, QuestionType.ADVERSARIAL, QuestionType.CONFLICTING_VERSIONS}
)


def review_priority(result: QuestionResult) -> float:
    """Score how much a human verdict on this case would tell us. Higher is sooner."""
    priority = 0.0

    # A pipeline failure or a rejected answer is always worth a look.
    if result.failed:
        priority += 10.0
    if result.rejected:
        priority += 8.0

    # Disagreement between layers: the deterministic checks say the citation
    # points at the right element, the judge says it does not support the claim
    # (or vice versa). Exactly one of them is wrong.
    element_match = result.deterministic_metrics.get("element_match")
    judge_citation = result.judge_metrics.get("judge_citation_correctness")
    if element_match is not None and judge_citation is not None:
        priority += 6.0 * abs(element_match - judge_citation)

    # Answered when it should have abstained is the most damaging failure mode,
    # so it outranks the reverse.
    if result.deterministic_metrics.get("abstention_correct") == 0.0:
        priority += 5.0 if not result.abstained else 3.0

    # Types where automatic scoring is least trustworthy.
    if result.question_type in ALWAYS_REVIEW_TYPES:
        priority += 4.0

    # Low faithfulness is a candidate hallucination.
    faithfulness = result.judge_metrics.get("judge_faithfulness")
    if faithfulness is not None:
        priority += 4.0 * (1.0 - faithfulness)

    # A judge that could not produce a score leaves a hole only a human fills.
    if result.judge_errors:
        priority += 3.0

    return priority


def select_for_review(
    run: ExperimentRun,
    limit: int = 25,
) -> list[QuestionResult]:
    """Return the cases most worth a human verdict, highest priority first."""
    ranked = sorted(
        run.results,
        key=lambda r: (-review_priority(r), r.question_id),
    )
    return ranked[:limit]


def export_review_batch(
    run: ExperimentRun,
    path: Path,
    limit: int = 25,
) -> Path:
    """Write a reviewable JSONL batch with blank verdict fields to fill in.

    JSONL rather than CSV: the citations and judge reasoning are nested and
    multi-line, and flattening them into cells is how reviewers end up judging a
    truncated version of the answer.
    """
    selected = select_for_review(run, limit=limit)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in selected:
            record = {
                "run_id": str(run.run_id),
                "question_id": result.question_id,
                "question": result.question,
                "question_type": result.question_type.value,
                "system_answer": result.answer,
                "abstained": result.abstained,
                "rejected": result.rejected,
                "citation_markers": result.citation_markers,
                "automatic_scores": {
                    **result.deterministic_metrics,
                    **result.judge_metrics,
                },
                "judge_reasoning": result.judge_reasoning,
                "review_priority": round(review_priority(result), 3),
                # --- fill these in ---
                "reviewer": "",
                "is_correct": None,
                "is_faithful": None,
                "citations_are_correct": None,
                "abstention_was_right": None,
                "severity": None,
                "comment": "",
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("review_batch_exported", path=str(path), cases=len(selected))
    return path


def import_review_batch(path: Path) -> list[HumanVerdict]:
    """Read a completed review batch back into verdicts.

    Rows with no reviewer name are skipped rather than rejected — a partially
    completed batch is the normal case, and refusing the whole file because two
    rows are blank would make reviewing in several sittings impossible.
    """
    if not path.is_file():
        raise ValidationException(
            message=f"Review batch not found: {path}", details={"path": str(path)}
        )

    verdicts: list[HumanVerdict] = []
    skipped = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationException(
                    message=f"{path.name}:{line_number} is not valid JSON: {exc}",
                    details={"path": str(path), "line": line_number},
                ) from exc

            if not str(record.get("reviewer") or "").strip():
                skipped += 1
                continue

            try:
                verdicts.append(
                    HumanVerdict(
                        run_id=uuid.UUID(str(record["run_id"])),
                        question_id=str(record["question_id"]),
                        reviewer=str(record["reviewer"]).strip(),
                        is_correct=record.get("is_correct"),
                        is_faithful=record.get("is_faithful"),
                        citations_are_correct=record.get("citations_are_correct"),
                        abstention_was_right=record.get("abstention_was_right"),
                        severity=record.get("severity"),
                        comment=str(record.get("comment") or ""),
                    )
                )
            except (KeyError, ValueError, ValidationError) as exc:
                raise ValidationException(
                    message=f"{path.name}:{line_number} is not a usable verdict: {exc}",
                    details={"path": str(path), "line": line_number},
                ) from exc

    logger.info("review_batch_imported", path=str(path), verdicts=len(verdicts), skipped=skipped)
    return verdicts


def agreement_report(
    run: ExperimentRun,
    verdicts: Sequence[HumanVerdict],
) -> dict[str, float]:
    """Measure how often the automatic layers agreed with the human reviewer.

    This is the number that says whether Layer 2 can be trusted to run unattended
    between human review cycles. It is reported, never used to adjust the judge's
    scores — a judge tuned to agree with the reviewers it is checked against
    stops being an independent signal.
    """
    by_id = {r.question_id: r for r in run.results}
    pairs: list[tuple[str, float, float]] = []

    for verdict in verdicts:
        result = by_id.get(verdict.question_id)
        if result is None:
            continue

        if verdict.is_faithful is not None:
            judged = result.judge_metrics.get("judge_faithfulness")
            if judged is not None:
                pairs.append(
                    ("faithfulness", float(verdict.is_faithful), 1.0 if judged >= 0.75 else 0.0)
                )

        if verdict.citations_are_correct is not None:
            judged = result.judge_metrics.get("judge_citation_correctness")
            if judged is not None:
                pairs.append(
                    (
                        "citation_correctness",
                        float(verdict.citations_are_correct),
                        1.0 if judged >= 0.75 else 0.0,
                    )
                )

        if verdict.abstention_was_right is not None:
            deterministic = result.deterministic_metrics.get("abstention_correct")
            if deterministic is not None:
                pairs.append(("abstention", float(verdict.abstention_was_right), deterministic))

    report: dict[str, float] = {"human_reviewed_cases": float(len(verdicts))}
    for dimension in ("faithfulness", "citation_correctness", "abstention"):
        matched = [(h, a) for name, h, a in pairs if name == dimension]
        if matched:
            report[f"agreement_{dimension}"] = sum(1.0 for h, a in matched if h == a) / len(matched)
    return report
