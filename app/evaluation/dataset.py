"""Golden dataset loading and validation (Task 4.1, master §48-49).

Loading is strict and validation is against the live corpus, because the two
failure modes of an evaluation dataset are both silent. A malformed record that
is skipped shrinks the split without anyone noticing; an evidence pointer that
does not resolve makes a perfect retrieval look like a miss. Both would move
every metric from Stage 5 onward in a direction nobody could explain.
"""

from __future__ import annotations

import json
import uuid
from collections import Counter
from pathlib import Path

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ValidationException
from app.core.logging import get_logger
from app.db.models.element import Element
from app.db.models.version import DocumentVersion
from app.evaluation.schemas import (
    DatasetSplit,
    DatasetStats,
    DatasetValidationReport,
    EvidenceResolutionIssue,
    GoldenQuestion,
    QuestionType,
)

logger = get_logger("app.evaluation.dataset")

DATASET_DIR = Path(__file__).resolve().parents[2] / "evaluation" / "datasets"

#: Set by the Stage 14 final-readiness run and by nothing else. Reading the test
#: split without it raises: an accidental `--split test` during tuning would
#: quietly contaminate the one number the whole roadmap builds toward.
TEST_SPLIT_UNLOCK_TOKEN = "stage-14-final-readiness"


class DatasetError(ValidationException):
    """Raised when a dataset file is missing, malformed, or improperly accessed."""


def dataset_path(split: DatasetSplit, version: str = "v1", base_dir: Path | None = None) -> Path:
    """Return the JSONL path for a split, e.g. ``golden_dataset_dev_v1.jsonl``."""
    return (base_dir or DATASET_DIR) / f"golden_dataset_{split.value}_{version}.jsonl"


def load_split(
    split: DatasetSplit,
    version: str = "v1",
    base_dir: Path | None = None,
    unlock_token: str | None = None,
) -> list[GoldenQuestion]:
    """Load and parse one split, failing on the first malformed record.

    Parsing is fail-fast rather than skip-and-warn. A skipped record changes the
    denominator of every metric computed from the split, and a warning in a log
    is not a strong enough signal for that.
    """
    if split is DatasetSplit.TEST and unlock_token != TEST_SPLIT_UNLOCK_TOKEN:
        raise DatasetError(
            message=(
                "The test split is locked and is opened exactly once, at Stage 14. "
                "Tuning against it would make the final readiness number a measurement "
                "of the tuning rather than of the system."
            ),
            details={"split": split.value},
        )

    path = dataset_path(split, version, base_dir)
    if not path.is_file():
        raise DatasetError(
            message=f"Golden dataset not found: {path}",
            details={"split": split.value, "version": version, "path": str(path)},
        )

    questions: list[GoldenQuestion] = []
    seen_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(
                    message=f"{path.name}:{line_number} is not valid JSON: {exc}",
                    details={"path": str(path), "line": line_number},
                ) from exc

            try:
                question = GoldenQuestion.model_validate(record)
            except ValidationError as exc:
                raise DatasetError(
                    message=f"{path.name}:{line_number} failed schema validation: {exc}",
                    details={"path": str(path), "line": line_number},
                ) from exc

            if question.split is not split:
                raise DatasetError(
                    message=(
                        f"{path.name}:{line_number} declares split "
                        f"'{question.split.value}' but lives in the '{split.value}' file"
                    ),
                    details={"path": str(path), "line": line_number},
                )
            if question.question_id in seen_ids:
                raise DatasetError(
                    message=f"{path.name}:{line_number} duplicates question_id "
                    f"'{question.question_id}'",
                    details={"path": str(path), "line": line_number},
                )

            seen_ids.add(question.question_id)
            questions.append(question)

    logger.info("dataset_loaded", split=split.value, version=version, questions=len(questions))
    return questions


def summarize(
    questions: list[GoldenQuestion],
    split: DatasetSplit,
    version: str = "v1",
) -> DatasetStats:
    """Compute the composition of a split, including which types are absent."""
    present_types = {q.question_type for q in questions}
    documents = {doc_id for q in questions for doc_id in q.expected_document_ids()}

    return DatasetStats(
        split=split,
        version=version,
        total=len(questions),
        by_type=dict(sorted(Counter(q.question_type.value for q in questions).items())),
        by_difficulty=dict(sorted(Counter(q.difficulty.value for q in questions).items())),
        by_source=dict(sorted(Counter(q.source for q in questions).items())),
        abstention_cases=sum(1 for q in questions if q.is_abstention_case),
        documents_covered=len(documents),
        missing_types=sorted(t.value for t in QuestionType if t not in present_types),
    )


async def validate_against_corpus(
    questions: list[GoldenQuestion],
    session: AsyncSession,
    split: DatasetSplit,
    version: str = "v1",
) -> DatasetValidationReport:
    """Resolve every ``expected_evidence`` pointer against the ingested corpus.

    Every pointer is checked in one pass and all failures are reported together,
    because fixing a generated dataset one error per run is not workable.
    """
    stats = summarize(questions, split=split, version=version)
    issues: list[EvidenceResolutionIssue] = []

    version_ids = {ev.version_id for q in questions for ev in q.expected_evidence}
    if not version_ids:
        return DatasetValidationReport(stats=stats, issues=issues)

    version_rows = (
        await session.execute(
            select(DocumentVersion.id, DocumentVersion.document_id).where(
                DocumentVersion.id.in_(version_ids)
            )
        )
    ).all()
    version_to_document: dict[uuid.UUID, uuid.UUID] = {
        row[0]: row[1]
        for row in version_rows  # noqa: PD011 - Row tuple, not a Series
    }

    element_rows = (
        await session.execute(
            select(Element.version_id, Element.element_id).where(
                Element.version_id.in_(version_ids)
            )
        )
    ).all()
    elements_by_version: dict[uuid.UUID, set[str]] = {}
    for version_id, element_id in element_rows:
        elements_by_version.setdefault(version_id, set()).add(element_id)

    for question in questions:
        for evidence in question.expected_evidence:
            known_document = version_to_document.get(evidence.version_id)
            if known_document is None:
                issues.append(
                    EvidenceResolutionIssue(
                        question_id=question.question_id,
                        reason="version_not_found",
                        detail={"version_id": str(evidence.version_id)},
                    )
                )
                continue

            if known_document != evidence.document_id:
                # A mismatched pair still resolves element-by-element, so this
                # would otherwise pass while attributing evidence to the wrong
                # document in every per-document metric.
                issues.append(
                    EvidenceResolutionIssue(
                        question_id=question.question_id,
                        reason="document_version_mismatch",
                        detail={
                            "version_id": str(evidence.version_id),
                            "declared_document_id": str(evidence.document_id),
                            "actual_document_id": str(known_document),
                        },
                    )
                )
                continue

            known_elements = elements_by_version.get(evidence.version_id, set())
            missing = sorted(set(evidence.element_ids) - known_elements)
            if missing:
                issues.append(
                    EvidenceResolutionIssue(
                        question_id=question.question_id,
                        reason="element_not_found",
                        detail={
                            "version_id": str(evidence.version_id),
                            "missing_element_ids": missing,
                        },
                    )
                )

    logger.info(
        "dataset_validated",
        split=split.value,
        questions=len(questions),
        issues=len(issues),
        missing_types=stats.missing_types,
    )
    return DatasetValidationReport(stats=stats, issues=issues)


def write_split(
    questions: list[GoldenQuestion],
    split: DatasetSplit,
    version: str = "v1",
    base_dir: Path | None = None,
) -> Path:
    """Write a split as JSONL, one question per line, with stable key ordering.

    Stable ordering keeps a regenerated dataset's diff reviewable — an unordered
    dump would show every line as changed and hide the questions that actually
    moved.
    """
    path = dataset_path(split, version, base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        json.dumps(q.model_dump(mode="json", exclude_none=True), sort_keys=True)
        for q in sorted(questions, key=lambda q: q.question_id)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    logger.info("dataset_written", path=str(path), questions=len(questions))
    return path
