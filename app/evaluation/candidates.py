"""Golden-dataset candidates promoted from user feedback (Task 13.4, master §55).

Feedback reaches the evaluation suite through two human gates, not one:

1. A reviewer promotes a piece of feedback in the review queue, writing the
   reference answer and choosing which evidence passages answer it. That creates a
   *candidate* here — schema-valid, but not yet part of any split.
2. A maintainer accepts candidates with ``python -m app.evaluation.cli
   accept-candidates``, which resolves their evidence against the corpus and only
   then appends them to a split.

The second gate exists because a split is a measuring instrument: a question
added to it changes the denominator of every metric computed from then on, and a
reviewer triaging feedback in a browser is not the right moment to make that call.
The locked test split can never receive a candidate.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from app.core.logging import get_logger
from app.evaluation.dataset import DATASET_DIR, DatasetError, dataset_path, load_split
from app.evaluation.schemas import DatasetSplit, GoldenQuestion

logger = get_logger("app.evaluation.candidates")

CANDIDATES_PATH = DATASET_DIR / "candidates" / "feedback_candidates_v1.jsonl"

#: Splits a candidate may target. The test split is opened once, at Stage 14.
CANDIDATE_SPLITS: frozenset[DatasetSplit] = frozenset({DatasetSplit.DEV, DatasetSplit.VALIDATION})


def load_candidates(path: Path | None = None) -> list[GoldenQuestion]:
    """Return every pending candidate, failing on the first malformed line."""
    path = path or CANDIDATES_PATH
    if not path.is_file():
        return []

    candidates: list[GoldenQuestion] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            candidates.append(GoldenQuestion.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise DatasetError(
                message=f"{path.name}:{line_number} is not a valid candidate: {exc}",
                details={"path": str(path), "line": line_number},
            ) from exc
    return candidates


def _write_candidates(candidates: list[GoldenQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(c.model_dump(mode="json", exclude_none=True), sort_keys=True) for c in candidates
    ]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def add_candidate(candidate: GoldenQuestion, path: Path | None = None) -> None:
    """Append one candidate, refusing a duplicate ID or a locked target split."""
    path = path or CANDIDATES_PATH
    if candidate.split not in CANDIDATE_SPLITS:
        raise DatasetError(
            message=f"Candidates cannot target the '{candidate.split.value}' split",
            details={"question_id": candidate.question_id},
        )
    existing = load_candidates(path)
    if any(c.question_id == candidate.question_id for c in existing):
        raise DatasetError(
            message=f"Candidate '{candidate.question_id}' already exists",
            details={"question_id": candidate.question_id},
        )
    _write_candidates([*existing, candidate], path)
    logger.info("candidate_added", question_id=candidate.question_id, split=candidate.split.value)


def remove_candidate(question_id: str, path: Path | None = None) -> bool:
    """Drop a pending candidate; returns False when it was not pending."""
    path = path or CANDIDATES_PATH
    existing = load_candidates(path)
    remaining = [c for c in existing if c.question_id != question_id]
    if len(remaining) == len(existing):
        return False
    _write_candidates(remaining, path)
    return True


def accept_candidates(
    question_ids: list[str] | None = None,
    path: Path | None = None,
    base_dir: Path | None = None,
    version: str = "v1",
) -> list[GoldenQuestion]:
    """Move candidates into their target splits; returns the ones accepted.

    Corpus resolution is the caller's job (the CLI runs it first), because it
    needs a database session. This function re-validates each split after the
    append by reloading it, so a duplicate ID or wrong-split record fails here
    rather than at the next experiment run.
    """
    path = path or CANDIDATES_PATH
    pending = load_candidates(path)
    chosen = [c for c in pending if question_ids is None or c.question_id in question_ids]
    unknown = set(question_ids or []) - {c.question_id for c in chosen}
    if unknown:
        raise DatasetError(
            message=f"Not pending: {', '.join(sorted(unknown))}",
            details={"question_ids": sorted(unknown)},
        )

    for split in {c.split for c in chosen}:
        target = dataset_path(split, version, base_dir)
        existing_ids = {q.question_id for q in load_split(split, version, base_dir)}
        additions = [c for c in chosen if c.split is split]
        clash = existing_ids & {c.question_id for c in additions}
        if clash:
            raise DatasetError(
                message=f"Already in {split.value}: {', '.join(sorted(clash))}",
                details={"question_ids": sorted(clash)},
            )
        with target.open("a", encoding="utf-8") as handle:
            for candidate in additions:
                record = candidate.model_dump(mode="json", exclude_none=True)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        load_split(split, version, base_dir)

    accepted_ids = {c.question_id for c in chosen}
    _write_candidates([c for c in pending if c.question_id not in accepted_ids], path)
    logger.info("candidates_accepted", count=len(chosen))
    return chosen
