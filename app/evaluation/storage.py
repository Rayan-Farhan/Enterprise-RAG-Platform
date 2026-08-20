"""Committed experiment result files (Task 4.5).

PostgreSQL is the queryable index over experiments; these JSON files are the
durable record. They are committed to the repository because the Stage 4 exit
gate requires ``experiment-001-baseline`` to be citable by every later stage,
and a number that only exists in a developer's local database is not citable.

They are also what the CI regression gate reads: CI has no access to a
developer's PostgreSQL, and requiring one would make the gate skip itself
exactly when it matters.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.core.logging import get_logger
from app.evaluation.results import ExperimentRun, QuestionResult

logger = get_logger("app.evaluation.storage")

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Partial runs live beside the finished records but out of the way of
#: `list_run_files`, so an interrupted run can never be mistaken for a result.
CHECKPOINT_DIRNAME = ".checkpoints"


def results_dir(configured: str = "evaluation/results") -> Path:
    """Resolve the results directory relative to the repository root."""
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def result_path(name: str, directory: Path | None = None) -> Path:
    """Path of the committed record for a named experiment."""
    safe = name.replace("/", "-").replace("\\", "-")
    return (directory or results_dir()) / f"{safe}.json"


def save_run_file(run: ExperimentRun, directory: Path | None = None) -> Path:
    """Write the full run record as indented, key-sorted JSON.

    Indented and sorted so that re-running an experiment produces a reviewable
    diff. A compact dump would make every rerun an unreadable single-line change,
    and the whole point of committing these is that a human can see what moved.
    """
    path = result_path(run.name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    logger.info("experiment_file_written", path=str(path), name=run.name)
    return path


def load_run_file(name: str, directory: Path | None = None) -> ExperimentRun | None:
    """Load a committed run record by experiment name, or None when absent."""
    path = result_path(name, directory)
    if not path.is_file():
        return None
    return ExperimentRun.model_validate_json(path.read_text(encoding="utf-8"))


def list_run_files(directory: Path | None = None) -> list[str]:
    """Every committed experiment name, sorted."""
    target = directory or results_dir()
    if not target.is_dir():
        return []
    return sorted(p.stem for p in target.glob("*.json"))


# --------------------------------------------------------------------------
# Checkpoints — multi-day runs on a metered provider
# --------------------------------------------------------------------------


def checkpoint_path(name: str, directory: Path | None = None) -> Path:
    """Path of the append-only partial-results file for a named experiment."""
    safe = name.replace("/", "-").replace("\\", "-")
    return (directory or results_dir()) / CHECKPOINT_DIRNAME / f"{safe}.jsonl"


def append_checkpoint(
    name: str,
    result: QuestionResult,
    directory: Path | None = None,
) -> None:
    """Record one evaluated question, immediately.

    Append-only JSONL, flushed per line, because the failure mode this exists for
    is the process dying mid-run when the daily token budget runs out. A format
    that has to be rewritten whole (or held in memory until the end) loses
    everything that was already paid for.

    Quota failures are never written: the question was refused, not measured.
    """
    if result.failed_on_quota:
        return

    path = checkpoint_path(name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json() + "\n")
        handle.flush()


def load_checkpoint(name: str, directory: Path | None = None) -> dict[str, QuestionResult]:
    """Read a partial run back, keyed by ``question_id``.

    A malformed trailing line is dropped rather than fatal: it is what a process
    killed mid-write leaves behind, and losing one question is better than losing
    the day's work. Later entries win, so re-evaluating a question overwrites it.
    """
    path = checkpoint_path(name, directory)
    if not path.is_file():
        return {}

    results: dict[str, QuestionResult] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            result = QuestionResult.model_validate_json(line)
        except ValueError:
            logger.warning("checkpoint_line_unreadable", path=str(path), line=line_number)
            continue
        results[result.question_id] = result

    logger.info("checkpoint_loaded", name=name, questions=len(results))
    return results


def clear_checkpoint(name: str, directory: Path | None = None) -> None:
    """Remove a completed run's checkpoint."""
    path = checkpoint_path(name, directory)
    if path.is_file():
        path.unlink()
        logger.info("checkpoint_cleared", name=name)


def is_comparable(run: ExperimentRun, reference: ExperimentRun) -> bool:
    """True when two runs were scored over the same questions.

    A run over a different split, dataset version, or question count is not a
    worse or better system — it is a different measurement. Comparing them would
    report a subset run as a catastrophic regression, which is how a gate earns
    a reputation for crying wolf.
    """
    return (
        run.dataset_split == reference.dataset_split
        and run.dataset_version == reference.dataset_version
        and run.dataset_size == reference.dataset_size
        and run.retrieval_only == reference.retrieval_only
    )


def latest_run_file(
    exclude: str | None = None,
    directory: Path | None = None,
    comparable_to: ExperimentRun | None = None,
) -> ExperimentRun | None:
    """The most recently *started* committed run, filtered to usable candidates.

    Ordered by ``started_at`` rather than by filename or mtime: filenames are
    author-chosen and a checkout rewrites every mtime, so on a fresh CI clone
    both would pick an arbitrary run. This is how the regression gate finds "the
    experiment this branch is proposing" without the branch having to name it.
    """
    runs = [load_run_file(name, directory) for name in list_run_files(directory)]
    candidates = [
        run
        for run in runs
        if run is not None
        and run.name != exclude
        and (comparable_to is None or is_comparable(run, comparable_to))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda run: run.started_at)
