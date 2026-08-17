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
from app.evaluation.results import ExperimentRun

logger = get_logger("app.evaluation.storage")

REPO_ROOT = Path(__file__).resolve().parents[2]


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
