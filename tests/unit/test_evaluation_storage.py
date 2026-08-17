"""Committed experiment records and candidate selection (Tasks 4.5, 4.6)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.evaluation import storage
from app.evaluation.results import ExperimentRun
from app.evaluation.schemas import DatasetSplit


def run(name: str, started_at: datetime, **overrides: Any) -> ExperimentRun:
    payload: dict[str, Any] = {
        "name": name,
        "started_at": started_at,
        "dataset_split": DatasetSplit.DEV,
        "dataset_version": "v1",
        "dataset_size": 100,
        "metrics": {"recall@5": 0.5},
    }
    payload.update(overrides)
    return ExperimentRun(**payload)


class TestRoundTrip:
    def test_a_saved_run_loads_back_identically(self, tmp_path: Path) -> None:
        original = run("experiment-001-baseline", datetime.now(UTC))
        storage.save_run_file(original, directory=tmp_path)

        loaded = storage.load_run_file("experiment-001-baseline", directory=tmp_path)

        assert loaded is not None
        assert loaded.run_id == original.run_id
        assert loaded.metrics == original.metrics

    def test_a_missing_run_is_none_rather_than_an_error(self, tmp_path: Path) -> None:
        assert storage.load_run_file("never-ran", directory=tmp_path) is None

    def test_names_with_path_separators_cannot_escape_the_results_directory(
        self, tmp_path: Path
    ) -> None:
        path = storage.result_path("../../etc/passwd", directory=tmp_path)

        assert path.parent == tmp_path


class TestLatestRunFile:
    def test_picks_the_most_recently_started_run(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        storage.save_run_file(run("experiment-001-baseline", now - timedelta(days=30)), tmp_path)
        storage.save_run_file(run("experiment-002-semantic", now - timedelta(days=2)), tmp_path)
        storage.save_run_file(run("experiment-003-hybrid", now), tmp_path)

        latest = storage.latest_run_file(directory=tmp_path)

        assert latest is not None
        assert latest.name == "experiment-003-hybrid"

    def test_ordering_ignores_filenames_and_mtimes(self, tmp_path: Path) -> None:
        # A fresh CI clone rewrites every mtime, and names are author-chosen; the
        # gate must still pick the experiment that actually ran last.
        now = datetime.now(UTC)
        storage.save_run_file(run("zzz-old-experiment", now - timedelta(days=10)), tmp_path)
        storage.save_run_file(run("aaa-new-experiment", now), tmp_path)

        latest = storage.latest_run_file(directory=tmp_path)

        assert latest is not None
        assert latest.name == "aaa-new-experiment"

    def test_excludes_the_baseline_so_the_gate_does_not_compare_it_to_itself(
        self, tmp_path: Path
    ) -> None:
        now = datetime.now(UTC)
        storage.save_run_file(run("experiment-001-baseline", now), tmp_path)
        storage.save_run_file(run("experiment-002-semantic", now - timedelta(days=1)), tmp_path)

        latest = storage.latest_run_file(exclude="experiment-001-baseline", directory=tmp_path)

        assert latest is not None
        assert latest.name == "experiment-002-semantic"

    def test_returns_none_when_only_the_excluded_run_exists(self, tmp_path: Path) -> None:
        storage.save_run_file(run("experiment-001-baseline", datetime.now(UTC)), tmp_path)

        assert storage.latest_run_file(exclude="experiment-001-baseline", directory=tmp_path) is None

    def test_returns_none_for_an_empty_results_directory(self, tmp_path: Path) -> None:
        assert storage.latest_run_file(directory=tmp_path) is None


class TestComparability:
    def test_a_subset_run_is_not_a_gate_candidate(self, tmp_path: Path) -> None:
        # A 20-question judged subset scoring lower than a 100-question baseline
        # is a different measurement, not a regression.
        now = datetime.now(UTC)
        baseline = run("experiment-001-baseline", now - timedelta(days=1))
        storage.save_run_file(baseline, tmp_path)
        storage.save_run_file(run("experiment-001-judged-subset", now, dataset_size=20), tmp_path)

        latest = storage.latest_run_file(
            exclude="experiment-001-baseline", directory=tmp_path, comparable_to=baseline
        )

        assert latest is None

    def test_a_run_over_the_same_split_and_size_is_a_candidate(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        baseline = run("experiment-001-baseline", now - timedelta(days=1))
        storage.save_run_file(baseline, tmp_path)
        storage.save_run_file(run("experiment-002-semantic", now), tmp_path)

        latest = storage.latest_run_file(
            exclude="experiment-001-baseline", directory=tmp_path, comparable_to=baseline
        )

        assert latest is not None
        assert latest.name == "experiment-002-semantic"

    def test_a_different_split_is_not_comparable(self, tmp_path: Path) -> None:
        baseline = run("experiment-001-baseline", datetime.now(UTC))
        other = run(
            "experiment-002-validation", datetime.now(UTC), dataset_split=DatasetSplit.VALIDATION
        )

        assert not storage.is_comparable(other, baseline)

    def test_a_different_dataset_version_is_not_comparable(self, tmp_path: Path) -> None:
        baseline = run("experiment-001-baseline", datetime.now(UTC))
        other = run("experiment-002-v2", datetime.now(UTC), dataset_version="v2")

        assert not storage.is_comparable(other, baseline)
