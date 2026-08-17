"""Evaluation command line interface (Tasks 4.4, 4.6).

Driven through the Makefile:

    make eval CONFIG=experiment-001-baseline SPLIT=dev
    make eval-diff RUN_A=experiment-001-baseline RUN_B=experiment-002-semantic
    make eval-gate RUN_A=experiment-001-baseline RUN_B=candidate
    make eval-validate SPLIT=dev

Every subcommand exits non-zero on failure, because the regression gate's whole
value is being able to fail a CI job.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.session import get_session_factory
from app.evaluation import storage
from app.evaluation.dataset import load_split, summarize, validate_against_corpus
from app.evaluation.diff import compare, format_diff, gate
from app.evaluation.human_review import (
    agreement_report,
    export_review_batch,
    import_review_batch,
)
from app.evaluation.repository import ExperimentRepository
from app.evaluation.results import ExperimentRun
from app.evaluation.runner import ExperimentRunner
from app.evaluation.schemas import DatasetSplit, GoldenQuestion

logger = get_logger("app.evaluation.cli")


#: Stands in for "whichever experiment this branch is proposing" so the CI gate
#: does not have to be edited every time a stage adds an experiment.
LATEST = "latest"


async def _resolve_run(
    name: str,
    split: DatasetSplit | None = None,
    exclude: str | None = None,
    comparable_to: ExperimentRun | None = None,
) -> ExperimentRun | None:
    """Find a run by name: committed file first, then the database.

    File first so the same command works in CI, which has committed results and
    no PostgreSQL, and on a developer machine, which has both.
    """
    if name == LATEST:
        return storage.latest_run_file(exclude=exclude, comparable_to=comparable_to)

    from_file = storage.load_run_file(name)
    if from_file is not None:
        return from_file

    try:
        session_factory = get_session_factory()
        async with session_factory() as session:
            return await ExperimentRepository(session).get_latest_by_name(name, split)
    except Exception as exc:  # noqa: BLE001 - a missing database is a normal CI condition
        logger.warning("run_lookup_db_unavailable", name=name, error=str(exc))
        return None


def sample_per_type(questions: list[GoldenQuestion], per_type: int) -> list[GoldenQuestion]:
    """Take the first N questions of each type, in question_id order.

    ``--limit`` alone is not a usable subset: question IDs sort by type, so the
    first 25 of the dev split are every adversarial and ambiguous question and
    nothing else. A subset that omits whole categories cannot be compared to a
    full run, and a fast CI subset that never exercises retrieval is worse than
    no subset. Deterministic rather than random for the same reason the metrics
    are: two runs of the same subset must evaluate the same questions.
    """
    taken: dict[str, int] = {}
    sampled: list[GoldenQuestion] = []
    for question in sorted(questions, key=lambda q: q.question_id):
        key = question.question_type.value
        if taken.get(key, 0) >= per_type:
            continue
        taken[key] = taken.get(key, 0) + 1
        sampled.append(question)
    return sampled


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


async def cmd_run(args: argparse.Namespace) -> int:
    """Execute an experiment against a dataset split."""
    settings = get_settings()
    split = DatasetSplit(args.split)

    if settings.INFERENCE_PROFILE == "stub":
        print(
            "REFUSING TO RUN: INFERENCE_PROFILE=stub produces canned text, so the "
            "resulting numbers would look like evaluation data without being it.\n"
            "Set INFERENCE_PROFILE=hosted (with provider keys) or =local.",
            file=sys.stderr,
        )
        return 2

    questions = load_split(
        split,
        version=args.dataset_version,
        unlock_token=args.unlock_test_split,
    )
    if args.per_type:
        questions = sample_per_type(questions, args.per_type)
    if args.limit:
        questions = questions[: args.limit]

    print(f"Running '{args.name}' over {len(questions)} {split.value} questions…")

    runner = ExperimentRunner(settings=settings)
    session_factory = get_session_factory()

    run = await runner.run(
        name=args.name,
        questions=questions,
        session_factory=session_factory,
        split=split,
        dataset_version=args.dataset_version,
        description=args.description,
        notes=args.notes,
        judge_enabled=None if args.judge else False,
    )

    path = storage.save_run_file(run)
    print(f"\n{run.summary_line()}")
    print(f"Wrote {path}")

    if not args.no_db:
        try:
            async with session_factory() as session:
                await ExperimentRepository(session).save(run)
                await session.commit()
            print(f"Persisted run {run.run_id} to PostgreSQL")
        except Exception as exc:  # noqa: BLE001 - the committed file is the durable record
            print(f"WARNING: could not persist to PostgreSQL: {exc}", file=sys.stderr)

    _print_metrics(run)
    return 0


def _print_metrics(run: ExperimentRun) -> None:
    print("\nOVERALL")
    for name, value in sorted(run.metrics.items()):
        print(f"  {name:<34}{value:>10.4f}")
    print("\nSYSTEM")
    for name, value in sorted(run.system_metrics.items()):
        print(f"  {name:<34}{value:>10.4f}")

    failures = [r for r in run.results if r.failed]
    if failures:
        print(f"\nFAILED QUESTIONS ({len(failures)})")
        for result in failures[:20]:
            print(f"  {result.question_id}: {result.error}")

    expected_failures = [r for r in run.results if r.expected_to_fail_until_stage is not None]
    if expected_failures:
        print(
            f"\n{len(expected_failures)} questions are documented as expected failures "
            f"until a later stage (see evaluation/datasets/README.md)."
        )


# --------------------------------------------------------------------------
# diff / gate
# --------------------------------------------------------------------------


async def cmd_diff(args: argparse.Namespace) -> int:
    """Compare two runs per metric and per question type."""
    baseline = await _resolve_run(args.baseline)
    candidate = await _resolve_run(args.candidate, exclude=args.baseline)

    for name, run in ((args.baseline, baseline), (args.candidate, candidate)):
        if run is None:
            print(f"Experiment not found: {name}", file=sys.stderr)
            return 2

    assert baseline is not None and candidate is not None
    diff = compare(baseline, candidate, tolerance=args.tolerance)
    print(format_diff(diff, tolerance=args.tolerance, show_types=not args.no_types))
    return 0


async def cmd_gate(args: argparse.Namespace) -> int:
    """Fail when a gated metric regressed beyond the tolerance."""
    settings = get_settings()
    tolerance = args.tolerance if args.tolerance is not None else settings.EVAL_REGRESSION_TOLERANCE

    baseline = await _resolve_run(args.baseline)
    if baseline is None:
        print(f"Baseline experiment not found: {args.baseline}", file=sys.stderr)
        return 2

    candidate = await _resolve_run(
        args.candidate, exclude=args.baseline, comparable_to=baseline
    )
    if candidate is None:
        if args.candidate == LATEST:
            # Nothing to gate is not a pass and not a failure: most commits do
            # not propose a new experiment, and failing them would train people
            # to bypass the gate.
            print(
                f"No committed experiment comparable to '{args.baseline}' "
                f"({baseline.dataset_split.value}/{baseline.dataset_version}, "
                f"{baseline.dataset_size} questions) — nothing to gate. Run "
                f"`make eval CONFIG=<name>` over the same split and commit the result."
            )
            return 0
        print(f"Candidate experiment not found: {args.candidate}", file=sys.stderr)
        return 2

    print(f"Gating '{candidate.name}' against '{baseline.name}' (tolerance {tolerance}).")

    result = gate(baseline, candidate, tolerance=tolerance)
    print(result.report())
    return 0 if result.passed else 1


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


async def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a split's schema, then resolve its evidence against the corpus."""
    split = DatasetSplit(args.split)
    questions = load_split(split, version=args.dataset_version, unlock_token=args.unlock_test_split)

    stats = summarize(questions, split=split, version=args.dataset_version)
    print(f"{split.value}/{args.dataset_version}: {stats.total} questions")
    print("  by type:")
    for name, count in stats.by_type.items():
        print(f"    {name:<24}{count:>5}")
    print(f"  abstention cases: {stats.abstention_cases}")
    print(f"  documents covered: {stats.documents_covered}")

    if stats.missing_types:
        print(f"  MISSING TYPES: {', '.join(stats.missing_types)}")

    if args.no_corpus:
        return 1 if stats.missing_types else 0

    session_factory = get_session_factory()
    async with session_factory() as session:
        report = await validate_against_corpus(
            questions, session, split=split, version=args.dataset_version
        )

    if report.issues:
        print(f"\n{len(report.issues)} UNRESOLVABLE EVIDENCE POINTERS")
        for issue in report.issues[:50]:
            print(f"  {issue.question_id}: {issue.reason} {issue.detail}")
    else:
        print("\nEvery expected_evidence pointer resolves to a real element.")

    return 0 if report.is_valid else 1


# --------------------------------------------------------------------------
# human review
# --------------------------------------------------------------------------


async def cmd_export_review(args: argparse.Namespace) -> int:
    """Export the cases most worth a human verdict."""
    run = await _resolve_run(args.run)
    if run is None:
        print(f"Experiment not found: {args.run}", file=sys.stderr)
        return 2

    path = export_review_batch(run, Path(args.out), limit=args.limit)
    print(f"Wrote {path} — fill in the reviewer/verdict fields, then run eval-import-review.")
    return 0


async def cmd_import_review(args: argparse.Namespace) -> int:
    """Import completed verdicts and report agreement with the automatic layers."""
    verdicts = import_review_batch(Path(args.file))
    if not verdicts:
        print("No completed verdicts found (rows need a `reviewer` value).")
        return 0

    session_factory = get_session_factory()
    async with session_factory() as session:
        saved = await ExperimentRepository(session).save_human_verdicts(verdicts)
        await session.commit()
    print(f"Stored {saved} human verdicts.")

    run = await _resolve_run(args.run) if args.run else None
    if run is not None:
        for name, value in sorted(agreement_report(run, verdicts).items()):
            print(f"  {name:<34}{value:>10.4f}")
    return 0


async def cmd_list(args: argparse.Namespace) -> int:
    """List committed experiment records."""
    names = storage.list_run_files()
    if not names:
        print("No committed experiments in evaluation/results/.")
        return 0
    for name in names:
        run = storage.load_run_file(name)
        if run is not None:
            print(run.summary_line())
    return 0


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eval", description="RAG evaluation harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an experiment against a split")
    run_parser.add_argument(
        "--name", required=True, help="experiment name, e.g. experiment-001-baseline"
    )
    run_parser.add_argument("--split", default="dev", choices=[s.value for s in DatasetSplit])
    run_parser.add_argument("--dataset-version", default="v1")
    run_parser.add_argument("--description", default="")
    run_parser.add_argument("--notes", default="")
    run_parser.add_argument(
        "--limit", type=int, default=0, help="evaluate only the first N questions"
    )
    run_parser.add_argument(
        "--per-type",
        type=int,
        default=0,
        help="evaluate only the first N questions of each question type (a comparable subset)",
    )
    run_parser.add_argument("--no-judge", dest="judge", action="store_false", help="skip Layer 2")
    run_parser.add_argument(
        "--no-db", action="store_true", help="write the file but skip PostgreSQL"
    )
    run_parser.add_argument("--unlock-test-split", default=None, help=argparse.SUPPRESS)
    run_parser.set_defaults(func=cmd_run, judge=True)

    diff_parser = subparsers.add_parser("diff", help="compare two experiments")
    diff_parser.add_argument("--baseline", required=True)
    diff_parser.add_argument("--candidate", required=True)
    diff_parser.add_argument("--tolerance", type=float, default=0.0)
    diff_parser.add_argument("--no-types", action="store_true", help="hide the per-type breakdown")
    diff_parser.set_defaults(func=cmd_diff)

    gate_parser = subparsers.add_parser("gate", help="fail on regression beyond tolerance")
    gate_parser.add_argument("--baseline", required=True)
    gate_parser.add_argument(
        "--candidate",
        required=True,
        help=f"experiment name, or '{LATEST}' for the newest committed run that is not the baseline",
    )
    gate_parser.add_argument("--tolerance", type=float, default=None)
    gate_parser.set_defaults(func=cmd_gate)

    validate_parser = subparsers.add_parser("validate", help="validate a golden dataset split")
    validate_parser.add_argument("--split", default="dev", choices=[s.value for s in DatasetSplit])
    validate_parser.add_argument("--dataset-version", default="v1")
    validate_parser.add_argument("--no-corpus", action="store_true", help="schema checks only")
    validate_parser.add_argument("--unlock-test-split", default=None, help=argparse.SUPPRESS)
    validate_parser.set_defaults(func=cmd_validate)

    export_parser = subparsers.add_parser("export-review", help="export a Layer 3 review batch")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument("--out", required=True)
    export_parser.add_argument("--limit", type=int, default=25)
    export_parser.set_defaults(func=cmd_export_review)

    import_parser = subparsers.add_parser("import-review", help="import completed verdicts")
    import_parser.add_argument("--file", required=True)
    import_parser.add_argument("--run", default=None, help="report agreement against this run")
    import_parser.set_defaults(func=cmd_import_review)

    list_parser = subparsers.add_parser("list", help="list committed experiments")
    list_parser.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code: int = asyncio.run(args.func(args))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
