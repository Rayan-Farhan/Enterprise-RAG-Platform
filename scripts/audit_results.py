"""Static audit of the committed experiment records (results verification, Phase 1).

Every number the project publishes traces back to a file in
``evaluation/results/``. This script checks those files against themselves:

* **Aggregates are recomputed** from the per-question ``results`` with the
  runner's own rules (macro-average per metric over non-failed questions;
  questions without relevance judgements contribute no retrieval metrics, so a
  metric absent from a question is neutral rather than zero). Any stored
  aggregate that differs by more than ``--tolerance`` is a mismatch, and so is
  an aggregate present on one side only.
* **Each run is profiled**: size, split, errors, failure rate, which generator
  answered each question, prompt versions and hashes, chunking version, commit,
  and whether it was retrieval-only.
* **Comparisons are checked for comparability**: two runs quoted side by side
  should share a question set, prompts and generator. ``--pair A B`` adds a
  comparison to the built-in list of the headline ones.

    python -m scripts.audit_results
    python -m scripts.audit_results --run experiment-019-sparse-groq-v1
    python -m scripts.audit_results --pair experiment-005-contextual-256-32-groq \\
        experiment-019-sparse-groq-v1 --json audit.json

Reads raw JSON rather than ``ExperimentRun`` on purpose: an audit has to see
what the file actually says, including fields older records leave null, and
must not fill gaps with today's model defaults.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.evaluation.metrics.system import percentile
from app.evaluation.storage import results_dir

#: The comparisons the stage docs make. Each pair is (baseline, candidate).
#: Missing files are reported, not fatal, so the list can run ahead of the runs.
HEADLINE_PAIRS: tuple[tuple[str, str], ...] = (
    ("experiment-001-baseline", "experiment-002-fixed-512-64"),
    ("experiment-001-baseline", "experiment-005-contextual-256-32-groq"),
    ("experiment-005-contextual-256-32-groq", "experiment-019-sparse-groq-v1"),
    ("experiment-001-baseline", "experiment-019-sparse-groq-v1"),
    ("experiment-004-hc-256-32-noexpand", "experiment-004-hc-256-32-expand"),
    ("experiment-020-hc-sparse-noexpand", "experiment-020-hc-sparse-expand"),
    ("experiment-018-val-dense", "experiment-018-val-sparse"),
    ("experiment-018-val-dense", "experiment-018-val-bm25"),
    ("experiment-018-val-dense", "experiment-018-val-rrf-all"),
    ("experiment-018-val-sparse", "experiment-018-val-sparse-reranked"),
    ("experiment-018-val-rrf-all", "experiment-018-val-rrf-all-reranked"),
)

#: Snapshot keys that legitimately differ between compared runs (they are the
#: variable under test), so they are reported but not flagged.
_GENERATION_SNAPSHOT_KEYS = (
    "generation_provider",
    "generation_temperature",
    "generation_max_tokens",
    "generation_max_context_tokens",
    "abstention_min_evidence_chunks",
)


# --------------------------------------------------------------------------
# Recomputation
# --------------------------------------------------------------------------


def _question_metrics(result: Mapping[str, Any]) -> dict[str, float]:
    """``QuestionResult.all_metrics()`` over raw JSON, dropping null values."""
    merged: dict[str, Any] = {}
    for layer in ("retrieval_metrics", "deterministic_metrics", "judge_metrics"):
        merged.update(result.get(layer) or {})
    return {name: float(value) for name, value in merged.items() if value is not None}


def _macro_average(per_case: Iterable[Mapping[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for metrics in per_case:
        for name, value in metrics.items():
            totals[name] += value
            counts[name] += 1
    return {name: totals[name] / counts[name] for name in sorted(totals)}


def recompute_metrics(results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    return _macro_average(_question_metrics(r) for r in results if r.get("error") is None)


def recompute_metrics_by_type(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for result in results:
        if result.get("error") is None:
            grouped[str(result["question_type"])].append(_question_metrics(result))
    return {qtype: _macro_average(metrics) for qtype, metrics in sorted(grouped.items())}


def recompute_system_metrics(
    results: Sequence[Mapping[str, Any]], dataset_size: int
) -> dict[str, float]:
    """The latency/failure subset of ``SystemMetrics.as_metrics`` that the rows can rebuild."""
    ok = [r for r in results if r.get("error") is None]
    failures = len(results) - len(ok)
    return {
        "latency_p50_ms": percentile([r.get("latency_ms", 0.0) for r in ok], 0.50),
        "latency_p95_ms": percentile([r.get("latency_ms", 0.0) for r in ok], 0.95),
        "latency_p99_ms": percentile([r.get("latency_ms", 0.0) for r in ok], 0.99),
        "retrieval_latency_p95_ms": percentile(
            [r.get("retrieval_latency_ms", 0.0) for r in ok], 0.95
        ),
        "failure_rate": failures / dataset_size if dataset_size else 0.0,
    }


def _mismatches(
    stored: Mapping[str, Any], recomputed: Mapping[str, float], tolerance: float, scope: str
) -> list[str]:
    found: list[str] = []
    for name in sorted(set(stored) | set(recomputed)):
        before, after = stored.get(name), recomputed.get(name)
        if before is None and after is None:
            continue
        if before is None:
            found.append(f"{scope}{name}: not stored, recomputes to {after:.6f}")
        elif after is None:
            found.append(f"{scope}{name}: stored {float(before):.6f}, no question reports it")
        elif not math.isclose(float(before), after, abs_tol=tolerance):
            found.append(
                f"{scope}{name}: stored {float(before):.6f}, recomputed {after:.6f} "
                f"(delta {after - float(before):+.6f})"
            )
    return found


# --------------------------------------------------------------------------
# Per-run profile
# --------------------------------------------------------------------------


@dataclass
class RunAudit:
    name: str
    split: str | None
    dataset_version: str | None
    dataset_size: int
    recorded_questions: int
    errors: int
    error_ids: list[str]
    failure_rate: float | None
    retrieval_only: bool
    git_commit: str | None
    chunking_version: str | None
    embedding_version: str | None
    retrieval_mode: str | None
    run_generator: str | None
    generator_mix: dict[str, int]
    prompt_versions: dict[str, str]
    prompt_hashes: dict[str, str]
    judge: str | None
    evaluation_days: list[str]
    started_at: str | None
    completed_at: str | None
    notes: str
    metric_mismatches: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.metric_mismatches and not self.flags


def audit_run(name: str, data: Mapping[str, Any], tolerance: float) -> RunAudit:
    results: list[Mapping[str, Any]] = list(data.get("results") or [])
    snapshot: Mapping[str, Any] = data.get("config_snapshot") or {}
    errored = [r for r in results if r.get("error") is not None]
    retrieval_only = bool(data.get("retrieval_only", False))

    mix: Counter[str] = Counter()
    if not retrieval_only:
        for r in results:
            if r.get("error") is not None:
                continue
            mix[f"{r.get('generator_provider')}/{r.get('generator_model')}"] += 1

    days = sorted({str(r["evaluated_at"])[:10] for r in results if r.get("evaluated_at")})
    run_generator = (
        f"{data.get('generator_provider')}/{data.get('generator_model')}"
        if data.get("generator_model")
        else None
    )
    judge = (
        f"{data.get('judge_provider')}/{data.get('judge_model')}"
        if data.get("judge_model")
        else None
    )

    audit = RunAudit(
        name=name,
        split=data.get("dataset_split"),
        dataset_version=data.get("dataset_version"),
        dataset_size=int(data.get("dataset_size") or 0),
        recorded_questions=len(results),
        errors=len(errored),
        error_ids=sorted(str(r["question_id"]) for r in errored),
        failure_rate=(data.get("system_metrics") or {}).get("failure_rate"),
        retrieval_only=retrieval_only,
        git_commit=data.get("git_commit"),
        chunking_version=data.get("chunking_version"),
        embedding_version=data.get("embedding_version"),
        retrieval_mode=snapshot.get("retrieval_mode"),
        run_generator=run_generator,
        generator_mix=dict(sorted(mix.items())),
        prompt_versions=dict(data.get("prompt_versions") or {}),
        prompt_hashes=dict(data.get("prompt_hashes") or {}),
        judge=judge,
        evaluation_days=days,
        started_at=data.get("started_at"),
        completed_at=data.get("completed_at"),
        notes=str(data.get("notes") or ""),
    )

    audit.metric_mismatches += _mismatches(
        data.get("metrics") or {}, recompute_metrics(results), tolerance, ""
    )
    stored_by_type: Mapping[str, Mapping[str, Any]] = data.get("metrics_by_type") or {}
    recomputed_by_type = recompute_metrics_by_type(results)
    for qtype in sorted(set(stored_by_type) | set(recomputed_by_type)):
        audit.metric_mismatches += _mismatches(
            stored_by_type.get(qtype, {}),
            recomputed_by_type.get(qtype, {}),
            tolerance,
            f"[{qtype}] ",
        )
    stored_system = data.get("system_metrics") or {}
    recomputed_system = recompute_system_metrics(results, audit.dataset_size)
    audit.metric_mismatches += _mismatches(
        {k: stored_system[k] for k in recomputed_system if k in stored_system},
        recomputed_system,
        tolerance,
        "[system] ",
    )

    if audit.recorded_questions != audit.dataset_size:
        audit.flags.append(
            f"dataset_size {audit.dataset_size} but {audit.recorded_questions} rows recorded"
        )
    if audit.errors:
        audit.flags.append(
            f"{audit.errors} errored question(s); metrics cover "
            f"{audit.recorded_questions - audit.errors} of {audit.dataset_size}"
        )
    if not retrieval_only:
        unattributed = sum(n for key, n in mix.items() if key.startswith("None/"))
        if unattributed:
            audit.flags.append(
                f"{unattributed} answered question(s) have no per-question generator "
                f"provider recorded; the generator mix is unprovable from this file"
            )
        if len(mix) > 1:
            audit.flags.append(f"mixes generators: {dict(mix)}")
        if run_generator and mix and run_generator not in mix:
            audit.flags.append(
                f"run-level generator {run_generator} answered none of the questions"
            )
        if not audit.prompt_hashes:
            audit.flags.append("no prompt hashes recorded")
    if len(days) > 1:
        audit.flags.append(f"evaluated across {len(days)} days: {', '.join(days)}")
    if not audit.git_commit:
        audit.flags.append("no git commit recorded")
    if audit.chunking_version != snapshot.get("chunking_version"):
        audit.flags.append(
            f"chunking_version {audit.chunking_version} differs from snapshot "
            f"{snapshot.get('chunking_version')}"
        )
    return audit


# --------------------------------------------------------------------------
# Pairwise comparability
# --------------------------------------------------------------------------


@dataclass
class PairAudit:
    baseline: str
    candidate: str
    missing: list[str] = field(default_factory=list)
    shared_questions: int = 0
    only_baseline: list[str] = field(default_factory=list)
    only_candidate: list[str] = field(default_factory=list)
    errored_either: list[str] = field(default_factory=list)
    snapshot_differences: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def audit_pair(
    baseline: str,
    candidate: str,
    runs: Mapping[str, Mapping[str, Any]],
    audits: Mapping[str, RunAudit],
) -> PairAudit:
    pair = PairAudit(baseline=baseline, candidate=candidate)
    pair.missing = [name for name in (baseline, candidate) if name not in runs]
    if pair.missing:
        return pair

    a, b = runs[baseline], runs[candidate]
    rows_a = {r["question_id"]: r for r in a.get("results") or []}
    rows_b = {r["question_id"]: r for r in b.get("results") or []}
    shared = set(rows_a) & set(rows_b)
    pair.shared_questions = len(shared)
    pair.only_baseline = sorted(set(rows_a) - set(rows_b))
    pair.only_candidate = sorted(set(rows_b) - set(rows_a))
    pair.errored_either = sorted(
        qid
        for qid in shared
        if rows_a[qid].get("error") is not None or rows_b[qid].get("error") is not None
    )

    snap_a: Mapping[str, Any] = a.get("config_snapshot") or {}
    snap_b: Mapping[str, Any] = b.get("config_snapshot") or {}
    pair.snapshot_differences = {
        key: (snap_a.get(key), snap_b.get(key))
        for key in sorted(set(snap_a) | set(snap_b))
        if snap_a.get(key) != snap_b.get(key)
    }

    ra, rb = audits[baseline], audits[candidate]
    if ra.split != rb.split:
        pair.flags.append(f"different splits: {ra.split} vs {rb.split}")
    if pair.only_baseline or pair.only_candidate:
        pair.flags.append(
            f"question sets differ: {len(pair.only_baseline)} only in baseline, "
            f"{len(pair.only_candidate)} only in candidate"
        )
    if pair.errored_either:
        pair.flags.append(
            f"{len(pair.errored_either)} shared question(s) errored in one run; "
            f"aggregates are over different subsets"
        )
    if not ra.retrieval_only and not rb.retrieval_only:
        if ra.generator_mix.keys() != rb.generator_mix.keys():
            pair.flags.append(
                f"generators differ: {ra.generator_mix or ra.run_generator} vs "
                f"{rb.generator_mix or rb.run_generator}"
            )
        if ra.prompt_hashes != rb.prompt_hashes:
            changed = sorted(
                key
                for key in set(ra.prompt_hashes) | set(rb.prompt_hashes)
                if ra.prompt_hashes.get(key) != rb.prompt_hashes.get(key)
            )
            pair.flags.append(f"prompt content differs for: {', '.join(changed)}")
        generation_diffs = [k for k in _GENERATION_SNAPSHOT_KEYS if k in pair.snapshot_differences]
        if generation_diffs:
            pair.flags.append(f"generation settings differ: {', '.join(generation_diffs)}")
    if ra.retrieval_only != rb.retrieval_only:
        pair.flags.append("one run is retrieval-only; Layer 1/2 metrics are not comparable")
    if ra.judge != rb.judge and (ra.judge or rb.judge):
        pair.flags.append(f"judged by different models: {ra.judge} vs {rb.judge}")
    return pair


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def load_runs(directory: Path, names: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
    paths = sorted(directory.glob("*.json"))
    runs: dict[str, dict[str, Any]] = {}
    for path in paths:
        if names and path.stem not in names:
            continue
        runs[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return runs


def format_report(audits: Sequence[RunAudit], pairs: Sequence[PairAudit], tolerance: float) -> str:
    lines: list[str] = [
        f"# Results audit ({len(audits)} files, tolerance {tolerance:g})",
        "",
        "| run | split | n | rows | err | ret-only | chunking | mode | generator mix | "
        "prompts | days | mismatches | flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in audits:
        mix = ", ".join(f"{k}×{v}" for k, v in a.generator_mix.items()) or "-"
        prompts = ", ".join(f"{k}={v}" for k, v in sorted(a.prompt_versions.items())) or "-"
        lines.append(
            f"| {a.name} | {a.split} | {a.dataset_size} | {a.recorded_questions} | {a.errors} | "
            f"{'yes' if a.retrieval_only else 'no'} | {a.chunking_version} | {a.retrieval_mode} | "
            f"{mix} | {prompts} | {len(a.evaluation_days)} | {len(a.metric_mismatches)} | "
            f"{len(a.flags)} |"
        )

    lines += ["", "## Per-run findings", ""]
    for a in audits:
        if a.ok:
            continue
        lines.append(f"### {a.name}")
        lines.append(
            f"commit `{a.git_commit}` · run generator `{a.run_generator}` · judge `{a.judge}`"
        )
        if a.prompt_hashes:
            hashes = ", ".join(f"{k}={v[:12]}" for k, v in sorted(a.prompt_hashes.items()))
            lines.append(f"prompt hashes: {hashes}")
        if a.error_ids:
            lines.append(f"errored: {', '.join(a.error_ids)}")
        for flag in a.flags:
            lines.append(f"- FLAG: {flag}")
        for mismatch in a.metric_mismatches:
            lines.append(f"- MISMATCH: {mismatch}")
        if a.notes:
            lines.append(f"- notes: {a.notes}")
        lines.append("")

    lines += ["## Comparisons", ""]
    for p in pairs:
        lines.append(f"### {p.baseline} → {p.candidate}")
        if p.missing:
            lines.append(f"- MISSING: {', '.join(p.missing)}")
            lines.append("")
            continue
        lines.append(f"- shared questions: {p.shared_questions}")
        for flag in p.flags:
            lines.append(f"- FLAG: {flag}")
        if p.snapshot_differences:
            diffs = "; ".join(f"{k}: {a!r} → {b!r}" for k, (a, b) in p.snapshot_differences.items())
            lines.append(f"- config differences: {diffs}")
        lines.append("")

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=results_dir(), help="results directory")
    parser.add_argument("--run", action="append", default=[], help="audit only these runs")
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        default=[],
        metavar=("BASELINE", "CANDIDATE"),
        help="add a comparison to the built-in headline pairs",
    )
    parser.add_argument("--no-headline-pairs", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--json", type=Path, help="also write the full audit as JSON")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 when any metric mismatch is found"
    )
    args = parser.parse_args(argv)

    all_runs = load_runs(args.dir)
    selected = {k: v for k, v in all_runs.items() if not args.run or k in args.run}
    audits = {name: audit_run(name, data, args.tolerance) for name, data in all_runs.items()}

    wanted_pairs = [] if args.no_headline_pairs else list(HEADLINE_PAIRS)
    wanted_pairs += [tuple(p) for p in args.pair]
    if args.run:
        wanted_pairs = [p for p in wanted_pairs if p[0] in args.run or p[1] in args.run]
    pairs = [audit_pair(a, b, all_runs, audits) for a, b in wanted_pairs]

    shown = [audits[name] for name in selected]
    print(format_report(shown, pairs, args.tolerance))

    if args.json:
        args.json.write_text(
            json.dumps(
                {"runs": [asdict(a) for a in shown], "pairs": [asdict(p) for p in pairs]},
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.strict and any(a.metric_mismatches for a in shown):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
