"""Paired bootstrap confidence intervals between committed experiment runs.

    python -m scripts.paired_ci experiment-002-fixed-512-64 experiment-002-contextual-256-32 \\
        --metric recall@5 --metric ndcg@10

Run names may be abbreviated to any unique prefix after ``experiment-``
("002-fixed", "019"). Method and fixed parameters: ``app.evaluation.stats``
(10,000 resamples, seed 2026, 95% percentile interval, paired by question over
the shared questions, with an exact sign test as a cross-check).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from app.evaluation.stats import compare_runs
from app.evaluation.storage import results_dir
from scripts.audit_results import load_runs


def resolve(runs: Mapping[str, Any], ref: str) -> str:
    if ref in runs:
        return ref
    matches = [name for name in runs if name.removeprefix("experiment-").startswith(ref)]
    if "verify" not in ref:
        matches = [name for name in matches if "verify" not in name]
    exact = [name for name in matches if name.removeprefix("experiment-") == ref]
    if len(exact) == 1:
        return exact[0]
    if len(matches) != 1:
        raise SystemExit(f"run reference {ref!r} matches {matches or 'nothing'}")
    return matches[0]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--metric", action="append", required=True)
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)

    runs = load_runs(results_dir())
    a, b = resolve(runs, args.baseline), resolve(runs, args.candidate)
    results = [compare_runs(runs[a], runs[b], metric) for metric in args.metric]
    if args.json:
        print(json.dumps({"baseline": a, "candidate": b, "results": [asdict(r) for r in results]}))
    else:
        print(f"{a} -> {b}")
        for result in results:
            print("  " + result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
