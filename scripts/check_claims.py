"""Check published numbers against the committed experiment records (results verification).

A claims file lists every number a document states, where it is stated, and an
expression that recomputes it from ``evaluation/results/``. This script evaluates
each expression and compares it with the stated value at the stated precision: a
claim written as ``0.811`` is confirmed when the recomputed value rounds to it.

    python -m scripts.check_claims docs/verified_claims.json
    python -m scripts.check_claims docs/verified_claims.json --markdown ledger.md

Claims file: a JSON list of objects with

* ``id``, ``group``, ``source`` (``file:line``) and ``claim`` (as written);
* ``expr`` and ``stated`` for a claim the records can settle. ``stated`` is a
  string so its precision is kept (``"0.30"`` means two decimals);
* or ``status`` (``unverifiable`` / ``pending``) plus a ``note`` for one they
  cannot, e.g. a latency measured outside any run, or a figure that needs a
  live re-run;
* optional ``note``, and ``corrected`` (a statement of the correct claim) when
  the number matches but its wording or attribution does not.

Expressions run against a small vocabulary over the result files; see
``ClaimContext``. Run names may be abbreviated to any unique prefix after
``experiment-`` (``"019"``, ``"018-val-sparse-"``).

The claims file sits beside the documents it audits; this script is the
reproducible part, so a ledger row names the expression that produced it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evaluation.metrics.system import percentile
from app.evaluation.storage import results_dir
from scripts.audit_results import load_runs


class ClaimContext:
    """The functions a claim expression can call."""

    def __init__(self, runs: Mapping[str, Mapping[str, Any]]) -> None:
        self.runs = runs

    def run(self, ref: str) -> Mapping[str, Any]:
        if ref in self.runs:
            return self.runs[ref]
        matches = [name for name in self.runs if name.removeprefix("experiment-").startswith(ref)]
        if len(matches) != 1:
            raise KeyError(f"run reference {ref!r} matches {matches or 'nothing'}")
        return self.runs[matches[0]]

    def _rows(self, ref: str, qtype: str | None = None) -> list[Mapping[str, Any]]:
        return [
            r
            for r in self.run(ref)["results"]
            if r.get("error") is None and (qtype is None or r["question_type"] == qtype)
        ]

    # -- aggregates as stored (scripts.audit_results proves they recompute) --

    def m(self, ref: str, metric: str, qtype: str | None = None) -> float:
        """A stored aggregate metric, overall or for one question type."""
        data = self.run(ref)
        source = data["metrics"] if qtype is None else data["metrics_by_type"][qtype]
        return float(source[metric])

    def sm(self, ref: str, key: str) -> float:
        """A stored system metric (latencies in ms, token averages, failure_rate)."""
        return float(self.run(ref)["system_metrics"][key])

    # -- quantities only the per-question rows carry ------------------------

    def answerable(self, ref: str) -> int:
        """Questions with relevance judgements (the retrieval-metric denominator)."""
        return sum(1 for r in self._rows(ref) if r.get("retrieval_metrics"))

    def answered(self, ref: str) -> int:
        """Answerable questions the system answered rather than abstained on."""
        return sum(1 for r in self._rows(ref) if r.get("retrieval_metrics") and not r["abstained"])

    def count(self, ref: str, qtype: str | None = None) -> int:
        return len(self._rows(ref, qtype))

    def correct(self, ref: str, qtype: str, metric: str = "abstention_correct") -> int:
        """Questions of a type scoring 1.0 on a deterministic metric."""
        return sum(
            1
            for r in self._rows(ref, qtype)
            if (r.get("deterministic_metrics") or {}).get(metric) == 1.0
        )

    def reported(self, ref: str, metric: str) -> int:
        """How many questions report a metric — the n behind its average."""
        return sum(
            1
            for r in self._rows(ref)
            if metric
            in {
                **(r.get("retrieval_metrics") or {}),
                **(r.get("deterministic_metrics") or {}),
                **(r.get("judge_metrics") or {}),
            }
        )

    def answered_mean(self, ref: str, metric: str) -> float:
        """A judge metric averaged over answered (non-abstained) questions only."""
        values = [
            float(r["judge_metrics"][metric])
            for r in self._rows(ref)
            if not r["abstained"] and metric in (r.get("judge_metrics") or {})
        ]
        return sum(values) / len(values)

    def ret_pct(self, ref: str, fraction: float) -> float:
        """Nearest-rank retrieval latency percentile, in seconds."""
        values = [float(r.get("retrieval_latency_ms") or 0.0) for r in self._rows(ref)]
        return percentile(values, fraction) / 1000

    def namespace(self) -> dict[str, Callable[..., Any]]:
        names = ("m sm answerable answered count correct reported answered_mean ret_pct").split()
        return {name: getattr(self, name) for name in names}


@dataclass
class ClaimResult:
    id: str
    group: str
    source: str
    claim: str
    stated: str | None
    verified: str | None
    status: str
    evidence: str
    note: str


def _decimals(stated: str) -> int:
    text = stated.strip().lstrip("+-−")
    return len(text.split(".")[1]) if "." in text else 0


def _format(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def check(claim: Mapping[str, Any], context: ClaimContext) -> ClaimResult:
    stated = claim.get("stated")
    note = str(claim.get("note", ""))
    base = {
        "id": str(claim["id"]),
        "group": str(claim.get("group", "")),
        "source": str(claim.get("source", "")),
        "claim": str(claim["claim"]),
        "stated": None if stated is None else str(stated),
    }

    expr = claim.get("expr")
    if not expr:
        return ClaimResult(
            **base,
            verified=claim.get("verified"),
            status=str(claim.get("status", "unverifiable")),
            evidence=str(claim.get("evidence", "")),
            note=note,
        )

    try:
        value = float(eval(expr, {"__builtins__": {}}, context.namespace()))  # noqa: S307
    except Exception as exc:  # noqa: BLE001 - a broken claim is a reported row, not a crash
        return ClaimResult(
            **base, verified=None, status="error", evidence=f"`{expr}`", note=f"{exc}"
        )

    decimals = _decimals(str(stated)) if stated is not None else 4
    verified = _format(value, decimals)
    if stated is None:
        status = "info"
    else:
        target = float(str(stated).replace("−", "-"))
        tolerance = 0.5 * 10**-decimals + 1e-9
        status = "confirmed" if math.isclose(value, target, abs_tol=tolerance) else "corrected"
    if status == "confirmed" and claim.get("corrected"):
        status = "corrected"
        note = f"{claim['corrected']} {note}".strip()
    evidence = f"`{expr}`"

    # A newer measurement of the same quantity (e.g. a "-verify" re-run) can
    # supersede the record the claim was copied from: the claim may match its
    # source and still be wrong about the system.
    supersede = claim.get("supersede_expr")
    if supersede and stated is not None:
        try:
            newer = float(eval(supersede, {"__builtins__": {}}, context.namespace()))  # noqa: S307
        except Exception as exc:  # noqa: BLE001
            return ClaimResult(
                **base, verified=verified, status="error", evidence=f"`{supersede}`", note=str(exc)
            )
        target = float(str(stated).replace("−", "-"))
        if not math.isclose(newer, target, abs_tol=0.5 * 10**-decimals + 1e-9):
            status = "corrected"
            note = (
                f"Matches its record ({verified}) but re-measured as {_format(newer, decimals)}. "
                f"{note}"
            ).strip()
            verified = _format(newer, decimals)
        evidence = f"`{expr}`; re-measured: `{supersede}`"
    return ClaimResult(**base, verified=verified, status=status, evidence=evidence, note=note)


def format_markdown(results: Sequence[ClaimResult]) -> str:
    lines: list[str] = []
    group = None
    for r in results:
        if r.group != group:
            group = r.group
            lines += [
                "",
                f"### {group}",
                "",
                "| # | claim (as written) | source | verified value | status | evidence | note |",
                "|---|---|---|---|---|---|---|",
            ]
        cells = [r.id, r.claim, f"`{r.source}`", r.verified or "—", r.status, r.evidence, r.note]
        lines.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    return "\n".join(lines).lstrip("\n") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("claims", type=Path)
    parser.add_argument("--dir", type=Path, default=results_dir())
    parser.add_argument("--markdown", type=Path, help="write the ledger rows as markdown")
    args = parser.parse_args(argv)

    claims = json.loads(args.claims.read_text(encoding="utf-8"))
    context = ClaimContext(load_runs(args.dir))
    results = [check(claim, context) for claim in claims]

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
        if r.status not in {"confirmed", "info"}:
            print(f"{r.id:<8} {r.status:<12} stated {r.stated} -> {r.verified}  {r.claim}")
    print("totals:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))

    if args.markdown:
        args.markdown.write_text(format_markdown(results), encoding="utf-8")
    return 1 if counts.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
