"""Apply the context-leak guard to a committed run's stored answers, offline.

The guard (``CitationValidator._leaks_context``, commit 52667c5) landed between
experiment-005 and experiment-019, so 019 was scored with it and 005 without.
Rather than regenerate 005, this replays the guard over the answers 005 already
stored and reports what it would have withheld, and how abstention_correct moves
when each flagged answer is scored as a withheld (abstained) answer. No model is
called.

    python -m scripts.reapply_leak_guard experiment-005-contextual-256-32-groq \\
        experiment-019-sparse-groq-v1

The system prompt is rebuilt from the run's recorded prompt versions, and the
script refuses to continue if their content hashes no longer match the
templates on disk, because then it would be checking a different prompt.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from types import SimpleNamespace

from app.evaluation.storage import results_dir
from app.generation.citation import CitationValidator
from app.generation.prompts.registry import get_prompt
from scripts.audit_results import load_runs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args(argv)

    runs = load_runs(results_dir())
    for name in args.runs:
        data = runs[name]
        versions = data["prompt_versions"]
        answer, citation = get_prompt(versions["answer"]), get_prompt(versions["citation"])
        on_disk = {"answer": answer.content_hash, "citation": citation.content_hash}
        recorded = {key: data["prompt_hashes"].get(key) for key in on_disk}
        if recorded != on_disk:
            print(f"{name}: prompt templates changed since the run ({recorded} vs {on_disk})")
            return 1
        system = f"{answer.text}\n\n## How evidence is formatted\n\n{citation.text}"

        flagged: list[str] = []
        scores: list[float] = []
        by_type: dict[str, list[float]] = {}
        for row in data["results"]:
            if row.get("error") is not None:
                continue
            context = SimpleNamespace(
                system_prompt=system, has_evidence=bool(row["context_element_ids"])
            )
            texts = [t for t in (row["raw_answer"], row["answer"]) if t]
            score = row["deterministic_metrics"].get("abstention_correct")
            if any(CitationValidator._leaks_context(t, context) for t in texts):  # type: ignore[arg-type]
                flagged.append(f"{row['question_id']} (was {score})")
                # A withheld answer is an abstention; questions that expect one
                # (adversarial, negative) then score correct.
                score = 1.0 if score is not None else None
            if score is not None:
                scores.append(score)
                by_type.setdefault(row["question_type"], []).append(score)

        stored = data["metrics"]["abstention_correct"]
        print(f"{name}")
        print(f"  flagged: {', '.join(flagged) or 'none'}")
        print(
            f"  abstention_correct: stored {stored:.3f}, with guard {sum(scores) / len(scores):.3f}"
        )
        for qtype in ("adversarial", "negative_unsupported"):
            values = by_type.get(qtype, [])
            print(f"  {qtype}: {int(sum(values))} / {len(values)} with guard")
    return 0


if __name__ == "__main__":
    sys.exit(main())
