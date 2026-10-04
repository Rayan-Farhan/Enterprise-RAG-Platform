"""Re-run the retrieval-only experiments from their recorded configuration.

Each committed retrieval-only experiment is run again as ``<name>-verify`` with
the settings its ``config_snapshot`` records. Settings added after the
original ran (hybrid channels, fusion method and weights, parent expansion) are
taken from the commit that recorded it. Questions are evaluated one at a time
(``EVAL_CONCURRENCY=1``), so latencies are uncontended apart from the Jina
reranker's per-minute token limit. Runs whose ``-verify`` file already exists
are skipped. Reranked runs come last, because they spend reranker tokens.

    python -m scripts.rerun_retrieval_experiments                       # all
    python -m scripts.rerun_retrieval_experiments experiment-008-neural-sparse-only

Needs the full stack (``make up``, sparse models loaded) and the indexed
chunking versions each run names. Compare with the originals using
``scripts.paired_ci`` or ``scripts.audit_results``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

BASE = {
    "RETRIEVAL_TOP_K": "8",
    "RETRIEVAL_MIN_SCORE": "0.35",
    "RETRIEVAL_CANDIDATE_LIMIT": "50",
    "EVAL_CONCURRENCY": "1",
    "ENABLE_LATE_INTERACTION": "false",
    "ENABLE_METADATA_NARROWING": "false",
    "ENABLE_RERANKING": "false",
    "ENABLE_PARENT_EXPANSION": "false",
    "HYBRID_CHANNELS": '["dense","bm25","sparse"]',
    "FUSION_METHOD": "rrf",
    "FUSION_RRF_K": "60",
    "FUSION_WEIGHTS": '{"dense":1.0,"bm25":1.0,"sparse":1.0}',
    "RERANK_CANDIDATES": "20",
}


def chunks(strategy: str, size: int, overlap: int) -> dict[str, str]:
    return {
        "CHUNKING_STRATEGY": strategy,
        "CHUNKING_VERSION": f"{strategy}-s{size}-o{overlap}",
        "CHUNK_SIZE_TOKENS": str(size),
        "CHUNK_OVERLAP_TOKENS": str(overlap),
    }


C256 = chunks("contextual", 256, 32)
HC256 = chunks("hierarchical_contextual", 256, 32)
RR = {"ENABLE_RERANKING": "true"}

RUNS: list[tuple[str, str, dict[str, str]]] = [
    # Stage 5 matrix (dense)
    *[
        (f"experiment-002-{s}-{z}-{o}", "dev", {**chunks(s, z, o), "RETRIEVAL_MODE": "dense"})
        for s, z, o in [
            ("fixed", 512, 64),
            ("structure_aware", 512, 64),
            ("hierarchical", 512, 64),
            ("contextual", 512, 64),
            ("hierarchical_contextual", 512, 64),
            ("hierarchical_contextual", 256, 32),
            ("contextual", 128, 16),
            ("contextual", 192, 24),
            ("contextual", 256, 32),
            ("contextual", 768, 96),
            ("contextual", 1024, 128),
        ]
    ],
    # Stage 6 dev, no reranking
    ("experiment-007-bm25-only", "dev", {**C256, "RETRIEVAL_MODE": "bm25"}),
    ("experiment-008-neural-sparse-only", "dev", {**C256, "RETRIEVAL_MODE": "sparse"}),
    (
        "experiment-009-sparse-narrowed",
        "dev",
        {**C256, "RETRIEVAL_MODE": "sparse", "ENABLE_METADATA_NARROWING": "true"},
    ),
    (
        "experiment-010-dense-narrowed",
        "dev",
        {**C256, "RETRIEVAL_MODE": "dense", "ENABLE_METADATA_NARROWING": "true"},
    ),
    ("experiment-011-hybrid-rrf-all", "dev", {**C256, "RETRIEVAL_MODE": "hybrid"}),
    (
        "experiment-012-hybrid-rrf-bm25-sparse",
        "dev",
        {**C256, "RETRIEVAL_MODE": "hybrid", "HYBRID_CHANNELS": '["bm25","sparse"]'},
    ),
    (
        "experiment-013-hybrid-weighted-all",
        "dev",
        {**C256, "RETRIEVAL_MODE": "hybrid", "FUSION_METHOD": "weighted"},
    ),
    (
        "experiment-014-hybrid-rrf-strength-weighted",
        "dev",
        {
            **C256,
            "RETRIEVAL_MODE": "hybrid",
            "FUSION_WEIGHTS": '{"dense":0.5,"bm25":1.0,"sparse":2.0}',
        },
    ),
    ("experiment-020-hc-sparse-noexpand", "dev", {**HC256, "RETRIEVAL_MODE": "sparse"}),
    (
        "experiment-020-hc-sparse-expand",
        "dev",
        {**HC256, "RETRIEVAL_MODE": "sparse", "ENABLE_PARENT_EXPANSION": "true"},
    ),
    # Stage 6 validation, no reranking
    ("experiment-018-val-dense", "validation", {**C256, "RETRIEVAL_MODE": "dense"}),
    ("experiment-018-val-bm25", "validation", {**C256, "RETRIEVAL_MODE": "bm25"}),
    ("experiment-018-val-sparse", "validation", {**C256, "RETRIEVAL_MODE": "sparse"}),
    ("experiment-018-val-rrf-all", "validation", {**C256, "RETRIEVAL_MODE": "hybrid"}),
    # Reranked (Jina reranker tokens), most decision-relevant first
    ("experiment-015-sparse-reranked", "dev", {**C256, "RETRIEVAL_MODE": "sparse", **RR}),
    (
        "experiment-018-val-sparse-reranked",
        "validation",
        {**C256, "RETRIEVAL_MODE": "sparse", **RR},
    ),
    ("experiment-016-hybrid-rrf-all-reranked", "dev", {**C256, "RETRIEVAL_MODE": "hybrid", **RR}),
    (
        "experiment-018-val-rrf-all-reranked",
        "validation",
        {**C256, "RETRIEVAL_MODE": "hybrid", **RR},
    ),
    (
        "experiment-020-hc-hybrid-reranked-expand",
        "dev",
        {**HC256, "RETRIEVAL_MODE": "hybrid", "ENABLE_PARENT_EXPANSION": "true", **RR},
    ),
    (
        "experiment-020-val-hc-hybrid-reranked-expand",
        "validation",
        {**HC256, "RETRIEVAL_MODE": "hybrid", "ENABLE_PARENT_EXPANSION": "true", **RR},
    ),
    (
        "experiment-017-hybrid-rrf-bm25-sparse-reranked",
        "dev",
        {**C256, "RETRIEVAL_MODE": "hybrid", "HYBRID_CHANNELS": '["bm25","sparse"]', **RR},
    ),
]


def main() -> int:
    only = set(sys.argv[1:])
    for original, split, env in RUNS:
        name = f"{original}-verify"
        if only and original not in only:
            continue
        if (ROOT / "evaluation" / "results" / f"{name}.json").exists():
            print(f"SKIP {name} (exists)", flush=True)
            continue
        run_env = {**os.environ, **BASE, **env, "PYTHONUTF8": "1"}
        cmd = [
            PY,
            "-m",
            "app.evaluation.cli",
            "run",
            "--name",
            name,
            "--split",
            split,
            "--retrieval-only",
            "--no-judge",
            "--description",
            f"Results verification (Phase 3): re-run of {original} from its recorded configuration.",
            "--notes",
            "config: " + json.dumps(env, sort_keys=True),
        ]
        print(f"RUN {name}", flush=True)
        done = subprocess.run(
            cmd,
            cwd=ROOT,
            env=run_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        tail = (done.stdout + done.stderr).strip().splitlines()[-3:]
        print(
            f"END {name} exit={done.returncode} :: " + " | ".join(t[:200] for t in tail), flush=True
        )
    print("ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
