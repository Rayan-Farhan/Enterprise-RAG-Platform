"""Uncontended retrieval latency per configuration (results verification, Phase 6).

Batch evaluation latencies mix the pipeline with whatever else was running and
with provider rate limiting. This times ``retrieve()`` alone, one query at a
time, over the same fixed sample of dev questions for every configuration, and
can space the queries so a metered provider is never throttled (the Jina
reranker allows 100k tokens/minute, one rerank is ~8.5k).

    python -m scripts.measure_retrieval_latency                       # every config
    python -m scripts.measure_retrieval_latency --configs sparse sparse+rerank --spacing 7

Each configuration gets two warm-up queries that are not timed (client set-up,
model loading). Output: p50/p95 in seconds per configuration, plus the machine,
date and sample size, as JSON (``--out``) and text.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import AppSettings
from app.db.session import get_session_factory
from app.evaluation.dataset import load_split
from app.evaluation.metrics.system import percentile
from app.evaluation.schemas import DatasetSplit
from app.retrieval.channels import get_retriever

C256 = {"CHUNKING_STRATEGY": "contextual", "CHUNKING_VERSION": "contextual-s256-o32"}

CONFIGS: dict[str, dict[str, Any]] = {
    "dense": {**C256, "RETRIEVAL_MODE": "dense"},
    "bm25": {**C256, "RETRIEVAL_MODE": "bm25"},
    "sparse": {**C256, "RETRIEVAL_MODE": "sparse"},
    "rrf-all": {**C256, "RETRIEVAL_MODE": "hybrid"},
    "weighted-all": {**C256, "RETRIEVAL_MODE": "hybrid", "FUSION_METHOD": "weighted"},
    "sparse+rerank": {**C256, "RETRIEVAL_MODE": "sparse", "ENABLE_RERANKING": True},
    "baseline-dense-fixed": {
        "CHUNKING_STRATEGY": "fixed",
        "CHUNKING_VERSION": "fixed-s512-o64",
        "RETRIEVAL_MODE": "dense",
    },
}


async def measure(
    name: str, overrides: dict[str, Any], queries: list[str], spacing: float
) -> dict[str, Any]:
    settings = AppSettings(**overrides)
    retriever = get_retriever(settings)
    timings: list[float] = []
    async with get_session_factory()() as session:
        for index, query in enumerate(queries):
            started = time.perf_counter()
            await retriever.retrieve(query=query, session=session)
            elapsed = time.perf_counter() - started
            if index >= 2:  # first two warm the clients and models
                timings.append(elapsed)
            if spacing:
                await asyncio.sleep(spacing)
    return {
        "config": name,
        "settings": overrides,
        "n": len(timings),
        "p50_s": round(percentile(timings, 0.50), 3),
        "p95_s": round(percentile(timings, 0.95), 3),
        "max_s": round(max(timings), 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--questions", type=int, default=32, help="timed questions + 2 warm-up")
    parser.add_argument("--spacing", type=float, default=0.0, help="seconds between queries")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    # Every answerable dev question has evidence; take a fixed, spread-out sample.
    dev = [q for q in load_split(DatasetSplit.DEV) if q.expected_element_ids()]
    step = max(1, len(dev) // (args.questions + 2))
    queries = [q.question for q in dev[::step]][: args.questions + 2]

    async def run_all() -> list[dict[str, Any]]:
        # One event loop for every config: the database engine is bound to it.
        return [await measure(n, CONFIGS[n], queries, args.spacing) for n in args.configs]

    rows = asyncio.run(run_all())
    report = {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": f"{platform.processor() or platform.machine()} / {os.cpu_count()} threads / "
        f"{platform.system()} {platform.release()}",
        "spacing_s": args.spacing,
        "results": rows,
    }
    for row in rows:
        print(
            f"{row['config']:22} n={row['n']:3}  p50 {row['p50_s']:.3f}s  p95 {row['p95_s']:.3f}s"
        )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
