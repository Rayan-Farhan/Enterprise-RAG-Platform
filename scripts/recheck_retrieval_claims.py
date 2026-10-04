"""Re-derive three hand-probed retrieval claims from the live index (Phase 7).

1. **Exact-code queries (Task 6.1 exit condition).** For each query, the rank of
   the first chunk whose text contains the literal code, under dense (top 10,
   no score floor) and BM25.
2. **Metadata narrowing (Task 6.3 exit condition).** The narrowing trace for the
   Academic Affairs question in every engine: inferred filters and candidate
   pool before and after.
3. **Abstention score separation (Stage 3).** Dense top-1 similarity for every
   dev question, answerable vs ``negative_unsupported``, on the Stage 3
   chunking (fixed 512/64), to see whether ``RETRIEVAL_MIN_SCORE`` = 0.35 sits
   in a gap. The original figure came from five hand-probed queries.

4. **Reranker cost (Task 6.5).** Jina-reported tokens for one 20-candidate
   rerank, over ten dev questions, spaced under the provider's minute limit.
5. **Parent-expansion budget (Task 5.2).** On experiment 004's 50 questions
   (hierarchical_contextual 256/32, dense, expansion on): parents expanded,
   leaves dropped for the expansion budget, and chunks the context assembler
   dropped for the generation budget.
6. **Chunk sizes (ADR-006).** Mean stored token count of hierarchical leaves
   and of contextual 256/32 chunks.

    python -m scripts.recheck_retrieval_claims

Needs the full stack with the contextual 256/32 and fixed 512/64 chunk sets
indexed. Uses Jina query embeddings for the dense parts; no generation.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from app.core.config import AppSettings
from app.db.session import get_session_factory
from app.evaluation.dataset import load_split
from app.evaluation.schemas import DatasetSplit, QuestionType
from app.retrieval.channels import get_retriever

C256 = {"CHUNKING_STRATEGY": "contextual", "CHUNKING_VERSION": "contextual-s256-o32"}

CODE_QUERIES = [
    ("Ala. Code 31-2-13", "31-2-13"),
    ("section 2.9 Faculty Handbook", "2.9"),
    ("1-800-248-2342", "1-800-248-2342"),
    ("1-800-292-8868", "1-800-292-8868"),
    ("Section 3.14", "3.14"),
]
NARROWING_QUERY = "What does Academic Affairs require for promotion and tenure review?"


def first_rank(chunks: list[Any], literal: str) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if literal in (chunk.content or ""):
            return rank
    return None


async def code_queries(session: Any) -> list[dict[str, Any]]:
    dense = get_retriever(AppSettings(**C256, RETRIEVAL_MODE="dense", RETRIEVAL_MIN_SCORE=0.0))
    bm25 = get_retriever(AppSettings(**C256, RETRIEVAL_MODE="bm25", RETRIEVAL_MIN_SCORE=0.0))
    rows = []
    for query, literal in CODE_QUERIES:
        d = await dense.retrieve(query=query, session=session, top_k=10, min_score=0.0)
        b = await bm25.retrieve(query=query, session=session, top_k=10, min_score=0.0)
        rows.append(
            {
                "query": query,
                "literal": literal,
                "dense_rank_top10": first_rank(d.chunks, literal),
                "bm25_rank_top10": first_rank(b.chunks, literal),
            }
        )
    return rows


async def narrowing(session: Any) -> list[dict[str, Any]]:
    rows = []
    for mode in ("dense", "bm25", "sparse"):
        retriever = get_retriever(
            AppSettings(**C256, RETRIEVAL_MODE=mode, ENABLE_METADATA_NARROWING=True)
        )
        result = await retriever.retrieve(query=NARROWING_QUERY, session=session)
        trace = dict(result.retrieval_config.get("narrowing") or {})
        rows.append(
            {
                "mode": mode,
                "applied": trace.get("applied"),
                "candidates_before": trace.get("candidates_before"),
                "candidates_after": trace.get("candidates_after"),
                "fell_back": trace.get("fell_back"),
            }
        )
    return rows


async def score_separation(session: Any) -> dict[str, Any]:
    dense = get_retriever(
        AppSettings(
            CHUNKING_STRATEGY="fixed",
            CHUNKING_VERSION="fixed-s512-o64",
            RETRIEVAL_MODE="dense",
            RETRIEVAL_MIN_SCORE=0.0,
        )
    )
    answerable: list[float] = []
    unsupported: list[float] = []
    for question in load_split(DatasetSplit.DEV):
        if question.question_type == QuestionType.ADVERSARIAL:
            continue
        result = await dense.retrieve(query=question.question, session=session, top_k=1)
        top = result.chunks[0].score if result.chunks else 0.0
        if question.question_type == QuestionType.NEGATIVE_UNSUPPORTED:
            unsupported.append(top)
        elif question.expected_element_ids():
            answerable.append(top)

    def summary(values: list[float]) -> dict[str, float]:
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "min": round(ordered[0], 3),
            "median": round(ordered[len(ordered) // 2], 3),
            "max": round(ordered[-1], 3),
        }

    floor = 0.35
    return {
        "answerable": summary(answerable),
        "negative_unsupported": summary(unsupported),
        "answerable_below_floor": sum(1 for s in answerable if s < floor),
        "unsupported_at_or_above_floor": sum(1 for s in unsupported if s >= floor),
        "floor": floor,
    }


async def rerank_tokens(session: Any) -> dict[str, Any]:
    from app.models.gateway import get_model_gateway

    sparse = get_retriever(AppSettings(**C256, RETRIEVAL_MODE="sparse"))
    gateway = get_model_gateway()
    totals: list[int] = []
    questions = [q for q in load_split(DatasetSplit.DEV) if q.expected_element_ids()][::8][:10]
    for question in questions:
        pool = await sparse.retrieve(query=question.question, session=session, top_k=20)
        result = await gateway.rerank(
            query=question.question, documents=[c.content for c in pool.chunks], top_k=8
        )
        totals.append(int(result.metadata.token_counts.total_tokens))
        await asyncio.sleep(7)  # stay under 100k tokens/minute
    ordered = sorted(totals)
    return {
        "calls": len(ordered),
        "candidates": 20,
        "tokens_min": ordered[0],
        "tokens_median": ordered[len(ordered) // 2],
        "tokens_max": ordered[-1],
    }


async def expansion_budget(session: Any) -> dict[str, Any]:
    from pathlib import Path

    from app.generation.context import ContextAssembler
    from app.retrieval.expansion import ParentExpander

    settings = AppSettings(
        CHUNKING_STRATEGY="hierarchical_contextual",
        CHUNKING_VERSION="hierarchical_contextual-s256-o32",
        RETRIEVAL_MODE="dense",
        ENABLE_PARENT_EXPANSION=True,
    )
    record = Path("evaluation/results/experiment-004-hc-256-32-expand.json")
    wanted = {r["question_id"] for r in json.loads(record.read_text(encoding="utf-8"))["results"]}
    retriever = get_retriever(settings)
    expander = ParentExpander(settings)
    assembler = ContextAssembler()
    tally = {"questions": 0, "expanded": 0, "kept_as_leaf": 0, "dropped_for_budget": 0}
    assembly_dropped = 0
    for question in load_split(DatasetSplit.DEV):
        if question.question_id not in wanted:
            continue
        tally["questions"] += 1
        retrieval = await retriever.retrieve(query=question.question, session=session)
        expanded = await expander.expand(retrieval.chunks, session=session)
        tally["expanded"] += expanded.expanded
        tally["kept_as_leaf"] += expanded.kept_as_leaf
        tally["dropped_for_budget"] += expanded.dropped_for_budget
        context = assembler.assemble(query=question.question, chunks=expanded.chunks)
        assembly_dropped += int(getattr(context, "dropped_for_budget", 0) or 0)
    return {**tally, "assembly_dropped_for_budget": assembly_dropped}


async def chunk_sizes(session: Any) -> dict[str, Any]:
    from sqlalchemy import func, select

    from app.db.models.chunk import Chunk

    async def mean(version: str, leaves_only: bool) -> float:
        query = select(func.avg(Chunk.token_count)).where(Chunk.chunking_version == version)
        if leaves_only:
            query = query.where(Chunk.parent_chunk_id.is_not(None))
        return round(float((await session.execute(query)).scalar() or 0), 1)

    return {
        "hierarchical-s512-o64 leaves": await mean("hierarchical-s512-o64", True),
        "contextual-s256-o32 chunks": await mean("contextual-s256-o32", False),
    }


async def run() -> dict[str, Any]:
    async with get_session_factory()() as session:
        return {
            "code_queries": await code_queries(session),
            "narrowing": await narrowing(session),
            "score_separation": await score_separation(session),
            "rerank_tokens": await rerank_tokens(session),
            "expansion_budget": await expansion_budget(session),
            "chunk_sizes": await chunk_sizes(session),
        }


def main() -> int:
    print(json.dumps(asyncio.run(run()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
