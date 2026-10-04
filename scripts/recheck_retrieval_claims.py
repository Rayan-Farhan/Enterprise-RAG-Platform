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


async def run() -> dict[str, Any]:
    async with get_session_factory()() as session:
        return {
            "code_queries": await code_queries(session),
            "narrowing": await narrowing(session),
            "score_separation": await score_separation(session),
        }


def main() -> int:
    print(json.dumps(asyncio.run(run()), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
