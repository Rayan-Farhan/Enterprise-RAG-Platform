"""The retrieval channel contract and its selection (Stage 6, ADR-007/008).

Generation and evaluation depend on this protocol, not on a concrete channel,
so switching ``RETRIEVAL_MODE`` changes which index answers a question without
touching either. Task 6.4 adds a fusion retriever that satisfies the same
protocol by combining the channels below.
"""

from __future__ import annotations

from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.retrieval.schemas import RetrievalFilters, RetrievalResult


class Retriever(Protocol):
    """Anything that turns a query into ranked, provenance-bearing chunks."""

    channel: str
    settings: AppSettings

    async def retrieve(
        self,
        query: str,
        session: AsyncSession | None = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult: ...


def get_retriever(settings: AppSettings | None = None) -> Retriever:
    """Return the retriever ``RETRIEVAL_MODE`` selects.

    Imported lazily so selecting one channel never constructs the other's client.
    """
    resolved = settings or get_settings()
    if resolved.RETRIEVAL_MODE == "bm25":
        from app.retrieval.lexical import LexicalRetriever, get_lexical_retriever

        return get_lexical_retriever() if settings is None else LexicalRetriever(settings=resolved)

    from app.retrieval.dense import DenseRetriever, get_dense_retriever

    return get_dense_retriever() if settings is None else DenseRetriever(settings=resolved)
