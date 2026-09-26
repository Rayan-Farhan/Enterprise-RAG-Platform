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

    settings: AppSettings

    @property
    def channel(self) -> str:
        """The channel name recorded on every hit and in the config snapshot."""
        ...

    async def retrieve(
        self,
        query: str,
        session: AsyncSession | None = None,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
        min_score: float | None = None,
    ) -> RetrievalResult: ...

    async def candidate_count(self, filters: RetrievalFilters | None) -> int:
        """How many chunks ``filters`` admit: the pool this channel would rank."""
        ...


def get_retriever(settings: AppSettings | None = None) -> Retriever:
    """Return the retriever ``RETRIEVAL_MODE`` selects, narrowed if enabled.

    Imported lazily so selecting one channel never constructs the other's client.
    """
    channel = _channel_retriever(settings)
    resolved = settings or get_settings()
    if resolved.ENABLE_METADATA_NARROWING:
        from app.retrieval.narrowing import NarrowingRetriever

        return NarrowingRetriever(channel, settings=resolved)
    return channel


def _channel_retriever(settings: AppSettings | None) -> Retriever:
    resolved = settings or get_settings()
    if resolved.RETRIEVAL_MODE == "hybrid":
        from app.retrieval.fusion import FusionRetriever

        return FusionRetriever(settings=resolved)
    if resolved.RETRIEVAL_MODE == "sparse" and not resolved.ENABLE_NEURAL_SPARSE:
        # Fail at selection, not at the first query: a sparse run against an
        # index nobody is populating would measure an empty channel.
        raise ValueError(
            "RETRIEVAL_MODE=sparse requires ENABLE_NEURAL_SPARSE=true and the "
            "models from scripts/setup_neural_sparse.py"
        )
    # Without explicit settings, reuse the process-wide singleton channels.
    return build_channel(resolved.RETRIEVAL_MODE, settings)


def build_channel(name: str, settings: AppSettings | None) -> Retriever:
    """One named channel. Explicit settings build a fresh instance from them."""
    if name == "sparse":
        from app.retrieval.sparse import SparseRetriever, get_sparse_retriever

        return get_sparse_retriever() if settings is None else SparseRetriever(settings=settings)
    if name == "bm25":
        from app.retrieval.lexical import LexicalRetriever, get_lexical_retriever

        return get_lexical_retriever() if settings is None else LexicalRetriever(settings=settings)
    if name == "dense":
        from app.retrieval.dense import DenseRetriever, get_dense_retriever

        return get_dense_retriever() if settings is None else DenseRetriever(settings=settings)
    raise ValueError(f"Unknown retrieval channel '{name}'")
