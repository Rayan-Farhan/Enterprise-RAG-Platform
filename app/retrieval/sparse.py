"""Neural sparse retrieval (Task 6.2, ADR-008).

Same contract, filters and PostgreSQL rehydration as the BM25 channel; only the
index and the scoring differ. Like BM25 its scores are unbounded, so the dense
cosine floor does not apply and the channel's cut is its rank.
"""

from __future__ import annotations

from app.core.config import AppSettings
from app.retrieval.lexical import LexicalRetriever
from app.retrieval.schemas import RetrievalFilters
from app.retrieval.sparse_store import OpenSearchSparseStore, get_sparse_store


class SparseRetriever(LexicalRetriever):
    """Runs a doc-only neural sparse query and returns the top chunks."""

    channel = "neural_sparse"

    def __init__(
        self,
        sparse_store: OpenSearchSparseStore | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        store = sparse_store or (
            OpenSearchSparseStore(settings=settings) if settings else get_sparse_store()
        )
        super().__init__(lexical_store=store, settings=settings)

    def config_snapshot(
        self,
        top_k: int,
        filters: RetrievalFilters | None,
    ) -> dict[str, object]:
        snapshot = super().config_snapshot(top_k, filters)
        snapshot.pop("lexical_index", None)
        snapshot.update(
            sparse_index=self.settings.SPARSE_INDEX_NAME,
            sparse_doc_model=f"{self.settings.SPARSE_DOC_MODEL}@{self.settings.SPARSE_DOC_MODEL_VERSION}",
            sparse_query_tokenizer=(
                f"{self.settings.SPARSE_QUERY_TOKENIZER}@{self.settings.SPARSE_QUERY_TOKENIZER_VERSION}"
            ),
        )
        return snapshot


_sparse_retriever: SparseRetriever | None = None


def get_sparse_retriever() -> SparseRetriever:
    """Return the singleton SparseRetriever."""
    global _sparse_retriever
    if _sparse_retriever is None:
        _sparse_retriever = SparseRetriever()
    return _sparse_retriever
