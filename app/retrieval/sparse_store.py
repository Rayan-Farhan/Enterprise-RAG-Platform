"""OpenSearch neural sparse index over chunks (Task 6.2, ADR-008).

The third retrieval signal. BM25 matches the words a chunk contains; a neural
sparse encoder adds weighted terms it *implies* — "late enrollees must wait 365
days" also scores for "dentist", "teeth", "delayed" — while staying an inverted
index, so it keeps BM25's exactness on terms that do appear.

It runs in OpenSearch's "doc-only" mode:

* at index time an ingest pipeline runs the sparse *encoder* over each chunk and
  stores the weighted terms in a ``rank_features`` field;
* at query time only the *tokenizer* runs, so a search costs no model inference.

Like the BM25 index it is derived from PostgreSQL (ADR-002) and keyed on the
deterministic chunk ID. It is a separate index from BM25 on purpose: encoding
is CPU-expensive and needs ML models deployed, so the channel can be switched
off (``ENABLE_NEURAL_SPARSE``) without touching BM25.

The models, the pipeline and the cluster settings they need are provisioned by
``scripts/setup_neural_sparse.py``; this module only finds and uses them.
"""

from __future__ import annotations

import copy
from typing import Any

from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import NotFoundError

from app.core.config import AppSettings
from app.core.logging import get_logger
from app.retrieval.lexical_store import (
    INDEX_SETTINGS,
    LexicalHit,
    OpenSearchLexicalStore,
    build_filter,
    to_document,
)
from app.retrieval.schemas import ChunkPayload, RetrievalFilters

logger = get_logger("app.retrieval.sparse_store")

SPARSE_FIELD = "content_sparse"

# Encoding runs inside the bulk request, roughly a second per chunk on CPU, so
# batches are small and the timeout is generous.
_BULK_CHUNK_SIZE = 16
_BULK_TIMEOUT_SECONDS = 600


class SparseModelNotReadyError(RuntimeError):
    """The sparse encoder or tokenizer is not registered and deployed."""


def sparse_index_settings(pipeline: str) -> dict[str, Any]:
    """The BM25 mapping plus a ``rank_features`` field filled by the pipeline.

    Reusing the BM25 mapping keeps every filter and Stage 8 ACL field identical
    across the two indexes, which Task 6.3's filter push-down relies on.
    """
    body = copy.deepcopy(INDEX_SETTINGS)
    body["settings"]["index"]["default_pipeline"] = pipeline
    body["mappings"]["properties"][SPARSE_FIELD] = {"type": "rank_features"}
    return body


def ingest_pipeline_body(doc_model_id: str) -> dict[str, Any]:
    return {
        "description": "Neural sparse document expansion for chunk content (Task 6.2)",
        "processors": [
            {
                "sparse_encoding": {
                    "model_id": doc_model_id,
                    "field_map": {"content": SPARSE_FIELD},
                }
            }
        ],
    }


def build_sparse_query(
    query: str,
    limit: int,
    tokenizer_model_id: str,
    filters: RetrievalFilters | None = None,
    chunking_version: str | None = None,
) -> dict[str, Any]:
    """Build the neural sparse request body.

    The expanded term weights are excluded from the returned source: they are
    large, and hits are rehydrated from PostgreSQL anyway.
    """
    return {
        "size": limit,
        "_source": {"excludes": [SPARSE_FIELD]},
        "query": {
            "bool": {
                "must": [
                    {
                        "neural_sparse": {
                            SPARSE_FIELD: {
                                "query_text": query,
                                "model_id": tokenizer_model_id,
                            }
                        }
                    }
                ],
                "filter": build_filter(filters, chunking_version),
            }
        },
    }


def find_models(client: OpenSearch, name: str) -> list[dict[str, Any]]:
    """Registered models with this name, newest first.

    Model weights are stored as chunk documents beside the model document; only
    the model document itself has no ``chunk_number``.
    """
    response = client.transport.perform_request(
        "POST",
        "/_plugins/_ml/models/_search",
        body={
            "size": 10,
            "query": {
                "bool": {
                    "must": [{"term": {"name.keyword": name}}],
                    "must_not": [{"exists": {"field": "chunk_number"}}],
                }
            },
            "sort": [{"created_time": {"order": "desc"}}],
        },
    )
    return list(response.get("hits", {}).get("hits", []))


class OpenSearchSparseStore(OpenSearchLexicalStore):
    """The neural sparse index: same client and filters as BM25, different scoring."""

    def __init__(
        self,
        client: OpenSearch | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        super().__init__(client=client, settings=settings)
        self.index_name = self.settings.SPARSE_INDEX_NAME
        self.pipeline = self.settings.SPARSE_INGEST_PIPELINE
        self._model_ids: dict[str, str] = {}

    # -- models -------------------------------------------------------------

    def model_id(self, name: str, version: str) -> str:
        """Resolve a deployed model's ID by name.

        OpenSearch assigns model IDs at registration, so they differ per cluster;
        resolving by name keeps configuration portable, and the result is cached
        because a deployed model's ID never changes. Matching is on name alone:
        the stored ``model_version`` is OpenSearch's counter within the model
        group ("1", "2", …), not the pretrained release ("1.0.0"), so it cannot
        be matched against configuration. ``version`` is kept in the cache key
        and error message so a model change is visible.
        """
        key = f"{name}@{version}"
        if key in self._model_ids:
            return self._model_ids[key]

        for hit in find_models(self.client, name):
            if hit.get("_source", {}).get("model_state") == "DEPLOYED":
                self._model_ids[key] = str(hit["_id"])
                return self._model_ids[key]

        raise SparseModelNotReadyError(
            f"Sparse model '{name}' version {version} is not deployed. "
            f"Run `python -m scripts.setup_neural_sparse`."
        )

    @property
    def doc_model_id(self) -> str:
        return self.model_id(self.settings.SPARSE_DOC_MODEL, self.settings.SPARSE_DOC_MODEL_VERSION)

    @property
    def tokenizer_model_id(self) -> str:
        return self.model_id(
            self.settings.SPARSE_QUERY_TOKENIZER, self.settings.SPARSE_QUERY_TOKENIZER_VERSION
        )

    # -- index --------------------------------------------------------------

    def ensure_pipeline(self) -> None:
        """Create or update the encoding pipeline to point at the deployed encoder."""
        self.client.ingest.put_pipeline(
            id=self.pipeline, body=ingest_pipeline_body(self.doc_model_id)
        )

    def ensure_index(self) -> bool:
        """Create the index, with the encoding pipeline as its default. True if created."""
        if self.client.indices.exists(index=self.index_name):
            return False
        self.ensure_pipeline()
        self.client.indices.create(index=self.index_name, body=sparse_index_settings(self.pipeline))
        logger.info("opensearch_sparse_index_created", index=self.index_name)
        return True

    def existing_ids(self, chunk_ids: list[str]) -> set[str]:
        """Chunk IDs already encoded. Encoding is the expensive step, so it is skipped."""
        if not chunk_ids or not self.client.indices.exists(index=self.index_name):
            return set()
        found: set[str] = set()
        for start in range(0, len(chunk_ids), 500):
            batch = chunk_ids[start : start + 500]
            response = self.client.mget(index=self.index_name, body={"ids": batch}, _source=False)
            found.update(doc["_id"] for doc in response.get("docs", []) if doc.get("found"))
        return found

    def upsert(self, payloads: list[ChunkPayload], refresh: bool = True) -> int:
        """Index chunks through the encoding pipeline, keyed on chunk ID."""
        if not payloads:
            return 0

        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": payload.chunk_id,
                "_source": to_document(payload),
            }
            for payload in payloads
        ]
        indexed, errors = helpers.bulk(
            self.client,
            actions,
            chunk_size=_BULK_CHUNK_SIZE,
            request_timeout=_BULK_TIMEOUT_SECONDS,
            raise_on_error=False,
        )
        if errors:
            raise RuntimeError(
                f"OpenSearch sparse bulk index failed for {len(errors)} of {len(actions)} "
                f"chunks; first error: {errors[0]}"
            )
        if refresh:
            self.client.indices.refresh(index=self.index_name)
        logger.info("opensearch_sparse_upsert_complete", documents=indexed)
        return int(indexed)

    def search(
        self,
        query: str,
        limit: int,
        filters: RetrievalFilters | None = None,
        chunking_version: str | None = None,
    ) -> list[LexicalHit]:
        """Run a filtered neural sparse search. An absent index yields no hits."""
        if not self.client.indices.exists(index=self.index_name):
            logger.warning("opensearch_sparse_index_absent_returning_no_results")
            return []

        body = build_sparse_query(query, limit, self.tokenizer_model_id, filters, chunking_version)
        try:
            response = self.client.search(index=self.index_name, body=body)
        except NotFoundError:
            return []

        return [
            LexicalHit(
                chunk_id=str(hit["_id"]),
                score=float(hit.get("_score") or 0.0),
                payload=dict(hit.get("_source") or {}),
            )
            for hit in response.get("hits", {}).get("hits", [])
        ]


_sparse_store: OpenSearchSparseStore | None = None


def get_sparse_store() -> OpenSearchSparseStore:
    """Return the singleton OpenSearchSparseStore."""
    global _sparse_store
    if _sparse_store is None:
        _sparse_store = OpenSearchSparseStore()
    return _sparse_store
