"""OpenSearch lexical index over chunks (Task 6.1, ADR-007/008).

OpenSearch is a *derived* store (ADR-002), like Qdrant: every document here is
rebuilt from PostgreSQL and nothing here is authoritative. Documents are keyed
on the chunk ID, which is already deterministic (ADR-036), so re-indexing
overwrites in place and never duplicates.

BM25 is here for what dense retrieval does badly (master §14): policy names,
clause numbers, acronyms, phone numbers, amounts, grades, legal citations. The
mapping therefore indexes chunk text twice:

* ``content`` — prose analysis: lower-cased, ASCII-folded, English stop words
  removed, gently stemmed (``kstem``), so "enrollees" matches "enrollee".
* ``content.exact`` — code analysis: split only on whitespace and bracketing
  punctuation, so ``31-2-13``, ``1-800-248-2342``, ``2.9`` and ``§`` survive as
  single tokens instead of being shredded into digits that match everything.

Chunk text already carries the contextual prefix (document title and section
path), so provenance is scored without separate title or section fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opensearchpy import OpenSearch, helpers
from opensearchpy.exceptions import NotFoundError

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.retrieval.schemas import APPLIES_TO_ALL, ChunkPayload, RetrievalFilters

logger = get_logger("app.retrieval.lexical_store")

# Metadata filters shared with the vector store, plus the Stage 8 ACL
# placeholders. All are keywords so filtering is exact and cheap, and the ACL
# fields exist from day one so enforcing them later is a query change, not a
# re-index — the same reasoning as the Qdrant payload indexes.
_KEYWORD_FIELDS = (
    "chunk_id",
    "document_id",
    "version_id",
    "chunking_version",
    "chunk_type",
    "department",
    "policy_type",
    "policy_status",
    "country",
    "employee_type",
    "grade",
    "confidentiality",
    "audience",
    "tenant_id",
    "department_id",
    "allowed_roles",
    "allowed_users",
    "classification",
)
_FILTER_FIELDS = (
    "department",
    "policy_type",
    "policy_status",
    "country",
    "grade",
)

INDEX_SETTINGS: dict[str, Any] = {
    "settings": {
        "index": {"number_of_shards": 1, "number_of_replicas": 0},
        "analysis": {
            "tokenizer": {
                "hr_code_tokenizer": {
                    "type": "char_group",
                    "tokenize_on_chars": [
                        "whitespace",
                        ",",
                        ";",
                        ":",
                        '"',
                        "'",
                        "!",
                        "?",
                        "(",
                        ")",
                        "[",
                        "]",
                        "{",
                        "}",
                        "|",
                    ],
                }
            },
            "filter": {
                "hr_english_stop": {"type": "stop", "stopwords": "_english_"},
                # Strip sentence punctuation and table-of-contents dot leaders
                # from token edges, keeping the dots and hyphens inside codes.
                "hr_edge_trim": {
                    "type": "pattern_replace",
                    "pattern": "^[.\\-_/*]+|[.\\-_/*]+$",
                    "replacement": "",
                },
                "hr_drop_empty": {"type": "length", "min": 1},
            },
            "analyzer": {
                "hr_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "hr_english_stop", "kstem"],
                },
                "hr_exact": {
                    "type": "custom",
                    "tokenizer": "hr_code_tokenizer",
                    "filter": ["lowercase", "asciifolding", "hr_edge_trim", "hr_drop_empty"],
                },
            },
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            **{name: {"type": "keyword"} for name in _KEYWORD_FIELDS},
            "chunk_index": {"type": "integer"},
            "page_number": {"type": "integer"},
            "token_count": {"type": "integer"},
            "document_title": {"type": "keyword"},
            "content": {
                "type": "text",
                "analyzer": "hr_text",
                "fields": {"exact": {"type": "text", "analyzer": "hr_exact"}},
            },
        },
    },
}


@dataclass
class LexicalHit:
    """A BM25-scored match. Scores are unbounded and only comparable within a query."""

    chunk_id: str
    score: float
    payload: dict[str, Any]


def to_document(payload: ChunkPayload) -> dict[str, Any]:
    """Project a chunk payload onto the index mapping.

    Only mapped fields are sent: the mapping is ``strict``, so a field added to
    ``ChunkPayload`` for Qdrant fails loudly here until it is mapped deliberately.
    """
    data = payload.model_dump(mode="json")
    mapped = set(INDEX_SETTINGS["mappings"]["properties"])
    return {key: value for key, value in data.items() if key in mapped}


def build_query(
    query: str,
    limit: int,
    filters: RetrievalFilters | None = None,
    chunking_version: str | None = None,
) -> dict[str, Any]:
    """Build the BM25 request body.

    Prose and code fields are scored together (``most_fields``) so a query that
    names a code gets the exact-token match on top of the prose match, and an
    ordinary question is unaffected by the code field. A phrase match on the
    prose field rewards chunks that keep the query's words together. Filters sit
    in ``filter`` context: they narrow the candidate set before scoring and never
    change a score (master §13 — narrowing, not re-ranking).
    """
    body: dict[str, Any] = {
        "size": limit,
        "_source": True,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "type": "most_fields",
                            "fields": ["content", "content.exact"],
                        }
                    },
                    {"match_phrase": {"content": {"query": query, "slop": 3, "boost": 2.0}}},
                ],
                "minimum_should_match": 1,
                "filter": build_filter(filters, chunking_version),
            }
        },
    }
    return body


def build_filter(
    filters: RetrievalFilters | None = None,
    chunking_version: str | None = None,
) -> list[dict[str, Any]]:
    """Translate metadata constraints into OpenSearch filter clauses."""
    clauses: list[dict[str, Any]] = []
    if chunking_version:
        clauses.append({"term": {"chunking_version": chunking_version}})
    if filters is None:
        return clauses

    if filters.document_ids:
        clauses.append({"terms": {"document_id": [str(v) for v in filters.document_ids]}})
    if filters.version_ids:
        clauses.append({"terms": {"version_id": [str(v) for v in filters.version_ids]}})
    for field_name in _FILTER_FIELDS:
        value = getattr(filters, field_name)
        if value:
            clauses.append({"term": {field_name: value}})
    if filters.employee_type:
        clauses.append({"terms": {"employee_type": [filters.employee_type, APPLIES_TO_ALL]}})
    if filters.page_number is not None:
        clauses.append({"term": {"page_number": filters.page_number}})
    return clauses


class OpenSearchLexicalStore:
    """Thin, testable wrapper over the OpenSearch index holding chunk text."""

    def __init__(
        self,
        client: OpenSearch | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.index_name = self.settings.OPENSEARCH_INDEX_NAME
        self._client = client

    @property
    def client(self) -> OpenSearch:
        """Lazily construct the client so importing the app never needs OpenSearch."""
        if self._client is None:
            auth = (
                (self.settings.OPENSEARCH_USER, self.settings.OPENSEARCH_PASSWORD)
                if self.settings.OPENSEARCH_USE_SSL
                else None
            )
            self._client = OpenSearch(
                hosts=[
                    {"host": self.settings.OPENSEARCH_HOST, "port": self.settings.OPENSEARCH_PORT}
                ],
                http_auth=auth,
                use_ssl=self.settings.OPENSEARCH_USE_SSL,
                verify_certs=self.settings.OPENSEARCH_VERIFY_CERTS,
                ssl_show_warn=False,
                timeout=30,
                # A derived index being down should cost fusion one channel, fast.
                # The client default (3 retries) took 16 s to give up on a
                # refused connection.
                max_retries=1,
            )
        return self._client

    def ensure_index(self) -> bool:
        """Create the index with its analyzers if absent. Returns True if created."""
        if self.client.indices.exists(index=self.index_name):
            return False
        self.client.indices.create(index=self.index_name, body=INDEX_SETTINGS)
        logger.info("opensearch_index_created", index=self.index_name)
        return True

    def upsert(self, payloads: list[ChunkPayload], refresh: bool = True) -> int:
        """Index chunks keyed on chunk ID. Repeated calls overwrite, never duplicate."""
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
        indexed, errors = helpers.bulk(self.client, actions, raise_on_error=False)
        if errors:
            # A partial bulk failure leaves the lexical channel silently missing
            # chunks, which reads as a retrieval-quality problem later. Fail now.
            raise RuntimeError(
                f"OpenSearch bulk index failed for {len(errors)} of {len(actions)} "
                f"chunks; first error: {errors[0]}"
            )
        if refresh:
            self.client.indices.refresh(index=self.index_name)
        logger.info("opensearch_upsert_complete", documents=indexed)
        return int(indexed)

    def search(
        self,
        query: str,
        limit: int,
        filters: RetrievalFilters | None = None,
        chunking_version: str | None = None,
    ) -> list[LexicalHit]:
        """Run a filtered BM25 search. An absent index yields no hits, not an error."""
        body = build_query(query, limit, filters, chunking_version)
        try:
            response = self.client.search(index=self.index_name, body=body)
        except NotFoundError:
            logger.warning("opensearch_index_absent_returning_no_results", index=self.index_name)
            return []

        return [
            LexicalHit(
                chunk_id=str(hit["_id"]),
                score=float(hit.get("_score") or 0.0),
                payload=dict(hit.get("_source") or {}),
            )
            for hit in response.get("hits", {}).get("hits", [])
        ]

    def count(
        self,
        chunking_version: str | None = None,
        filters: RetrievalFilters | None = None,
    ) -> int:
        """Count indexed chunks a filter admits — the candidate pool a search ranks."""
        if not self.client.indices.exists(index=self.index_name):
            return 0
        body = {"query": {"bool": {"filter": build_filter(filters, chunking_version)}}}
        return int(self.client.count(index=self.index_name, body=body)["count"])

    def update_version_fields(self, version_id: str, fields: dict[str, Any]) -> int:
        """Overwrite metadata fields on every chunk of a version, in place.

        ``pipeline="_none"`` bypasses an index's default pipeline, so on the
        neural sparse index a metadata change does not re-run the encoder over
        every chunk. Returns the number of documents updated.
        """
        if not self.client.indices.exists(index=self.index_name):
            return 0
        response = self.client.update_by_query(
            index=self.index_name,
            body={
                "query": {"term": {"version_id": version_id}},
                "script": {
                    "lang": "painless",
                    "source": "for (e in params.fields.entrySet()) "
                    "{ ctx._source[e.getKey()] = e.getValue(); }",
                    "params": {"fields": fields},
                },
            },
            params={"pipeline": "_none", "refresh": "true", "conflicts": "proceed"},
        )
        return int(response.get("updated", 0))

    def delete_by_version(self, version_id: str) -> None:
        """Remove all documents belonging to a document version."""
        self.client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"version_id": version_id}}},
            refresh=True,
        )
        logger.info("opensearch_version_documents_deleted", version_id=version_id)

    def analyze(self, text: str, analyzer: str) -> list[str]:
        """Tokens an analyzer produces — for tests and for debugging a missed query."""
        response = self.client.indices.analyze(
            index=self.index_name, body={"analyzer": analyzer, "text": text}
        )
        return [token["token"] for token in response.get("tokens", [])]

    def health_check(self) -> bool:
        """Return True when OpenSearch answers a trivial request."""
        try:
            return bool(self.client.ping())
        except Exception as exc:  # noqa: BLE001 - health probe must not raise
            logger.warning("opensearch_health_check_failed", error=str(exc))
            return False


_lexical_store: OpenSearchLexicalStore | None = None


def get_lexical_store() -> OpenSearchLexicalStore:
    """Return the singleton OpenSearchLexicalStore."""
    global _lexical_store
    if _lexical_store is None:
        _lexical_store = OpenSearchLexicalStore()
    return _lexical_store
