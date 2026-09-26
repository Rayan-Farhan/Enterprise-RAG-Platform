"""Neural sparse channel: index, pipeline, query, and toggling (Task 6.2)."""

from __future__ import annotations

import uuid

import pytest

from app.core.config import AppSettings
from app.retrieval.channels import get_retriever
from app.retrieval.lexical_store import INDEX_SETTINGS
from app.retrieval.schemas import RetrievalFilters
from app.retrieval.sparse import SparseRetriever
from app.retrieval.sparse_store import (
    SPARSE_FIELD,
    build_sparse_query,
    ingest_pipeline_body,
    sparse_index_settings,
)


class TestIndexAndPipeline:
    def test_sparse_index_extends_the_bm25_mapping_with_rank_features(self) -> None:
        body = sparse_index_settings("my-pipeline")
        properties = body["mappings"]["properties"]

        assert properties[SPARSE_FIELD] == {"type": "rank_features"}
        assert body["settings"]["index"]["default_pipeline"] == "my-pipeline"
        # Same filter and ACL fields as BM25, so Task 6.3 pushes one filter to both.
        for field in ("chunking_version", "department", "allowed_roles", "classification"):
            assert properties[field] == INDEX_SETTINGS["mappings"]["properties"][field]

    def test_building_the_sparse_mapping_does_not_mutate_the_bm25_one(self) -> None:
        sparse_index_settings("p")

        assert SPARSE_FIELD not in INDEX_SETTINGS["mappings"]["properties"]
        assert "default_pipeline" not in INDEX_SETTINGS["settings"]["index"]

    def test_pipeline_encodes_content_into_the_sparse_field(self) -> None:
        processor = ingest_pipeline_body("model-123")["processors"][0]["sparse_encoding"]

        assert processor == {"model_id": "model-123", "field_map": {"content": SPARSE_FIELD}}


class TestQuery:
    def test_query_uses_the_tokenizer_not_the_encoder(self) -> None:
        """Doc-only mode: query time runs no model inference."""
        body = build_sparse_query("late enrollees", limit=6, tokenizer_model_id="tok-1")
        clause = body["query"]["bool"]["must"][0]["neural_sparse"][SPARSE_FIELD]

        assert clause == {"query_text": "late enrollees", "model_id": "tok-1"}
        assert body["size"] == 6

    def test_term_weights_are_not_returned(self) -> None:
        body = build_sparse_query("q", limit=1, tokenizer_model_id="t")
        assert body["_source"] == {"excludes": [SPARSE_FIELD]}

    def test_filters_are_pushed_into_filter_context(self) -> None:
        doc = uuid.uuid4()
        body = build_sparse_query(
            "q",
            limit=3,
            tokenizer_model_id="t",
            filters=RetrievalFilters(document_ids=[doc], policy_type="leave"),
            chunking_version="contextual-s256-o32",
        )
        clauses = body["query"]["bool"]["filter"]

        assert {"term": {"chunking_version": "contextual-s256-o32"}} in clauses
        assert {"terms": {"document_id": [str(doc)]}} in clauses
        assert {"term": {"policy_type": "leave"}} in clauses


class TestToggle:
    def test_sparse_mode_is_refused_while_the_channel_is_disabled(self) -> None:
        settings = AppSettings(
            APP_ENV="testing", RETRIEVAL_MODE="sparse", ENABLE_NEURAL_SPARSE=False
        )

        with pytest.raises(ValueError, match="ENABLE_NEURAL_SPARSE"):
            get_retriever(settings)

    def test_sparse_mode_selects_the_sparse_channel_when_enabled(self) -> None:
        settings = AppSettings(
            APP_ENV="testing", RETRIEVAL_MODE="sparse", ENABLE_NEURAL_SPARSE=True
        )
        retriever = get_retriever(settings)

        assert isinstance(retriever, SparseRetriever)
        assert retriever.channel == "neural_sparse"

    def test_the_channel_is_on_by_default(self) -> None:
        assert AppSettings(APP_ENV="testing").ENABLE_NEURAL_SPARSE is True

    def test_config_snapshot_names_the_index_and_both_models(self) -> None:
        settings = AppSettings(APP_ENV="testing", ENABLE_NEURAL_SPARSE=True)
        snapshot = SparseRetriever(settings=settings).config_snapshot(8, None)

        assert snapshot["channels"] == ["neural_sparse"]
        assert snapshot["sparse_index"] == settings.SPARSE_INDEX_NAME
        assert str(snapshot["sparse_doc_model"]).endswith("@1.0.0")
        assert "lexical_index" not in snapshot
