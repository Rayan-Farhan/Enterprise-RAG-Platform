"""Jina AI provider for dense embeddings and neural reranking (ADR-010, ADR-011, ADR-051)."""

import time
from typing import Any

import httpx

from app.models.providers.base import BaseProvider
from app.models.schemas import (
    EmbeddingResult,
    EmbeddingsResponse,
    ModelMetadata,
    RerankResult,
    ScoredDocument,
    TokenCounts,
)


class JinaProvider(BaseProvider):
    """Provider communicating with Jina AI embeddings and reranker endpoints."""

    def __init__(
        self,
        api_key: str,
        default_embed_model: str = "jina-embeddings-v3",
        default_rerank_model: str = "jina-reranker-v2-base-multilingual",
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(provider_name="jina", timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.default_embed_model = default_embed_model
        self.default_rerank_model = default_rerank_model
        self.embed_url = "https://api.jina.ai/v1/embeddings"
        self.rerank_url = "https://api.jina.ai/v1/rerank"

    async def embed(
        self,
        texts: list[str],
        model_name: str | None = None,
        task: str = "retrieval.passage",
        dimensions: int | None = 1024,
    ) -> EmbeddingsResponse:
        """Generate dense vector embeddings."""
        target_model = model_name or self.default_embed_model
        start_time = time.perf_counter()

        if not texts:
            return EmbeddingsResponse(
                embeddings=[],
                metadata=ModelMetadata(
                    provider="jina",
                    model_name=target_model,
                    latency_ms=0.0,
                ),
            )

        self.require_credentials(self.api_key, "JINA_API_KEY")

        payload: dict[str, Any] = {
            "model": target_model,
            "input": texts,
            "task": task,
        }
        if dimensions:
            payload["dimensions"] = dimensions

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async def _call() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(self.embed_url, json=payload, headers=headers)
                self.raise_for_response(res)
                return res.json()  # type: ignore[no-any-return]

        data = await self.execute_with_retry(_call)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        raw_data = data.get("data", [])
        embeddings = [
            EmbeddingResult(embedding=item.get("embedding", []), index=item.get("index", idx))
            for idx, item in enumerate(raw_data)
        ]
        usage = data.get("usage", {})

        return EmbeddingsResponse(
            embeddings=embeddings,
            metadata=ModelMetadata(
                provider="jina",
                model_name=target_model,
                latency_ms=duration_ms,
                token_counts=TokenCounts(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
            ),
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        model_name: str | None = None,
    ) -> RerankResult:
        """Rerank documents against a query."""
        target_model = model_name or self.default_rerank_model
        start_time = time.perf_counter()

        if not documents:
            return RerankResult(
                results=[],
                metadata=ModelMetadata(
                    provider="jina",
                    model_name=target_model,
                    latency_ms=0.0,
                ),
            )

        self.require_credentials(self.api_key, "JINA_API_KEY")

        payload: dict[str, Any] = {
            "model": target_model,
            "query": query,
            "documents": documents,
        }
        if top_k:
            payload["top_n"] = top_k

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async def _call() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(self.rerank_url, json=payload, headers=headers)
                self.raise_for_response(res)
                return res.json()  # type: ignore[no-any-return]

        data = await self.execute_with_retry(_call)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        raw_results = data.get("results", [])
        scored_docs = [
            ScoredDocument(
                index=item.get("index", 0),
                text=documents[item.get("index", 0)]
                if item.get("index", 0) < len(documents)
                else item.get("document", {}).get("text", ""),
                score=float(item.get("relevance_score", 0.0)),
            )
            for item in raw_results
        ]
        usage = data.get("usage", {})

        return RerankResult(
            results=scored_docs,
            metadata=ModelMetadata(
                provider="jina",
                model_name=target_model,
                latency_ms=duration_ms,
                token_counts=TokenCounts(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
            ),
        )
