"""Local TEI provider for self-hosted text embeddings and reranking (ADR-016, ADR-045, ADR-051)."""

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


class LocalTEIProvider(BaseProvider):
    """Provider communicating with self-hosted Text Embeddings Inference (TEI) instances."""

    def __init__(
        self,
        embed_base_url: str = "http://localhost:8080",
        rerank_base_url: str = "http://localhost:8081",
        default_embed_model: str = "Qwen/Qwen3-Embedding-0.6B",
        default_rerank_model: str = "BAAI/bge-reranker-v2-m3",
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(provider_name="tei", timeout_seconds=timeout_seconds)
        self.embed_base_url = embed_base_url.rstrip("/")
        self.rerank_base_url = rerank_base_url.rstrip("/")
        self.default_embed_model = default_embed_model
        self.default_rerank_model = default_rerank_model

    async def embed(
        self,
        texts: list[str],
        model_name: str | None = None,
    ) -> EmbeddingsResponse:
        """Generate vector embeddings using local TEI."""
        target_model = model_name or self.default_embed_model
        start_time = time.perf_counter()

        if not texts:
            return EmbeddingsResponse(
                embeddings=[],
                metadata=ModelMetadata(provider="tei", model_name=target_model, latency_ms=0.0),
            )

        payload = {"inputs": texts}

        async def _call() -> list[list[float]]:
            url = f"{self.embed_base_url}/embed"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                self.raise_for_response(res)
                return res.json()  # type: ignore[no-any-return]

        raw_embeddings = await self.execute_with_retry(_call)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        embeddings = [
            EmbeddingResult(embedding=emb, index=i) for i, emb in enumerate(raw_embeddings)
        ]

        return EmbeddingsResponse(
            embeddings=embeddings,
            metadata=ModelMetadata(
                provider="tei",
                model_name=target_model,
                latency_ms=duration_ms,
                token_counts=TokenCounts(prompt_tokens=len(texts) * 5, total_tokens=len(texts) * 5),
            ),
        )

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        model_name: str | None = None,
    ) -> RerankResult:
        """Rerank documents using local TEI reranker."""
        target_model = model_name or self.default_rerank_model
        start_time = time.perf_counter()

        if not documents:
            return RerankResult(
                results=[],
                metadata=ModelMetadata(provider="tei", model_name=target_model, latency_ms=0.0),
            )

        payload: dict[str, Any] = {
            "query": query,
            "texts": documents,
        }
        if top_k:
            payload["return_k"] = top_k

        async def _call() -> list[dict[str, Any]]:
            url = f"{self.rerank_base_url}/rerank"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                self.raise_for_response(res)
                return res.json()  # type: ignore[no-any-return]

        raw_results = await self.execute_with_retry(_call)
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        scored_docs = [
            ScoredDocument(
                index=item.get("index", 0),
                text=documents[item.get("index", 0)]
                if item.get("index", 0) < len(documents)
                else "",
                score=float(item.get("score", 0.0)),
            )
            for item in raw_results
        ]

        return RerankResult(
            results=scored_docs,
            metadata=ModelMetadata(
                provider="tei",
                model_name=target_model,
                latency_ms=duration_ms,
                token_counts=TokenCounts(
                    prompt_tokens=len(documents) * 10, total_tokens=len(documents) * 10
                ),
            ),
        )
