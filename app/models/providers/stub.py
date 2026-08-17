"""Explicitly fake provider for keyless development (INFERENCE_PROFILE=stub).

This exists because the alternative was worse. Providers previously substituted
placeholder output whenever an API key was missing, reporting it under the real
provider's name — so an unconfigured environment produced embeddings derived from
list position rather than text, indexed them cleanly, and returned confident
nonsense with every health check green.

The rules that make this safe:

1. It is selected only by an explicit `INFERENCE_PROFILE=stub`, never as a fallback.
2. Every response is labelled `provider="stub"` in metadata, so any consumer —
   including Stage 4's experiment records — can tell it apart from a real run.
3. Embeddings are derived from the *text content*, not from list position, so the
   same text always yields the same vector and identical inputs collide correctly.

It is a plumbing test double, not a model.

What it can demonstrate: that chunking, indexing, filtering, hydration, citation
resolution, streaming, and the abstention *path* are wired together correctly.

What it cannot do — measured, not assumed:

- **Rank by meaning.** These are hashed bag-of-words sketches. Lexical overlap of
  common words dominates, so scores are not comparable across queries: on the real
  HR corpus, "What is the capital city of Mongolia?" scored 0.451 while the
  genuinely relevant "annual leave accrual and overtime compensation" scored 0.375.
- **Judge relevance.** `generate()` reports `grounded` whenever any evidence block
  is present, because it cannot read. Abstention driven by *irrelevant* evidence
  therefore cannot be exercised under this profile — only abstention driven by
  empty retrieval can.

Consequently a score threshold is meaningless here, and no number produced under
this profile is valid evaluation data. Stage 4 baselines require a real provider.
"""

from __future__ import annotations

import hashlib
import math
import time

from app.models.providers.base import BaseProvider
from app.models.schemas import (
    EmbeddingResult,
    EmbeddingsResponse,
    GenerationResult,
    ImagePayload,
    ModelMetadata,
    RerankResult,
    ScoredDocument,
    TokenCounts,
)

STUB_NOTICE = "[STUB PROFILE — no model was called; INFERENCE_PROFILE=stub]"

_TOKEN_SPLIT = str.maketrans(dict.fromkeys(".,;:!?()[]{}\"'/\\-\n\t", " "))


class StubProvider(BaseProvider):
    """Deterministic, content-derived fake implementations of every capability."""

    def __init__(self, dimensions: int = 1024) -> None:
        super().__init__(provider_name="stub", timeout_seconds=1.0, max_retries=1)
        self.dimensions = dimensions

    # ---- embeddings ----------------------------------------------------

    def _vector(self, text: str) -> list[float]:
        """Hash text tokens into a fixed-width unit vector (a bag-of-words sketch).

        Deterministic and content-dependent: the same text always yields the same
        vector, and texts sharing vocabulary yield similar ones.
        """
        vector = [0.0] * self.dimensions
        tokens = text.lower().translate(_TOKEN_SPLIT).split()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            # Sign from an independent byte so unrelated tokens can cancel.
            vector[index] += 1.0 if digest[4] % 2 == 0 else -1.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            # Empty or punctuation-only text: return a valid unit vector.
            vector[0] = 1.0
            return vector
        return [v / norm for v in vector]

    async def embed(
        self,
        texts: list[str],
        model_name: str | None = None,
    ) -> EmbeddingsResponse:
        started = time.perf_counter()
        return EmbeddingsResponse(
            embeddings=[
                EmbeddingResult(embedding=self._vector(text), index=i)
                for i, text in enumerate(texts)
            ],
            metadata=self._metadata(
                model_name=model_name or "stub-embed",
                started=started,
                tokens=TokenCounts(
                    prompt_tokens=sum(len(t.split()) for t in texts),
                    total_tokens=sum(len(t.split()) for t in texts),
                ),
            ),
        )

    # ---- reranking -----------------------------------------------------

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        model_name: str | None = None,
    ) -> RerankResult:
        """Score by cosine similarity of the stub vectors, then sort."""
        started = time.perf_counter()
        query_vector = self._vector(query)

        scored = sorted(
            (
                ScoredDocument(
                    index=i,
                    text=document,
                    # Map cosine [-1, 1] onto [0, 1] to match reranker conventions.
                    score=(
                        sum(a * b for a, b in zip(query_vector, self._vector(document), strict=True))
                        + 1.0
                    )
                    / 2.0,
                )
                for i, document in enumerate(documents)
            ),
            key=lambda d: -d.score,
        )

        return RerankResult(
            results=scored[: top_k or len(scored)],
            metadata=self._metadata(model_name=model_name or "stub-rerank", started=started),
        )

    # ---- generation ----------------------------------------------------

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        """Emit a clearly-labelled, citation-valid answer.

        It cites `[1]` when evidence was supplied so the citation-validation and
        abstention paths can both be exercised without a real model.
        """
        started = time.perf_counter()
        has_evidence = "BEGIN EVIDENCE [1]" in prompt

        if has_evidence:
            text = (
                f"{STUB_NOTICE} This placeholder answer cites the first supplied "
                f"evidence block [1] so citation validation can run.\n\nSUPPORT: grounded"
            )
        else:
            text = (
                f"{STUB_NOTICE} No evidence was supplied, so no answer can be "
                f"grounded.\n\nSUPPORT: insufficient"
            )

        return GenerationResult(
            text=text,
            metadata=self._metadata(
                model_name=model_name or "stub-llm",
                started=started,
                prompt_version=prompt_version,
                tokens=TokenCounts(
                    prompt_tokens=len(prompt.split()),
                    completion_tokens=len(text.split()),
                    total_tokens=len(prompt.split()) + len(text.split()),
                ),
            ),
        )

    async def vision(
        self,
        prompt: str,
        images: list[ImagePayload],
        system_prompt: str | None = None,
        model_name: str | None = None,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        started = time.perf_counter()
        text = (
            f"{STUB_NOTICE} Received {len(images)} image(s); no vision model was "
            f"called.\n\nSUPPORT: insufficient"
        )
        return GenerationResult(
            text=text,
            metadata=self._metadata(
                model_name=model_name or "stub-vlm",
                started=started,
                prompt_version=prompt_version,
            ),
        )

    # ---- helpers -------------------------------------------------------

    def _metadata(
        self,
        model_name: str,
        started: float,
        prompt_version: str | None = None,
        tokens: TokenCounts | None = None,
    ) -> ModelMetadata:
        return ModelMetadata(
            provider="stub",
            model_name=model_name,
            model_version="stub",
            prompt_version=prompt_version,
            latency_ms=max((time.perf_counter() - started) * 1000.0, 0.001),
            token_counts=tokens or TokenCounts(),
            details={"stub": True, "warning": "no model was called"},
        )
