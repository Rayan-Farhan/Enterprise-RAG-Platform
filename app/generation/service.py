"""Grounded answer generation orchestration (Tasks 3.5/3.6, master §21-22)."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.generation.citation import (
    CitationValidator,
    SupportState,
    get_citation_validator,
)
from app.generation.context import AssembledContext, ContextAssembler, get_context_assembler
from app.models.gateway import ModelGateway, get_model_gateway
from app.retrieval.dense import DenseRetriever, get_dense_retriever
from app.retrieval.expansion import ParentExpander, get_parent_expander
from app.retrieval.schemas import Citation, RetrievalFilters, RetrievedChunk

logger = get_logger("app.generation.service")


@dataclass
class AnswerResult:
    """A complete grounded answer with the full provenance of its production.

    Every field below `retrieved_chunk_ids` exists to satisfy the Stage 3 exit
    gate and to feed Stage 4's experiment tracking without a retrofit.
    """

    query: str
    answer: str
    support: SupportState
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    retrieved_chunk_ids: list[uuid.UUID] = field(default_factory=list)
    # The full ranked hits and the subset that survived assembly. Stage 4's
    # retrieval metrics are computed at element granularity, and context recall
    # is specifically the difference between these two lists — recording only the
    # chunk IDs would make evidence dropped by the token budget look identical to
    # evidence never retrieved.
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    context_chunk_ids: list[uuid.UUID] = field(default_factory=list)
    fabricated_markers: list[str] = field(default_factory=list)
    rejected: bool = False
    rejection_reason: str | None = None

    # Reproducibility metadata
    provider: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    prompt_versions: dict[str, str] = field(default_factory=dict)
    prompt_hashes: dict[str, str] = field(default_factory=dict)
    retrieval_config: dict[str, Any] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    evidence_tokens: int = 0
    evidence_block: str = ""
    degradations: list[str] = field(default_factory=list)


class GenerationService:
    """Runs the thin end-to-end RAG path: retrieve, assemble, generate, validate."""

    def __init__(
        self,
        retriever: DenseRetriever | None = None,
        assembler: ContextAssembler | None = None,
        validator: CitationValidator | None = None,
        gateway: ModelGateway | None = None,
        expander: ParentExpander | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or get_dense_retriever()
        self.assembler = assembler or get_context_assembler()
        self.validator = validator or get_citation_validator()
        self.gateway = gateway or get_model_gateway()
        self.expander = expander or get_parent_expander()

    async def answer(
        self,
        query: str,
        session: AsyncSession,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> AnswerResult:
        """Produce a grounded, cited answer — or abstain when evidence is absent."""
        started = time.perf_counter()

        retrieval = await self.retriever.retrieve(
            query=query,
            session=session,
            top_k=top_k,
            filters=filters,
        )

        if len(retrieval.chunks) < self.settings.ABSTENTION_MIN_EVIDENCE_CHUNKS:
            logger.info(
                "abstaining_insufficient_evidence",
                retrieved=len(retrieval.chunks),
                required=self.settings.ABSTENTION_MIN_EVIDENCE_CHUNKS,
            )
            return await self._abstain(
                query=query,
                retrieval_config=retrieval.retrieval_config,
                retrieval_latency_ms=retrieval.latency_ms,
                started=started,
            )

        # Parent-child expansion sits between retrieval and assembly (Task 5.2):
        # retrieval matched small leaves for precision, generation reads whole
        # sections for coherence. A no-op unless ENABLE_PARENT_EXPANSION is on
        # and the corpus was chunked by a strategy that emits a hierarchy.
        expansion = await self.expander.expand(retrieval.chunks, session=session)
        context = self.assembler.assemble(query=query, chunks=expansion.chunks)

        gen_started = time.perf_counter()
        generation = await self.gateway.generate(
            prompt=context.user_message(),
            system_prompt=context.system_prompt,
            temperature=self.settings.GENERATION_TEMPERATURE,
            max_tokens=self.settings.GENERATION_MAX_TOKENS,
            prompt_version=context.prompt_versions.get("answer"),
        )
        gen_latency = (time.perf_counter() - gen_started) * 1000

        validated = self.validator.validate(generation.text, context)

        result = AnswerResult(
            query=query,
            answer=validated.answer,
            support=validated.support,
            citations=validated.citations,
            abstained=validated.support is SupportState.INSUFFICIENT,
            retrieved_chunk_ids=[c.chunk_id for c in retrieval.chunks],
            retrieved_chunks=list(retrieval.chunks),
            context_chunk_ids=[c.chunk_id for c in context.included_chunks],
            fabricated_markers=validated.fabricated_markers,
            rejected=validated.rejected,
            rejection_reason=validated.rejection_reason,
            provider=generation.metadata.provider,
            model_name=generation.metadata.model_name,
            model_version=generation.metadata.model_version,
            prompt_versions=context.prompt_versions,
            prompt_hashes=context.prompt_hashes,
            retrieval_config=retrieval.retrieval_config,
            token_counts=generation.metadata.token_counts.model_dump(),
            retrieval_latency_ms=retrieval.latency_ms,
            generation_latency_ms=gen_latency,
            evidence_tokens=context.evidence_tokens,
            # Retained verbatim so Stage 4's judge scores the answer against the
            # exact text the model saw, not a reconstruction of it.
            evidence_block=context.evidence_block,
        )
        result.total_latency_ms = (time.perf_counter() - started) * 1000

        if validated.rejected:
            # Rejection is not a silent downgrade: the answer text is replaced so
            # an unsupported claim cannot reach the user through this path.
            result.answer = (
                "I could not produce an answer that is verifiably supported by the "
                "HR knowledge base. Please rephrase the question or contact HR directly."
            )
            result.support = SupportState.INSUFFICIENT
            result.abstained = True
            result.citations = []
            result.degradations.append("answer_rejected_citation_validation")

        logger.info(
            "answer_generated",
            support=str(result.support),
            citations=len(result.citations),
            fabricated=len(result.fabricated_markers),
            rejected=result.rejected,
            total_latency_ms=round(result.total_latency_ms, 2),
        )
        return result

    async def stream_answer(
        self,
        query: str,
        session: AsyncSession,
        top_k: int | None = None,
        filters: RetrievalFilters | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield ``(event_name, payload)`` pairs for SSE streaming.

        Citations are emitted only after validation, so a fabricated citation can
        never reach the client mid-stream. Consequently the answer text is streamed
        as one chunk rather than token-by-token: real token streaming arrives with
        the provider streaming APIs in a later stage, and streaming unvalidated
        text now would break the Stage 3 exit gate's citation guarantee.
        """
        result = await self.answer(query=query, session=session, top_k=top_k, filters=filters)

        yield (
            "metadata",
            {
                "support": str(result.support),
                "abstained": result.abstained,
                "model_name": result.model_name,
                "provider": result.provider,
                "prompt_versions": result.prompt_versions,
                "retrieval_config": result.retrieval_config,
                "retrieved_chunk_ids": [str(c) for c in result.retrieved_chunk_ids],
            },
        )
        yield ("token", {"text": result.answer})
        yield (
            "citations",
            {"citations": [c.model_dump(mode="json") for c in result.citations]},
        )
        yield (
            "done",
            {
                "support": str(result.support),
                "abstained": result.abstained,
                "citation_count": len(result.citations),
                "token_counts": result.token_counts,
                "retrieval_latency_ms": round(result.retrieval_latency_ms, 2),
                "generation_latency_ms": round(result.generation_latency_ms, 2),
                "total_latency_ms": round(result.total_latency_ms, 2),
                "degradations": result.degradations,
            },
        )

    async def _abstain(
        self,
        query: str,
        retrieval_config: dict[str, Any],
        retrieval_latency_ms: float,
        started: float,
    ) -> AnswerResult:
        """Generate an explicit refusal rather than inventing an answer (master §21)."""
        context = self.assembler.assemble_abstention(query)

        gen_started = time.perf_counter()
        provider: str | None
        model_name: str | None
        model_version: str | None
        try:
            generation = await self.gateway.generate(
                prompt=context.user_message(),
                system_prompt=context.system_prompt,
                temperature=self.settings.GENERATION_TEMPERATURE,
                max_tokens=300,
                prompt_version=context.prompt_versions.get("abstention"),
            )
            answer = generation.text.strip()
            provider = generation.metadata.provider
            model_name = generation.metadata.model_name
            model_version = generation.metadata.model_version
            token_counts = generation.metadata.token_counts.model_dump()
            degradations: list[str] = []
        except Exception as exc:  # noqa: BLE001 - abstention must never fail the request
            logger.warning("abstention_generation_failed_using_static", error=str(exc))
            answer = (
                "The HR knowledge base does not contain information that answers this "
                "question. Please try rephrasing with the specific policy name, or "
                "contact HR directly."
            )
            provider = model_name = model_version = None
            token_counts = {}
            degradations = ["abstention_static_fallback"]

        # Strip any SUPPORT control line the abstention prompt asked for.
        validated = self.validator.validate(answer, context)

        result = AnswerResult(
            query=query,
            answer=validated.answer or answer,
            support=SupportState.INSUFFICIENT,
            citations=[],
            abstained=True,
            provider=provider,
            model_name=model_name,
            model_version=model_version,
            prompt_versions=context.prompt_versions,
            prompt_hashes=context.prompt_hashes,
            retrieval_config=retrieval_config,
            token_counts=token_counts,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=(time.perf_counter() - gen_started) * 1000,
            degradations=degradations,
        )
        result.total_latency_ms = (time.perf_counter() - started) * 1000
        return result


def build_context_for_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    settings: AppSettings | None = None,
) -> AssembledContext:
    """Assemble a context outside the service — used by prompt snapshot tests."""
    return ContextAssembler(settings=settings).assemble(query=query, chunks=chunks)


_generation_service: GenerationService | None = None


def get_generation_service() -> GenerationService:
    """Return the singleton GenerationService."""
    global _generation_service
    if _generation_service is None:
        _generation_service = GenerationService()
    return _generation_service
