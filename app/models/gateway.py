"""Model Gateway interface and factory implementation (ADR-046, ADR-051)."""

from functools import lru_cache
from typing import Protocol, runtime_checkable

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.models.providers.gemini import GeminiProvider
from app.models.providers.groq import GroqProvider
from app.models.providers.jina import JinaProvider
from app.models.providers.local_tei import LocalTEIProvider
from app.models.providers.local_vllm import LocalVLLMProvider
from app.models.providers.stub import StubProvider
from app.models.schemas import (
    EmbeddingsResponse,
    GenerationResult,
    ImagePayload,
    RerankResult,
)


@runtime_checkable
class ModelGateway(Protocol):
    """Unified provider-agnostic model gateway interface (ADR-046)."""

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        ...

    async def embed(
        self,
        texts: list[str],
        model_name: str | None = None,
    ) -> EmbeddingsResponse:
        """Generate dense vector embeddings for texts."""
        ...

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        model_name: str | None = None,
    ) -> RerankResult:
        """Rerank candidate documents against a query."""
        ...

    async def vision(
        self,
        prompt: str,
        images: list[ImagePayload],
        system_prompt: str | None = None,
        model_name: str | None = None,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        """Perform multimodal visual reasoning on prompt and images."""
        ...


class HostedModelGateway:
    """Hosted profile implementation using free API key platforms (ADR-051)."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.logger = get_logger("app.models.gateway.hosted")
        self.gemini = GeminiProvider(
            api_key=settings.GEMINI_API_KEY,
            default_model=settings.GEMINI_MODEL,
            vision_model=settings.GEMINI_VISION_MODEL,
        )
        self.groq = GroqProvider(
            api_key=settings.GROQ_API_KEY,
            default_model=settings.GROQ_MODEL,
        )
        self.jina = JinaProvider(
            api_key=settings.JINA_API_KEY,
            default_embed_model=settings.JINA_EMBED_MODEL,
            default_rerank_model=settings.JINA_RERANK_MODEL,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        """Generate text via Gemini (primary) with Groq fallback."""
        try:
            return await self.gemini.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_version=prompt_version,
            )
        except Exception as exc:
            self.logger.warning(
                "Primary Gemini generation failed, falling back to Groq", error=str(exc)
            )
            return await self.groq.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                prompt_version=prompt_version,
            )

    async def embed(
        self,
        texts: list[str],
        model_name: str | None = None,
    ) -> EmbeddingsResponse:
        """Embed texts via Jina AI."""
        return await self.jina.embed(texts=texts, model_name=model_name)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        model_name: str | None = None,
    ) -> RerankResult:
        """Rerank documents via Jina AI."""
        return await self.jina.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
            model_name=model_name,
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
        """Process visual reasoning via Gemini Vision."""
        return await self.gemini.vision(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
        )


class LocalModelGateway:
    """Local profile implementation using self-hosted vLLM + TEI (ADR-015, ADR-016, ADR-045)."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.logger = get_logger("app.models.gateway.local")
        self.vllm = LocalVLLMProvider(
            base_url=settings.VLLM_BASE_URL,
            default_model=settings.VLLM_MODEL,
        )
        self.tei = LocalTEIProvider(
            embed_base_url=settings.TEI_EMBED_BASE_URL,
            rerank_base_url=settings.TEI_RERANK_BASE_URL,
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        return await self.vllm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
        )

    async def embed(
        self,
        texts: list[str],
        model_name: str | None = None,
    ) -> EmbeddingsResponse:
        return await self.tei.embed(texts=texts, model_name=model_name)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        model_name: str | None = None,
    ) -> RerankResult:
        return await self.tei.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
            model_name=model_name,
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
        return await self.vllm.vision(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
        )


class StubModelGateway:
    """Explicitly fake gateway for keyless development (INFERENCE_PROFILE=stub).

    Selected only by explicit configuration, never as a fallback. Every response is
    labelled `provider="stub"`, so results produced under this profile can never be
    mistaken for a real run. See `app/models/providers/stub.py`.
    """

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.logger = get_logger("app.models.gateway.stub")
        self.stub = StubProvider(dimensions=settings.EMBEDDING_DIMENSIONS)
        self.logger.warning(
            "stub_inference_profile_active",
            detail="No model is being called. Results are not valid evaluation data.",
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        return await self.stub.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
        )

    async def embed(
        self,
        texts: list[str],
        model_name: str | None = None,
    ) -> EmbeddingsResponse:
        return await self.stub.embed(texts=texts, model_name=model_name)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int | None = None,
        model_name: str | None = None,
    ) -> RerankResult:
        return await self.stub.rerank(
            query=query, documents=documents, top_k=top_k, model_name=model_name
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
        return await self.stub.vision(
            prompt=prompt,
            images=images,
            system_prompt=system_prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            prompt_version=prompt_version,
        )


@lru_cache(maxsize=3)
def get_model_gateway(profile: str | None = None) -> ModelGateway:
    """Factory function returning the active ModelGateway instance.

    Controlled by INFERENCE_PROFILE ('hosted', 'local', or 'stub'). This is the only
    place that decides which provider classes are constructed — application code
    never learns which profile is active (ADR-046, ADR-051).
    """
    settings = get_settings()
    active_profile = profile or settings.INFERENCE_PROFILE

    if active_profile == "hosted":
        return HostedModelGateway(settings)
    elif active_profile == "local":
        return LocalModelGateway(settings)
    elif active_profile == "stub":
        return StubModelGateway(settings)
    else:
        raise ValueError(
            f"Unknown INFERENCE_PROFILE: {active_profile}. "
            f"Must be 'hosted', 'local', or 'stub'."
        )
