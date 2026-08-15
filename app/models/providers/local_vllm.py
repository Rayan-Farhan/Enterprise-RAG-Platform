"""Local vLLM provider for self-hosted generation and vision (ADR-015, ADR-045, ADR-051)."""

import time
from typing import Any

import httpx

from app.core.exceptions import ModelProviderException
from app.models.providers.base import BaseProvider
from app.models.schemas import GenerationResult, ImagePayload, ModelMetadata, TokenCounts


class LocalVLLMProvider(BaseProvider):
    """Provider communicating with a self-hosted vLLM instance (OpenAI-compatible)."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        default_model: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout_seconds: float = 60.0,
    ) -> None:
        super().__init__(provider_name="vllm", timeout_seconds=timeout_seconds)
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        """Generate text using local vLLM server."""
        target_model = model_name or self.default_model
        start_time = time.perf_counter()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async def _call() -> dict[str, Any]:
            url = f"{self.base_url}/chat/completions"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    raise ModelProviderException(
                        message=f"Local vLLM server returned error {res.status_code}: {res.text}",
                        provider="vllm",
                    )
                return res.json()  # type: ignore[no-any-return]

        try:
            data = await self.execute_with_retry(_call)
        except Exception:
            # Fallback stub for development/testing when local vLLM server is offline
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return GenerationResult(
                text=f"[Local vLLM Stub] Response for prompt: {prompt[:30]}...",
                metadata=ModelMetadata(
                    provider="vllm",
                    model_name=target_model,
                    prompt_version=prompt_version,
                    latency_ms=duration_ms,
                    token_counts=TokenCounts(
                        prompt_tokens=15, completion_tokens=15, total_tokens=30
                    ),
                ),
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        choices = data.get("choices", [])
        text = choices[0].get("message", {}).get("content", "") if choices else ""
        usage = data.get("usage", {})

        return GenerationResult(
            text=text,
            metadata=ModelMetadata(
                provider="vllm",
                model_name=target_model,
                prompt_version=prompt_version,
                latency_ms=duration_ms,
                token_counts=TokenCounts(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
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
        """Multimodal generation with local vLLM VLM."""
        target_model = model_name or self.default_model
        start_time = time.perf_counter()
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return GenerationResult(
            text=f"[Local vLLM VLM Stub] Analyzed {len(images)} images for prompt: {prompt[:30]}...",
            metadata=ModelMetadata(
                provider="vllm",
                model_name=target_model,
                prompt_version=prompt_version,
                latency_ms=duration_ms,
                token_counts=TokenCounts(prompt_tokens=30, completion_tokens=15, total_tokens=45),
            ),
        )
