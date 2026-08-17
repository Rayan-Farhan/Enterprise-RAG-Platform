"""Groq provider implementation for fast text generation (ADR-046, ADR-051)."""

import time
from typing import Any

import httpx

from app.core.exceptions import ModelProviderException
from app.models.providers.base import BaseProvider
from app.models.schemas import GenerationResult, ModelMetadata, TokenCounts


class GroqProvider(BaseProvider):
    """Provider communicating with the Groq OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "llama-3.3-70b-versatile",
        timeout_seconds: float = 30.0,
    ) -> None:
        super().__init__(provider_name="groq", timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.default_model = default_model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        """Generate text using Groq LLMs."""
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

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async def _call() -> dict[str, Any]:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(self.base_url, json=payload, headers=headers)
                if res.status_code != 200:
                    raise ModelProviderException(
                        message=f"Groq API returned error {res.status_code}: {res.text}",
                        provider="groq",
                    )
                return res.json()  # type: ignore[no-any-return]

        self.require_credentials(self.api_key, "GROQ_API_KEY")

        data = await self.execute_with_retry(_call)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        choices = data.get("choices", [])
        if not choices:
            raise ModelProviderException("No choices returned from Groq", provider="groq")

        text = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        return GenerationResult(
            text=text,
            metadata=ModelMetadata(
                provider="groq",
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
