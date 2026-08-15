"""Google Gemini provider implementation for generation and vision (ADR-046, ADR-051)."""

import base64
import time
from typing import Any

import httpx

from app.core.exceptions import ModelProviderException
from app.models.providers.base import BaseProvider
from app.models.schemas import GenerationResult, ImagePayload, ModelMetadata, TokenCounts


class GeminiProvider(BaseProvider):
    """Provider communicating with the Google Gemini API."""

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-2.0-flash",
        vision_model: str = "gemini-2.0-flash",
        timeout_seconds: float = 45.0,
    ) -> None:
        super().__init__(provider_name="gemini", timeout_seconds=timeout_seconds)
        self.api_key = api_key
        self.default_model = default_model
        self.vision_model = vision_model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        prompt_version: str | None = None,
    ) -> GenerationResult:
        """Generate text using Gemini models."""
        target_model = model_name or self.default_model
        start_time = time.perf_counter()

        contents: list[dict[str, Any]] = []
        if system_prompt:
            contents.append(
                {"role": "user", "parts": [{"text": f"System Instructions:\n{system_prompt}"}]}
            )
            contents.append(
                {
                    "role": "model",
                    "parts": [{"text": "Understood. I will strictly follow these instructions."}],
                }
            )
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        async def _call() -> dict[str, Any]:
            url = f"{self.base_url}/{target_model}:generateContent?key={self.api_key}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    raise ModelProviderException(
                        message=f"Gemini API returned error {res.status_code}: {res.text}",
                        provider="gemini",
                        details={"status_code": res.status_code, "response": res.text},
                    )
                return res.json()  # type: ignore[no-any-return]

        if not self.api_key:
            # Mock / stub response when API key is not yet set in dev environment
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return GenerationResult(
                text="[Dev Mock] Gemini response (GEMINI_API_KEY not configured)",
                metadata=ModelMetadata(
                    provider="gemini",
                    model_name=target_model,
                    prompt_version=prompt_version,
                    latency_ms=duration_ms,
                    token_counts=TokenCounts(
                        prompt_tokens=10, completion_tokens=10, total_tokens=20
                    ),
                ),
            )

        data = await self.execute_with_retry(_call)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        candidates = data.get("candidates", [])
        if not candidates:
            raise ModelProviderException("No candidates returned from Gemini", provider="gemini")

        candidate = candidates[0]
        text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
        usage = data.get("usageMetadata", {})

        return GenerationResult(
            text=text,
            metadata=ModelMetadata(
                provider="gemini",
                model_name=target_model,
                prompt_version=prompt_version,
                latency_ms=duration_ms,
                token_counts=TokenCounts(
                    prompt_tokens=usage.get("promptTokenCount", 0),
                    completion_tokens=usage.get("candidatesTokenCount", 0),
                    total_tokens=usage.get("totalTokenCount", 0),
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
        """Multimodal generation with text and images."""
        target_model = model_name or self.vision_model
        start_time = time.perf_counter()

        parts: list[dict[str, Any]] = []
        if system_prompt:
            parts.append({"text": f"System Instructions:\n{system_prompt}\n\n"})
        parts.append({"text": prompt})

        for img in images:
            if img.image_bytes:
                b64_data = base64.b64encode(img.image_bytes).decode("utf-8")
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": img.mime_type,
                            "data": b64_data,
                        }
                    }
                )

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
            },
        }

        async def _call() -> dict[str, Any]:
            url = f"{self.base_url}/{target_model}:generateContent?key={self.api_key}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                if res.status_code != 200:
                    raise ModelProviderException(
                        message=f"Gemini Vision API error {res.status_code}: {res.text}",
                        provider="gemini",
                    )
                return res.json()  # type: ignore[no-any-return]

        if not self.api_key:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            return GenerationResult(
                text="[Dev Mock] Gemini Vision response (GEMINI_API_KEY not configured)",
                metadata=ModelMetadata(
                    provider="gemini",
                    model_name=target_model,
                    prompt_version=prompt_version,
                    latency_ms=duration_ms,
                    token_counts=TokenCounts(
                        prompt_tokens=20, completion_tokens=10, total_tokens=30
                    ),
                ),
            )

        data = await self.execute_with_retry(_call)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        candidates = data.get("candidates", [])
        if not candidates:
            raise ModelProviderException(
                "No candidates returned from Gemini Vision", provider="gemini"
            )

        candidate = candidates[0]
        text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "")
        usage = data.get("usageMetadata", {})

        return GenerationResult(
            text=text,
            metadata=ModelMetadata(
                provider="gemini",
                model_name=target_model,
                prompt_version=prompt_version,
                latency_ms=duration_ms,
                token_counts=TokenCounts(
                    prompt_tokens=usage.get("promptTokenCount", 0),
                    completion_tokens=usage.get("candidatesTokenCount", 0),
                    total_tokens=usage.get("totalTokenCount", 0),
                ),
            ),
        )
