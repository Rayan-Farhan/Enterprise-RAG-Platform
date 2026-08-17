"""Google Gemini provider implementation for generation and vision (ADR-046, ADR-051)."""

import base64
import time
from typing import Any

import httpx

from app.core.exceptions import ModelProviderException
from app.models.providers.base import BaseProvider
from app.models.schemas import GenerationResult, ImagePayload, ModelMetadata, TokenCounts


class GeminiProvider(BaseProvider):
    """Provider communicating with the Google Gemini API.

    Gemini 3.x models are reasoning models: they always spend "thought" tokens
    before answering, and `thinkingBudget: 0` is rejected with a 400 on
    `gemini-3.6-flash`. Two consequences are handled here.

    1. Thought tokens come out of `maxOutputTokens`. A small budget is consumed
       entirely by thinking and the response contains no text at all, with
       `finishReason: MAX_TOKENS` — so a floor is enforced.
    2. Response parts may include thought parts. Reading `parts[0].text` returns
       an empty string in that case, which previously surfaced as a silently blank
       answer rather than an error.
    """

    # Observed: a trivial prompt spent 13-123 thought tokens before emitting any
    # text. This floor keeps short calls (smoke tests, classification in Stage 10)
    # from returning empty results.
    MIN_OUTPUT_TOKENS = 512

    def __init__(
        self,
        api_key: str,
        default_model: str = "gemini-3.6-flash",
        vision_model: str = "gemini-3.6-flash",
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
                "maxOutputTokens": max(max_tokens, self.MIN_OUTPUT_TOKENS),
            },
        }

        async def _call() -> dict[str, Any]:
            url = f"{self.base_url}/{target_model}:generateContent?key={self.api_key}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                self.raise_for_response(res)
                return res.json()  # type: ignore[no-any-return]

        self.require_credentials(self.api_key, "GEMINI_API_KEY")

        data = await self.execute_with_retry(_call)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return self._to_result(
            data=data,
            target_model=target_model,
            prompt_version=prompt_version,
            duration_ms=duration_ms,
        )

    def _to_result(
        self,
        data: dict[str, Any],
        target_model: str,
        prompt_version: str | None,
        duration_ms: float,
    ) -> GenerationResult:
        """Convert a Gemini response into a GenerationResult.

        Concatenates every non-thought text part rather than reading `parts[0]`,
        and treats an empty answer as a failure rather than returning "".
        """
        candidates = data.get("candidates", [])
        if not candidates:
            raise ModelProviderException(
                f"No candidates returned from Gemini. "
                f"promptFeedback={data.get('promptFeedback')}",
                provider="gemini",
                details={"prompt_feedback": data.get("promptFeedback")},
            )

        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        parts = candidate.get("content", {}).get("parts", []) or []
        text = "".join(part.get("text", "") for part in parts if not part.get("thought"))

        usage = data.get("usageMetadata", {})
        thought_tokens = int(usage.get("thoughtsTokenCount", 0) or 0)

        if not text.strip():
            if finish_reason == "MAX_TOKENS":
                raise ModelProviderException(
                    f"Gemini returned no text: the output budget was exhausted, "
                    f"{thought_tokens} of it by reasoning tokens. Raise max_tokens "
                    f"(minimum enforced: {self.MIN_OUTPUT_TOKENS}).",
                    provider="gemini",
                    details={
                        "finish_reason": finish_reason,
                        "thought_tokens": thought_tokens,
                    },
                )
            raise ModelProviderException(
                f"Gemini returned an empty response (finishReason={finish_reason}). "
                f"This usually means the prompt was blocked by a safety filter.",
                provider="gemini",
                details={"finish_reason": finish_reason},
            )

        return GenerationResult(
            text=text,
            finish_reason=finish_reason or "stop",
            metadata=ModelMetadata(
                provider="gemini",
                model_name=target_model,
                # Gemini reports the resolved model, which matters when the request
                # used a floating alias such as `gemini-flash-latest`.
                model_version=data.get("modelVersion"),
                prompt_version=prompt_version,
                latency_ms=duration_ms,
                token_counts=TokenCounts(
                    prompt_tokens=usage.get("promptTokenCount", 0),
                    completion_tokens=usage.get("candidatesTokenCount", 0),
                    total_tokens=usage.get("totalTokenCount", 0),
                ),
                details={"thought_tokens": thought_tokens} if thought_tokens else {},
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
                "maxOutputTokens": max(max_tokens, self.MIN_OUTPUT_TOKENS),
            },
        }

        async def _call() -> dict[str, Any]:
            url = f"{self.base_url}/{target_model}:generateContent?key={self.api_key}"
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                res = await client.post(url, json=payload)
                self.raise_for_response(res)
                return res.json()  # type: ignore[no-any-return]

        self.require_credentials(self.api_key, "GEMINI_API_KEY")

        data = await self.execute_with_retry(_call)
        duration_ms = (time.perf_counter() - start_time) * 1000.0

        return self._to_result(
            data=data,
            target_model=target_model,
            prompt_version=prompt_version,
            duration_ms=duration_ms,
        )
