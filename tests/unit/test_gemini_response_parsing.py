"""Regression tests for Gemini reasoning-model response handling (ADR-046).

Gemini 3.x models always emit "thought" tokens before answering, and
`thinkingConfig.thinkingBudget: 0` is rejected with a 400 on `gemini-3.6-flash`,
so thinking cannot be switched off. Two defects followed from that:

1. The parser read `parts[0].text`. When the first part is a thought part, that
   yields `""` — a silently blank answer rather than an error.
2. Thought tokens are drawn from `maxOutputTokens`. A 16-token budget was spent
   entirely on reasoning, returning `finishReason: MAX_TOKENS` with no content.

Both were caught only by calling the real API; the mocked suite was green.
"""

from __future__ import annotations

import pytest

from app.core.exceptions import ModelProviderException
from app.models.providers.gemini import GeminiProvider


@pytest.fixture
def provider() -> GeminiProvider:
    return GeminiProvider(api_key="test-key")


def response(
    parts: list[dict] | None = None,
    finish_reason: str = "STOP",
    thoughts: int = 0,
    model_version: str = "gemini-3.6-flash",
) -> dict:
    candidate: dict = {"finishReason": finish_reason}
    if parts is not None:
        candidate["content"] = {"parts": parts}
    else:
        candidate["content"] = {}
    return {
        "candidates": [candidate],
        "usageMetadata": {
            "promptTokenCount": 9,
            "candidatesTokenCount": 5,
            "totalTokenCount": 22,
            "thoughtsTokenCount": thoughts,
        },
        "modelVersion": model_version,
    }


class TestThoughtParts:
    def test_thought_parts_are_excluded_from_the_answer(
        self, provider: GeminiProvider
    ) -> None:
        data = response(
            parts=[
                {"text": "Let me reason about the policy...", "thought": True},
                {"text": "The provisional period is three months."},
            ],
            thoughts=13,
        )
        result = provider._to_result(data, "gemini-3.6-flash", "answer_v1", 100.0)

        assert result.text == "The provisional period is three months."
        assert "reason about" not in result.text

    def test_all_text_parts_are_joined_not_just_the_first(
        self, provider: GeminiProvider
    ) -> None:
        """Reading `parts[0]` alone truncated multi-part answers."""
        data = response(parts=[{"text": "Part one. "}, {"text": "Part two."}])
        result = provider._to_result(data, "m", None, 1.0)
        assert result.text == "Part one. Part two."

    def test_thought_token_count_is_recorded(self, provider: GeminiProvider) -> None:
        data = response(parts=[{"text": "answer"}], thoughts=123)
        result = provider._to_result(data, "m", None, 1.0)
        assert result.metadata.details["thought_tokens"] == 123

    def test_model_version_comes_from_the_response(self, provider: GeminiProvider) -> None:
        data = response(parts=[{"text": "answer"}], model_version="gemini-3.6-flash")
        result = provider._to_result(data, "gemini-flash-latest", None, 1.0)
        # The alias was requested; the resolved model is what gets recorded.
        assert result.metadata.model_name == "gemini-flash-latest"
        assert result.metadata.model_version == "gemini-3.6-flash"


class TestEmptyResponsesRaise:
    def test_budget_exhausted_by_thinking_raises_with_guidance(
        self, provider: GeminiProvider
    ) -> None:
        """The exact failure observed live: no content, MAX_TOKENS, 13 thoughts."""
        data = response(parts=None, finish_reason="MAX_TOKENS", thoughts=13)

        with pytest.raises(ModelProviderException) as exc:
            provider._to_result(data, "gemini-3.6-flash", None, 1.0)

        assert "reasoning tokens" in str(exc.value)
        assert exc.value.details["thought_tokens"] == 13

    def test_thought_only_response_raises(self, provider: GeminiProvider) -> None:
        data = response(
            parts=[{"text": "thinking...", "thought": True}],
            finish_reason="MAX_TOKENS",
            thoughts=20,
        )
        with pytest.raises(ModelProviderException, match="reasoning tokens"):
            provider._to_result(data, "m", None, 1.0)

    def test_safety_blocked_response_raises_distinctly(
        self, provider: GeminiProvider
    ) -> None:
        data = response(parts=None, finish_reason="SAFETY")
        with pytest.raises(ModelProviderException, match="safety filter"):
            provider._to_result(data, "m", None, 1.0)

    def test_no_candidates_raises(self, provider: GeminiProvider) -> None:
        with pytest.raises(ModelProviderException, match="No candidates"):
            provider._to_result({"candidates": []}, "m", None, 1.0)

    def test_whitespace_only_answer_is_treated_as_empty(
        self, provider: GeminiProvider
    ) -> None:
        data = response(parts=[{"text": "   \n  "}], finish_reason="STOP")
        with pytest.raises(ModelProviderException):
            provider._to_result(data, "m", None, 1.0)


class TestOutputTokenFloor:
    def test_floor_is_large_enough_for_observed_thinking(self) -> None:
        # A trivial prompt consumed 123 thought tokens; the floor must clear that
        # with room for an actual answer.
        assert GeminiProvider.MIN_OUTPUT_TOKENS >= 512

    @pytest.fixture
    def captured_payloads(self, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
        """Intercept the outgoing HTTP body so the real request can be inspected."""
        payloads: list[dict] = []

        class FakeResponse:
            status_code = 200

            def json(self) -> dict:
                return response(parts=[{"text": "ok"}])

        async def fake_post(self, url, json=None, **kwargs):  # type: ignore[no-untyped-def]
            payloads.append(json)
            return FakeResponse()

        monkeypatch.setattr("httpx.AsyncClient.post", fake_post)
        return payloads

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [(16, 512), (100, 512), (512, 512), (1500, 1500)],
    )
    async def test_output_budget_is_never_below_the_floor(
        self,
        provider: GeminiProvider,
        captured_payloads: list[dict],
        requested: int,
        expected: int,
    ) -> None:
        await provider.generate(prompt="hi", max_tokens=requested)
        assert captured_payloads[0]["generationConfig"]["maxOutputTokens"] == expected

    async def test_vision_applies_the_same_floor(
        self, provider: GeminiProvider, captured_payloads: list[dict]
    ) -> None:
        from app.models.schemas import ImagePayload

        await provider.vision(
            prompt="describe", images=[ImagePayload(image_bytes=b"x")], max_tokens=16
        )
        assert captured_payloads[0]["generationConfig"]["maxOutputTokens"] == 512
