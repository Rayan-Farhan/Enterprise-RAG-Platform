"""Live model gateway smoke test (Task 0.6 done-when).

One real call per capability against a free hosted provider, asserting a non-empty
result plus complete metadata. Each test skips when its provider key is absent, so
the suite stays green locally and in CI without secrets — but it is a real network
test when the keys are present, which is the point.

Run explicitly:  pytest -m live tests/e2e/test_gateway_smoke.py
"""

from __future__ import annotations

import base64

import pytest

from app.core.config import AppSettings
from app.models.gateway import (
    HostedModelGateway,
    LocalModelGateway,
    get_model_gateway,
)
from app.models.schemas import ImagePayload, ModelMetadata

pytestmark = pytest.mark.live

# A 1x1 red PNG — the smallest valid image that exercises the vision path.
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
)


@pytest.fixture(scope="module")
def live_settings() -> AppSettings:
    """Settings read from the real environment and `.env`.

    The session-wide fixture in `tests/conftest.py` detaches `env_file` so unit and
    integration tests cannot inherit a developer's local configuration. These tests
    need the opposite: they are the one suite that must see real credentials, so the
    env file is re-attached explicitly here.
    """
    return AppSettings(_env_file=".env")  # type: ignore[call-arg]


def _require(key: str, name: str) -> None:
    if not key:
        pytest.skip(f"{name} not configured; skipping live provider call")


def assert_complete_metadata(metadata: ModelMetadata) -> None:
    """Every gateway call must return the full metadata contract (Task 0.6)."""
    assert metadata.provider
    assert metadata.model_name
    assert metadata.latency_ms > 0
    assert metadata.token_counts is not None


class TestHostedProfileSmoke:
    async def test_generate_returns_text_and_metadata(self, live_settings: AppSettings) -> None:
        _require(live_settings.GEMINI_API_KEY or live_settings.GROQ_API_KEY, "GEMINI/GROQ_API_KEY")
        gateway = HostedModelGateway(live_settings)

        result = await gateway.generate(
            prompt="Reply with exactly the word: ACKNOWLEDGED",
            max_tokens=16,
            prompt_version="smoke_v1",
        )

        assert result.text.strip()
        assert_complete_metadata(result.metadata)
        assert result.metadata.prompt_version == "smoke_v1"

    async def test_embed_returns_vectors_and_metadata(self, live_settings: AppSettings) -> None:
        _require(live_settings.JINA_API_KEY, "JINA_API_KEY")
        gateway = HostedModelGateway(live_settings)

        result = await gateway.embed(texts=["annual leave policy", "sick leave policy"])

        assert len(result.embeddings) == 2
        assert all(item.embedding for item in result.embeddings)
        # All vectors must share a dimensionality or the Qdrant collection is wrong.
        assert len({len(item.embedding) for item in result.embeddings}) == 1
        assert_complete_metadata(result.metadata)

    async def test_rerank_returns_scores_and_metadata(self, live_settings: AppSettings) -> None:
        _require(live_settings.JINA_API_KEY, "JINA_API_KEY")
        gateway = HostedModelGateway(live_settings)

        result = await gateway.rerank(
            query="How many annual leave days do employees receive?",
            documents=[
                "The office cafeteria serves lunch from 12:00 to 14:00.",
                "Full-time employees are entitled to 21 days of paid annual leave.",
                "Parking permits are issued by the facilities team.",
            ],
        )

        assert result.results
        assert_complete_metadata(result.metadata)
        # The genuinely relevant document should outrank the irrelevant ones.
        assert result.results[0].index == 1

    async def test_vision_returns_text_and_metadata(self, live_settings: AppSettings) -> None:
        _require(live_settings.GEMINI_API_KEY, "GEMINI_API_KEY")
        gateway = HostedModelGateway(live_settings)

        result = await gateway.vision(
            prompt="What single colour dominates this image? Answer in one word.",
            images=[ImagePayload(image_bytes=_TINY_PNG, mime_type="image/png")],
            max_tokens=16,
        )

        assert result.text.strip()
        assert_complete_metadata(result.metadata)


class TestProfileSwitching:
    """The `INFERENCE_PROFILE` switch must be the only thing selecting a provider."""

    def test_hosted_profile_builds_hosted_gateway(self) -> None:
        get_model_gateway.cache_clear()
        gateway = get_model_gateway(profile="hosted")
        assert isinstance(gateway, HostedModelGateway)

    def test_local_profile_builds_local_gateway(self) -> None:
        get_model_gateway.cache_clear()
        gateway = get_model_gateway(profile="local")
        assert isinstance(gateway, LocalModelGateway)

    def test_unknown_profile_fails_loudly(self) -> None:
        get_model_gateway.cache_clear()
        with pytest.raises(ValueError, match="Unknown INFERENCE_PROFILE"):
            get_model_gateway(profile="somewhere-else")
