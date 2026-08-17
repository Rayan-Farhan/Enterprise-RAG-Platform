"""Unit tests for the Model Gateway abstraction (ADR-046, ADR-051).

These tests deliberately assert on *structure* — which provider class a profile
selects, and that every profile satisfies the `ModelGateway` protocol — not on
model output. Real provider calls live in `tests/e2e/test_gateway_smoke.py` behind
the `live` marker.

An earlier version of this file asserted on the placeholder text that providers
returned when no API key was set (including a literal `"[Local vLLM Stub]"` check).
Those assertions passed against fabricated data and hid the defect that made the
fabrication possible; see `test_provider_credentials.py`.
"""

from __future__ import annotations

import pytest

from app.core.config import AppSettings
from app.core.exceptions import ModelProviderException
from app.models.gateway import (
    HostedModelGateway,
    LocalModelGateway,
    ModelGateway,
    StubModelGateway,
    get_model_gateway,
)
from app.models.schemas import ImagePayload


@pytest.fixture(autouse=True)
def _clear_gateway_cache() -> None:
    get_model_gateway.cache_clear()


class TestProfileSelection:
    """`INFERENCE_PROFILE` must be the only thing that selects providers."""

    @pytest.mark.parametrize(
        ("profile", "expected"),
        [
            ("hosted", HostedModelGateway),
            ("local", LocalModelGateway),
            ("stub", StubModelGateway),
        ],
    )
    def test_profile_selects_its_gateway(self, profile: str, expected: type) -> None:
        gateway = get_model_gateway(profile=profile)
        assert isinstance(gateway, expected)
        assert isinstance(gateway, ModelGateway)

    def test_unknown_profile_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown INFERENCE_PROFILE"):
            get_model_gateway(profile="not-a-profile")

    def test_hosted_gateway_constructs_hosted_providers_only(self) -> None:
        gateway = get_model_gateway(profile="hosted")
        assert isinstance(gateway, HostedModelGateway)
        assert {gateway.gemini.provider_name, gateway.groq.provider_name, gateway.jina.provider_name} == {
            "gemini",
            "groq",
            "jina",
        }

    def test_local_gateway_constructs_local_providers_only(self) -> None:
        gateway = get_model_gateway(profile="local")
        assert isinstance(gateway, LocalModelGateway)
        assert gateway.vllm.provider_name == "vllm"
        assert gateway.tei.provider_name == "tei"

    def test_switching_profile_changes_nothing_else(self) -> None:
        """Application code must not be able to tell which profile is active."""
        hosted = get_model_gateway(profile="hosted")
        get_model_gateway.cache_clear()
        local = get_model_gateway(profile="local")

        capabilities = ("generate", "embed", "rerank", "vision")
        for capability in capabilities:
            assert callable(getattr(hosted, capability))
            assert callable(getattr(local, capability))


class TestHostedGatewayWithoutCredentials:
    """Without keys the gateway must fail, not fabricate."""

    async def test_generate_exhausts_the_fallback_chain_then_raises(self) -> None:
        # Gemini fails first, then the Groq fallback also has no key.
        with pytest.raises(ModelProviderException, match="GROQ_API_KEY"):
            await get_model_gateway(profile="hosted").generate(prompt="probation period?")

    async def test_embed_raises(self) -> None:
        with pytest.raises(ModelProviderException, match="JINA_API_KEY"):
            await get_model_gateway(profile="hosted").embed(texts=["annual leave"])

    async def test_rerank_raises(self) -> None:
        with pytest.raises(ModelProviderException, match="JINA_API_KEY"):
            await get_model_gateway(profile="hosted").rerank(
                query="leave", documents=["a", "b"]
            )

    async def test_vision_raises(self) -> None:
        with pytest.raises(ModelProviderException, match="GEMINI_API_KEY"):
            await get_model_gateway(profile="hosted").vision(
                prompt="explain this table",
                images=[ImagePayload(image_bytes=b"png", mime_type="image/png")],
            )


class TestStubGatewayImplementsEveryCapability:
    """The stub profile is what makes a keyless run possible without lying."""

    @pytest.fixture
    def gateway(self) -> StubModelGateway:
        return StubModelGateway(AppSettings(APP_ENV="testing", EMBEDDING_DIMENSIONS=64))

    async def test_generate(self, gateway: StubModelGateway) -> None:
        result = await gateway.generate(prompt="What is the probation period?", prompt_version="v1")
        assert result.text
        assert result.metadata.provider == "stub"
        assert result.metadata.prompt_version == "v1"
        assert result.metadata.latency_ms > 0

    async def test_embed(self, gateway: StubModelGateway) -> None:
        texts = ["HR policy on annual leave", "Remote work policy guidelines"]
        response = await gateway.embed(texts=texts)

        assert len(response.embeddings) == len(texts)
        assert len(response.embeddings[0].embedding) == 64
        assert response.metadata.provider == "stub"

    async def test_rerank_respects_top_k(self, gateway: StubModelGateway) -> None:
        result = await gateway.rerank(
            query="annual leave carry forward",
            documents=[
                "Annual leave can be carried forward up to 5 days.",
                "Dress code guidelines for office presence.",
                "Sick leave policy and medical certificate requirements.",
            ],
            top_k=2,
        )
        assert len(result.results) == 2
        assert result.metadata.provider == "stub"

    async def test_vision(self, gateway: StubModelGateway) -> None:
        result = await gateway.vision(
            prompt="Explain this salary table",
            images=[ImagePayload(image_bytes=b"png", mime_type="image/png")],
            prompt_version="v1",
        )
        assert result.text
        assert result.metadata.provider == "stub"
        assert result.metadata.prompt_version == "v1"

    async def test_every_response_carries_complete_metadata(
        self, gateway: StubModelGateway
    ) -> None:
        for result in (
            await gateway.generate(prompt="q"),
            await gateway.vision(prompt="q", images=[ImagePayload(image_bytes=b"x")]),
        ):
            assert result.metadata.provider
            assert result.metadata.model_name
            assert result.metadata.latency_ms > 0
            assert result.metadata.token_counts is not None
