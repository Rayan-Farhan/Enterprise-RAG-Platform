"""Providers must fail loudly without credentials (ADR-046, ADR-051).

Regression suite for a defect found by running against real infrastructure: with
no API key configured, every provider silently returned fabricated output under the
*real* provider's name. `embed()` produced vectors derived from list position
rather than text — so two different documents at the same batch index got identical
vectors — and the pipeline indexed them, retrieved them, and answered from them with
every health signal green. Stage 4's baseline would have been measured on noise.

Fake output is now available only under an explicit `INFERENCE_PROFILE=stub`, and is
labelled `provider="stub"` so it can never be mistaken for a real run.
"""

from __future__ import annotations

import pytest

from app.core.config import AppSettings
from app.core.exceptions import ModelProviderException
from app.models.gateway import (
    HostedModelGateway,
    LocalModelGateway,
    StubModelGateway,
    get_model_gateway,
)
from app.models.providers.gemini import GeminiProvider
from app.models.providers.groq import GroqProvider
from app.models.providers.jina import JinaProvider
from app.models.providers.local_vllm import LocalVLLMProvider
from app.models.providers.stub import STUB_NOTICE, StubProvider
from app.models.schemas import ImagePayload


class TestKeylessProvidersRaise:
    async def test_jina_embed_without_key_raises(self) -> None:
        with pytest.raises(ModelProviderException, match="JINA_API_KEY"):
            await JinaProvider(api_key="").embed(texts=["annual leave policy"])

    async def test_jina_rerank_without_key_raises(self) -> None:
        with pytest.raises(ModelProviderException, match="JINA_API_KEY"):
            await JinaProvider(api_key="").rerank(query="q", documents=["a", "b"])

    async def test_gemini_generate_without_key_raises(self) -> None:
        with pytest.raises(ModelProviderException, match="GEMINI_API_KEY"):
            await GeminiProvider(api_key="").generate(prompt="hello")

    async def test_gemini_vision_without_key_raises(self) -> None:
        with pytest.raises(ModelProviderException, match="GEMINI_API_KEY"):
            await GeminiProvider(api_key="").vision(
                prompt="describe", images=[ImagePayload(image_bytes=b"x")]
            )

    async def test_groq_generate_without_key_raises(self) -> None:
        with pytest.raises(ModelProviderException, match="GROQ_API_KEY"):
            await GroqProvider(api_key="").generate(prompt="hello")

    async def test_local_vllm_vision_is_explicitly_unimplemented(self) -> None:
        """Stage 9 wires this; until then it must not return placeholder text."""
        with pytest.raises(ModelProviderException, match="not implemented"):
            await LocalVLLMProvider(base_url="http://localhost:8000/v1").vision(
                prompt="describe", images=[ImagePayload(image_bytes=b"x")]
            )

    async def test_empty_input_still_short_circuits_before_the_key_check(self) -> None:
        """Embedding nothing is a legitimate no-op and must not require a key."""
        result = await JinaProvider(api_key="").embed(texts=[])
        assert result.embeddings == []

        reranked = await JinaProvider(api_key="").rerank(query="q", documents=[])
        assert reranked.results == []


class TestNoSilentFallbacksRemain:
    @pytest.mark.parametrize(
        "module",
        ["gemini", "groq", "jina", "local_tei", "local_vllm"],
    )
    def test_provider_source_contains_no_mock_fallback(self, module: str) -> None:
        """Guards against the pattern being reintroduced."""
        from pathlib import Path

        source = Path(f"app/models/providers/{module}.py").read_text(encoding="utf-8").lower()
        for banned in ("dev mock", "mock_embeddings", "mock_results", "fallback stub"):
            assert banned not in source, f"{module}.py reintroduced a silent fallback: {banned}"


class TestStubProfileIsExplicitAndLabelled:
    def test_stub_profile_builds_the_stub_gateway(self) -> None:
        get_model_gateway.cache_clear()
        assert isinstance(get_model_gateway(profile="stub"), StubModelGateway)

    def test_other_profiles_are_unaffected(self) -> None:
        get_model_gateway.cache_clear()
        assert isinstance(get_model_gateway(profile="hosted"), HostedModelGateway)
        get_model_gateway.cache_clear()
        assert isinstance(get_model_gateway(profile="local"), LocalModelGateway)

    def test_unknown_profile_still_fails(self) -> None:
        get_model_gateway.cache_clear()
        with pytest.raises(ValueError, match="Must be 'hosted', 'local', or 'stub'"):
            get_model_gateway(profile="pretend")

    def test_stub_is_not_reachable_by_accident(self) -> None:
        """A missing key must not silently select the stub profile."""
        settings = AppSettings(APP_ENV="testing", GEMINI_API_KEY="", JINA_API_KEY="")
        assert settings.INFERENCE_PROFILE == "hosted"

    def test_config_rejects_an_invalid_profile(self) -> None:
        with pytest.raises(ValueError, match="INFERENCE_PROFILE"):
            AppSettings(APP_ENV="testing", INFERENCE_PROFILE="mock")  # type: ignore[arg-type]

    async def test_stub_metadata_is_labelled_stub(self) -> None:
        gateway = StubModelGateway(AppSettings(APP_ENV="testing", EMBEDDING_DIMENSIONS=64))

        embedded = await gateway.embed(texts=["annual leave"])
        assert embedded.metadata.provider == "stub"
        assert embedded.metadata.details.get("stub") is True

        generated = await gateway.generate(prompt="anything")
        assert generated.metadata.provider == "stub"
        assert STUB_NOTICE in generated.text


class TestStubEmbeddingsAreContentDerived:
    """The original mock keyed vectors off list position; that must not recur."""

    @pytest.fixture
    def provider(self) -> StubProvider:
        return StubProvider(dimensions=128)

    async def test_same_text_gives_same_vector(self, provider: StubProvider) -> None:
        first = await provider.embed(["annual leave policy"])
        second = await provider.embed(["annual leave policy"])
        assert first.embeddings[0].embedding == second.embeddings[0].embedding

    async def test_different_texts_give_different_vectors(self, provider: StubProvider) -> None:
        result = await provider.embed(["annual leave policy", "fire safety procedure"])
        assert result.embeddings[0].embedding != result.embeddings[1].embedding

    async def test_position_does_not_determine_the_vector(self, provider: StubProvider) -> None:
        """The exact failure mode of the removed mock."""
        a = await provider.embed(["alpha text", "beta text"])
        b = await provider.embed(["beta text", "alpha text"])
        assert a.embeddings[0].embedding == b.embeddings[1].embedding
        assert a.embeddings[1].embedding == b.embeddings[0].embedding

    async def test_vectors_are_unit_length(self, provider: StubProvider) -> None:
        result = await provider.embed(["annual leave", "", "!!!"])
        for item in result.embeddings:
            norm = sum(v * v for v in item.embedding) ** 0.5
            assert norm == pytest.approx(1.0, abs=1e-6)

    async def test_dimensions_follow_configuration(self) -> None:
        result = await StubProvider(dimensions=256).embed(["text"])
        assert len(result.embeddings[0].embedding) == 256

    async def test_similar_text_scores_above_unrelated_text(
        self, provider: StubProvider
    ) -> None:
        """Retrieval plumbing should behave sanely under the stub profile."""
        ranked = await provider.rerank(
            query="annual leave entitlement days",
            documents=[
                "The cafeteria serves lunch between noon and two.",
                "Employees receive 21 annual leave days each year.",
            ],
        )
        assert ranked.results[0].index == 1

    async def test_rerank_scores_are_normalised(self, provider: StubProvider) -> None:
        ranked = await provider.rerank(query="q", documents=["a", "b", "c"])
        assert all(0.0 <= item.score <= 1.0 for item in ranked.results)


class TestStubGenerationSupportsBothPaths:
    async def test_cites_evidence_when_evidence_is_present(self) -> None:
        provider = StubProvider()
        result = await provider.generate(prompt="--- BEGIN EVIDENCE [1] ---\ntext\n")
        assert "[1]" in result.text
        assert "SUPPORT: grounded" in result.text

    async def test_declares_insufficient_without_evidence(self) -> None:
        result = await StubProvider().generate(prompt="just a question")
        assert "SUPPORT: insufficient" in result.text
