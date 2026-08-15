"""Unit tests for Model Gateway abstraction and providers."""

import pytest

from app.models.gateway import (
    HostedModelGateway,
    LocalModelGateway,
    ModelGateway,
    get_model_gateway,
)
from app.models.schemas import ImagePayload


@pytest.mark.asyncio
async def test_hosted_gateway_generate() -> None:
    gateway = get_model_gateway(profile="hosted")
    assert isinstance(gateway, HostedModelGateway)
    assert isinstance(gateway, ModelGateway)

    result = await gateway.generate(
        prompt="What is the standard probation period?", prompt_version="v1"
    )
    assert result.text is not None
    assert result.metadata.provider in ("gemini", "groq")
    assert result.metadata.prompt_version == "v1"
    assert result.metadata.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_hosted_gateway_embed() -> None:
    gateway = get_model_gateway(profile="hosted")
    texts = ["HR policy on annual leave", "Remote work policy guidelines"]
    response = await gateway.embed(texts=texts)

    assert len(response.embeddings) == len(texts)
    assert response.metadata.provider == "jina"
    assert len(response.embeddings[0].embedding) > 0


@pytest.mark.asyncio
async def test_hosted_gateway_rerank() -> None:
    gateway = get_model_gateway(profile="hosted")
    query = "annual leave carry forward"
    docs = [
        "Annual leave can be carried forward up to 5 days.",
        "Dress code guidelines for office presence.",
        "Sick leave policy and medical certificate requirements.",
    ]
    rerank_result = await gateway.rerank(query=query, documents=docs, top_k=2)

    assert len(rerank_result.results) <= 2
    assert rerank_result.metadata.provider == "jina"
    assert rerank_result.results[0].score >= 0.0


@pytest.mark.asyncio
async def test_hosted_gateway_vision() -> None:
    gateway = get_model_gateway(profile="hosted")
    images = [ImagePayload(image_bytes=b"fake_png_bytes", mime_type="image/png")]
    result = await gateway.vision(
        prompt="Explain this salary table", images=images, prompt_version="v1"
    )

    assert result.text is not None
    assert result.metadata.provider == "gemini"
    assert result.metadata.prompt_version == "v1"


@pytest.mark.asyncio
async def test_local_gateway_switching() -> None:
    gateway = get_model_gateway(profile="local")
    assert isinstance(gateway, LocalModelGateway)
    assert isinstance(gateway, ModelGateway)

    gen_res = await gateway.generate(prompt="Local test")
    assert "[Local vLLM Stub]" in gen_res.text or gen_res.metadata.provider == "vllm"
    assert gen_res.metadata.provider == "vllm"

    embed_res = await gateway.embed(texts=["Local text"])
    assert embed_res.metadata.provider == "tei"

    rerank_res = await gateway.rerank(query="test", documents=["doc1", "doc2"])
    assert rerank_res.metadata.provider == "tei"
