"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import AppSettings, get_settings
from app.main import app

# Settings keys a developer's local .env may define. They are cleared for the whole
# test session so results never depend on one machine's configuration.
_LEAKY_ENV_PREFIXES = (
    "APP_",
    "DEBUG",
    "SECRET_KEY",
    "INFERENCE_PROFILE",
    "POSTGRES_",
    "REDIS_",
    "RABBITMQ_",
    "MINIO_",
    "QDRANT_",
    "OPENSEARCH_",
    "GEMINI_",
    "GROQ_",
    "JINA_",
    "VLLM_",
    "TEI_",
    "CHUNK_",
    "CHUNKING_",
    "EMBEDDING_",
    "RETRIEVAL_",
    "GENERATION_",
    "PROMPT_VERSION_",
    "ABSTENTION_",
    "ENABLE_",
    "RATE_LIMIT_",
    "MAX_UPLOAD_",
)


@pytest.fixture(autouse=True, scope="session")
def isolate_settings_from_dotenv() -> Iterator[None]:
    """Stop `AppSettings` from reading the developer's `.env` during tests.

    `AppSettings.model_config` pins `env_file=".env"`, so every construction in the
    suite silently inherited local overrides — a developer running with, say,
    `INFERENCE_PROFILE=stub` or a remapped `POSTGRES_PORT` got different assertions
    than CI, where no `.env` exists. Tests must state their own configuration.
    """
    import os

    session_patch = pytest.MonkeyPatch()
    original = AppSettings.model_config.get("env_file")
    AppSettings.model_config["env_file"] = None

    # Real environment variables outrank the env file, so clear those too.
    for key in [k for k in os.environ if k.startswith(_LEAKY_ENV_PREFIXES)]:
        session_patch.delenv(key, raising=False)

    get_settings.cache_clear()
    try:
        yield
    finally:
        AppSettings.model_config["env_file"] = original
        session_patch.undo()
        get_settings.cache_clear()


@pytest.fixture(autouse=True)
def override_settings() -> AppSettings:
    """Provide deterministic test settings."""
    settings = AppSettings(
        APP_ENV="testing",
        DEBUG=True,
        INFERENCE_PROFILE="hosted",
        POSTGRES_DB="test_enterprise_rag",
        GEMINI_API_KEY="",
        GROQ_API_KEY="",
        JINA_API_KEY="",
    )
    get_settings.cache_clear()
    return settings


@pytest.fixture
def client() -> TestClient:
    """FastAPI synchronous test client."""
    return TestClient(app)
