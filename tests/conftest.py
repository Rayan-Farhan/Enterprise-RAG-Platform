"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import AppSettings, get_settings
from app.main import app


@pytest.fixture(autouse=True)
def override_settings(monkeypatch: pytest.MonkeyPatch) -> AppSettings:
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
