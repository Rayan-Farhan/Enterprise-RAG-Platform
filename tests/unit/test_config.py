"""Unit tests for 12-factor configuration."""

import pytest
from pydantic import ValidationError

from app.core.config import AppSettings


def test_default_settings() -> None:
    settings = AppSettings()
    assert settings.APP_NAME == "Enterprise Multimodal RAG Platform"
    assert settings.INFERENCE_PROFILE in ("hosted", "local")
    assert "postgresql://" in settings.sync_database_url
    assert "postgresql+asyncpg://" in settings.async_database_url
    assert "redis://" in settings.redis_url
    assert "amqp://" in settings.rabbitmq_url


def test_invalid_inference_profile() -> None:
    with pytest.raises(ValidationError):
        AppSettings(INFERENCE_PROFILE="unsupported_profile")  # type: ignore[arg-type]


def test_custom_settings_override() -> None:
    settings = AppSettings(
        APP_ENV="production",
        POSTGRES_HOST="db.internal",
        POSTGRES_PORT=5433,
        INFERENCE_PROFILE="local",
    )
    assert settings.APP_ENV == "production"
    assert settings.POSTGRES_HOST == "db.internal"
    assert settings.POSTGRES_PORT == 5433
    assert settings.INFERENCE_PROFILE == "local"
    assert "db.internal:5433" in settings.sync_database_url
