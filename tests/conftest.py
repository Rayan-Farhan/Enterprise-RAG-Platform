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
    "WORKER_",
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


class InMemoryLockRedis:
    """The slice of Redis the distributed lock uses, shared by every client in a test.

    A plain dict behind a threading lock: chain tests run each job on its own
    thread and event loop, as worker threads do, and all of them must see one
    store - just as every worker sees one Redis. The two Lua scripts are
    interpreted by identity; tests/integration/test_locks_live.py runs the
    real scripts against a real Redis.
    """

    def __init__(self) -> None:
        import threading

        self._mutex = threading.Lock()
        self._values: dict[str, tuple[str, float]] = {}

    def _live(self, name: str) -> str | None:
        import time

        entry = self._values.get(name)
        if entry is None or entry[1] <= time.monotonic():
            self._values.pop(name, None)
            return None
        return entry[0]

    def client(self) -> InMemoryLockClient:
        return InMemoryLockClient(self)


class InMemoryLockClient:
    def __init__(self, store: InMemoryLockRedis) -> None:
        self.store = store

    async def set(self, name: str, value: str, *, nx: bool, px: int) -> bool | None:
        import time

        with self.store._mutex:
            if nx and self.store._live(name) is not None:
                return None
            self.store._values[name] = (value, time.monotonic() + px / 1000)
            return True

    async def get(self, name: str) -> str | None:
        with self.store._mutex:
            return self.store._live(name)

    async def eval(self, script: str, numkeys: int, *keys_and_args: object) -> int:
        import time

        from app.core.locks import EXTEND_SCRIPT, RELEASE_SCRIPT

        key, token, *rest = keys_and_args
        with self.store._mutex:
            if self.store._live(str(key)) != token:
                return 0
            if script == RELEASE_SCRIPT:
                del self.store._values[str(key)]
                return 1
            if script == EXTEND_SCRIPT:
                self.store._values[str(key)] = (str(token), time.monotonic() + int(rest[0]) / 1000)  # type: ignore[call-overload]
                return 1
        raise NotImplementedError("unknown lock script")

    async def aclose(self) -> None:
        return None


@pytest.fixture(autouse=True)
def lock_store(monkeypatch: pytest.MonkeyPatch) -> InMemoryLockRedis:
    """Every lock taken in a test goes to one in-memory store instead of Redis."""
    from app.core import locks

    store = InMemoryLockRedis()
    monkeypatch.setattr(locks, "lock_client_factory", store.client)
    return store
