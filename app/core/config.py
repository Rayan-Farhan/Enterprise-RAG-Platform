"""Application configuration based on 12-factor principles (ADR-033, ADR-051, ADR-052)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Core application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # General
    APP_NAME: str = "Enterprise Multimodal RAG Platform"
    APP_VERSION: str = "0.1.0"
    APP_ENV: Literal["development", "testing", "staging", "production"] = "development"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "dev-insecure-secret-key-change-in-production"

    # Inference Profile (ADR-051)
    INFERENCE_PROFILE: Literal["hosted", "local"] = "hosted"

    # PostgreSQL Database (ADR-002)
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "enterprise_rag"
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 10

    @property
    def sync_database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Redis Cache & Locks (ADR-020, ADR-039)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str | None = None
    REDIS_DB: int = 0

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # RabbitMQ Broker (ADR-017)
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"

    @property
    def rabbitmq_url(self) -> str:
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}//"

    # MinIO / Object Storage (ADR-003)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_SECURE: bool = False
    MINIO_BUCKET_NAME: str = "enterprise-rag-documents"

    # Qdrant Vector Engine (ADR-007, ADR-009, ADR-010)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "enterprise_rag_chunks"

    # OpenSearch Lexical & Neural Sparse Engine (ADR-007, ADR-008)
    OPENSEARCH_HOST: str = "localhost"
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_USER: str = "admin"
    OPENSEARCH_PASSWORD: str = "admin"
    OPENSEARCH_USE_SSL: bool = False
    OPENSEARCH_VERIFY_CERTS: bool = False
    OPENSEARCH_INDEX_NAME: str = "enterprise_rag_chunks"

    # Hosted Providers (ADR-051 - Free Tier Development)
    GEMINI_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_VISION_MODEL: str = "gemini-2.0-flash"

    GROQ_API_KEY: str = Field(default="")
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    JINA_API_KEY: str = Field(default="")
    JINA_EMBED_MODEL: str = "jina-embeddings-v3"
    JINA_RERANK_MODEL: str = "jina-reranker-v2-base-multilingual"

    # Local Inference Providers (ADR-015, ADR-016 - Production Profile)
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    TEI_EMBED_BASE_URL: str = "http://localhost:8080"
    TEI_RERANK_BASE_URL: str = "http://localhost:8081"

    # Feature Flags (Master Plan §2)
    ENABLE_RERANKING: bool = False
    ENABLE_VISUAL_RETRIEVAL: bool = False
    ENABLE_QUERY_DECOMPOSITION: bool = False
    ENABLE_MULTI_HOP: bool = False
    ENABLE_SEMANTIC_CACHE: bool = False

    # Security & Rate Limiting (ADR-021, ADR-022)
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_UPLOAD_SIZE_MB: int = 100

    @field_validator("INFERENCE_PROFILE")
    @classmethod
    def validate_inference_profile(cls, v: str) -> str:
        if v not in ("hosted", "local"):
            raise ValueError(f"Invalid INFERENCE_PROFILE: {v}. Must be 'hosted' or 'local'")
        return v


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached singleton application settings.

    No module should read os.environ directly; use get_settings() instead.
    """
    return AppSettings()
