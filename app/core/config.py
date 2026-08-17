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
    # hosted = free hosted APIs (dev default) | local = vLLM + TEI (production)
    # stub   = explicitly fake gateway for keyless development; never a fallback,
    #          and results under it are not valid evaluation data.
    INFERENCE_PROFILE: Literal["hosted", "local", "stub"] = "hosted"

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
    # Model IDs are pinned explicitly rather than to a floating alias so that
    # `model_version` recorded on every answer identifies a specific model. Hosted
    # providers retire models: `gemini-2.0-flash` and `llama-3.3-70b-versatile` were
    # both already decommissioned and returned 404. Re-check with
    # `GET /v1beta/models` (Gemini) and `GET /openai/v1/models` (Groq) when calls
    # start failing with NOT_FOUND.
    GEMINI_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = "gemini-3.6-flash"
    GEMINI_VISION_MODEL: str = "gemini-3.6-flash"

    GROQ_API_KEY: str = Field(default="")
    GROQ_MODEL: str = "openai/gpt-oss-120b"

    JINA_API_KEY: str = Field(default="")
    JINA_EMBED_MODEL: str = "jina-embeddings-v3"
    JINA_RERANK_MODEL: str = "jina-reranker-v2-base-multilingual"

    # Local Inference Providers (ADR-015, ADR-016 - Production Profile)
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    VLLM_MODEL: str = "Qwen/Qwen2.5-7B-Instruct"
    TEI_EMBED_BASE_URL: str = "http://localhost:8080"
    TEI_RERANK_BASE_URL: str = "http://localhost:8081"

    # Chunking (Stage 3 baseline, ADR-006, ADR-036)
    CHUNKING_STRATEGY: Literal["fixed"] = "fixed"
    CHUNKING_VERSION: str = Field(
        default="fixed-v2",
        description="Participates in deterministic chunk IDs; bump to force re-chunking (ADR-036)",
    )
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64

    # Embedding & Indexing (Stage 3, ADR-009, ADR-036)
    EMBEDDING_VERSION: str = Field(
        default="jina-embeddings-v3",
        description="Participates in deterministic point IDs; bump to force re-embedding",
    )
    EMBEDDING_DIMENSIONS: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_RPM: int = 60
    EMBEDDING_MAX_RETRIES: int = 5

    # Dense Retrieval (Stage 3, ADR-007)
    RETRIEVAL_TOP_K: int = 8
    RETRIEVAL_CANDIDATE_LIMIT: int = 50
    # PROVISIONAL — set by Stage 4 experiment, not by preference.
    # Hand-probed on the HR handbook with jina-embeddings-v3: in-corpus questions
    # scored 0.48-0.59, out-of-corpus questions 0.24-0.29. 0.35 sits in that gap.
    # Five queries is not an experiment; Stage 4 must re-derive this from the
    # golden dataset by measuring abstention accuracy against recall.
    RETRIEVAL_MIN_SCORE: float = 0.35

    # Generation & Grounding (Stage 3, ADR-024, ADR-025, ADR-047)
    GENERATION_MAX_CONTEXT_TOKENS: int = 6000
    GENERATION_TEMPERATURE: float = 0.1
    GENERATION_MAX_TOKENS: int = 1500
    PROMPT_VERSION_ANSWER: str = "answer_v1"
    PROMPT_VERSION_ABSTENTION: str = "abstention_v1"
    PROMPT_VERSION_CITATION: str = "citation_v1"
    ABSTENTION_MIN_EVIDENCE_CHUNKS: int = 1

    # Evaluation Subsystem (Stage 4, ADR-028, ADR-029)
    EVAL_DATASET_VERSION: str = "v1"
    EVAL_RESULTS_DIR: str = "evaluation/results"
    EVAL_CONCURRENCY: int = Field(
        default=2,
        ge=1,
        description="Questions evaluated in parallel; hosted free tiers rate-limit above ~2",
    )

    # LLM-as-judge. The judge deliberately runs on a different provider than the
    # generator: under the hosted profile answers come from Gemini, so the judge
    # runs on Groq. A model scoring its own output has a documented
    # self-preference bias, and the golden dataset is partly LLM-drafted.
    EVAL_JUDGE_ENABLED: bool = True
    EVAL_JUDGE_PROVIDER: str = "groq"
    EVAL_JUDGE_MODEL: str = ""
    EVAL_JUDGE_TEMPERATURE: float = 0.0
    EVAL_JUDGE_MAX_TOKENS: int = 800
    EVAL_JUDGE_SAMPLES: int = Field(
        default=1,
        ge=1,
        description="Repeat judgements per question; >1 measures the variance band",
    )
    EVAL_JUDGE_PARALLEL_PROMPTS: bool = Field(
        default=False,
        description=(
            "Issue the three judge prompts concurrently. Off by default: on a hosted "
            "free tier the token-per-minute window is the constraint, and concurrent "
            "prompts get rate-limited and re-spend their tokens on retry"
        ),
    )
    PROMPT_VERSION_JUDGE_ANSWER: str = "judge_answer_v1"
    PROMPT_VERSION_JUDGE_CITATION: str = "judge_citation_v1"
    PROMPT_VERSION_JUDGE_ABSTENTION: str = "judge_abstention_v1"

    # Regression gate (Task 4.6). Tolerance is absolute, on metrics scaled 0-1.
    EVAL_REGRESSION_TOLERANCE: float = Field(
        default=0.05,
        ge=0.0,
        description="How far a gated metric may fall below the baseline before CI fails",
    )

    # Feature Flags (Master Plan §2)
    ENABLE_RERANKING: bool = False
    ENABLE_VISUAL_RETRIEVAL: bool = False
    ENABLE_QUERY_DECOMPOSITION: bool = False
    ENABLE_MULTI_HOP: bool = False
    ENABLE_SEMANTIC_CACHE: bool = False

    # Security & Rate Limiting (ADR-021, ADR-022)
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_UPLOAD_SIZE_MB: int = 100

    @property
    def effective_embedding_version(self) -> str:
        """The embedding version actually written to chunks and vector points.

        Under the `stub` profile the vectors are not the configured model's output,
        so they are namespaced. Sharing the real model's version would let a later
        switch to a real provider mistake stub vectors for current ones, and would
        make Stage 4 experiment records claim a model that never ran.
        """
        if self.INFERENCE_PROFILE == "stub":
            return f"stub:{self.EMBEDDING_VERSION}"
        return self.EMBEDDING_VERSION

    @field_validator("INFERENCE_PROFILE")
    @classmethod
    def validate_inference_profile(cls, v: str) -> str:
        if v not in ("hosted", "local", "stub"):
            raise ValueError(
                f"Invalid INFERENCE_PROFILE: {v}. Must be 'hosted', 'local', or 'stub'"
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached singleton application settings.

    No module should read os.environ directly; use get_settings() instead.
    """
    return AppSettings()
