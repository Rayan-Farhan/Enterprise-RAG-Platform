"""Application configuration based on 12-factor principles (ADR-033, ADR-051, ADR-052)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
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

    # Chunking (ADR-006, ADR-036; strategies from Task 5.1)
    CHUNKING_STRATEGY: Literal[
        "fixed",
        "structure_aware",
        "hierarchical",
        "contextual",
        "hierarchical_contextual",
    ] = "contextual"
    CHUNKING_VERSION: str = Field(
        default="contextual-s256-o32",
        description=(
            "Participates in deterministic chunk IDs; bump to force re-chunking "
            "(ADR-036). Must start with the strategy name so two strategies can "
            "never collide on a chunk identity."
        ),
    )
    # Locked by the Stage 5 sweep (Task 5.3, ADR-006), not by preference:
    # contextual at 256/32 beat the fixed baseline by +0.134 recall@5
    # (95% CI [+0.031, +0.232]) and passed the regression gate on Layer 1.
    # Sizes from 128 to 512 did not separate at 100 questions; 256 was chosen on
    # the best point estimate and the lowest evidence-token cost per query.
    CHUNK_SIZE_TOKENS: int = 256
    CHUNK_OVERLAP_TOKENS: int = 32

    # Embedding & Indexing (Stage 3, ADR-009, ADR-036)
    EMBEDDING_VERSION: str = Field(
        default="jina-embeddings-v3",
        description="Participates in deterministic point IDs; bump to force re-embedding",
    )
    EMBEDDING_DIMENSIONS: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_MAX_RPM: int = 60
    EMBEDDING_MAX_RETRIES: int = 5

    # Retrieval channel selection (Stage 6, ADR-007/008). Task 6.4 adds "hybrid"
    # (RRF fusion); until then one channel serves generation at a time, which is
    # also what the Task 6.7 single-channel experiments need.
    # Task 6.7 set the production default to neural sparse. Dense alone was the
    # worst channel on both dev and validation splits (recall@10 0.54/0.56 vs
    # 0.81/0.82). Sparse, BM25, hybrid and the reranked variants tied on the
    # held-out split; sparse was chosen for quality per second - 0.26 s p50 and
    # no per-query model cost, against ~1.2-2.3 s for the reranked paths. It has
    # no fallback channel: "hybrid" survives a channel outage if that matters more.
    RETRIEVAL_MODE: Literal["dense", "bm25", "sparse", "hybrid"] = Field(
        default="sparse",
        description="Which retrieval channel feeds generation and evaluation",
    )
    # Fusion (Task 6.4, ADR-013). RETRIEVAL_MODE=hybrid queries every channel in
    # HYBRID_CHANNELS for RETRIEVAL_CANDIDATE_LIMIT candidates and fuses them.
    # "sparse" is skipped while ENABLE_NEURAL_SPARSE is off.
    HYBRID_CHANNELS: list[Literal["dense", "bm25", "sparse"]] = Field(
        # A plain default is safe: pydantic copies mutable defaults per instance.
        default=["dense", "bm25", "sparse"],
        description="Channels fused in hybrid mode, e.g. '[\"bm25\", \"sparse\"]'",
    )
    FUSION_METHOD: Literal["rrf", "weighted"] = Field(
        default="rrf",
        description=(
            "rrf: sum of weight/(k + rank) - scale-free, so unbounded BM25 and cosine "
            "scores combine without calibration. weighted: sum of weight x min-max "
            "normalised score, kept for comparison"
        ),
    )
    FUSION_RRF_K: int = Field(
        default=60,
        ge=1,
        description="RRF damping constant; 60 is the value from the original RRF paper",
    )
    FUSION_WEIGHTS: dict[str, float] = Field(
        default_factory=lambda: {"dense": 1.0, "bm25": 1.0, "sparse": 1.0},
        description="Per-channel weight in either fusion method; a missing channel weighs 1.0",
    )
    ENABLE_METADATA_NARROWING: bool = Field(
        default=False,
        description=(
            "Infer metadata constraints from the question (Task 6.3) and search only the "
            "chunks they admit, falling back to the full corpus if that finds nothing"
        ),
    )
    ENABLE_LEXICAL_INDEXING: bool = Field(
        default=True,
        description=(
            "Write chunks to the OpenSearch BM25 index alongside Qdrant when a version "
            "is indexed. Off means OpenSearch is not required, and bm25 mode finds nothing"
        ),
    )

    # Neural sparse retrieval (Stage 6, Task 6.2, ADR-008). Runs inside OpenSearch
    # in "doc-only" mode: documents are expanded by a sparse encoder at index
    # time through an ingest pipeline, and queries are only tokenized, so search
    # needs no model inference. On by default since Task 6.7 made it the
    # production channel; it needs the ML models deployed
    # (scripts/setup_neural_sparse.py) and ~1 GB of OpenSearch native memory.
    ENABLE_NEURAL_SPARSE: bool = Field(
        default=True,
        description="Index chunks into the neural sparse index and allow RETRIEVAL_MODE=sparse",
    )
    SPARSE_INDEX_NAME: str = "enterprise_rag_chunks_sparse"
    SPARSE_INGEST_PIPELINE: str = "enterprise_rag_sparse_encoding"
    SPARSE_DOC_MODEL: str = "amazon/neural-sparse/opensearch-neural-sparse-encoding-doc-v2-distill"
    SPARSE_DOC_MODEL_VERSION: str = "1.0.0"
    SPARSE_QUERY_TOKENIZER: str = "amazon/neural-sparse/opensearch-neural-sparse-tokenizer-v1"
    SPARSE_QUERY_TOKENIZER_VERSION: str = "1.0.1"

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
    GENERATION_PROVIDER: Literal["", "gemini", "groq"] = Field(
        default="",
        description=(
            "Pin answer generation to one hosted provider, disabling the Gemini->Groq "
            "fallback. Empty keeps the fallback. Evaluation runs should pin: with the "
            "fallback, Gemini's 20-per-day cap sends most of a split to Groq, and the "
            "run measures two models under one name. Ignored by the local profile"
        ),
    )
    # answer_v2 (2026-09-26): adds refusal rules for context dumps, fabricated
    # "official" policy text, and user-supplied premises, and pins the citation
    # syntax to [n]. Driven by experiment-005's adversarial failures.
    PROMPT_VERSION_ANSWER: str = "answer_v2"
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
    # 2000, not 800: the Groq judge (gpt-oss-120b) is a reasoning model and spends
    # part of the budget thinking before it writes the JSON verdict. At 800,
    # experiment-021 lost 2 of 60 judgements - one verdict cut off mid-JSON, one
    # empty. The verdict itself is ~150 tokens; the rest is headroom.
    EVAL_JUDGE_MAX_TOKENS: int = 2000
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

    # Parent-child retrieval (Stage 5, Task 5.2, master §19)
    ENABLE_PARENT_EXPANSION: bool = Field(
        default=False,
        description=(
            "Replace retrieved leaf chunks with their parent section for generation. "
            "Off by default so experiment-001-baseline stays reproducible; Task 5.3 "
            "turns it on as a measured comparison, not a default"
        ),
    )
    PARENT_EXPANSION_BUDGET_TOKENS: int = Field(
        default=6000,
        gt=0,
        description=(
            "Token ceiling for expanded context. A section is several times a leaf, "
            "so unbudgeted expansion of eight hits can exceed the generation window"
        ),
    )

    # Feature Flags (Master Plan §2)
    ENABLE_RERANKING: bool = Field(
        default=False,
        description="Rerank RERANK_CANDIDATES retrieved chunks down to RETRIEVAL_TOP_K (Task 6.5)",
    )
    # Late interaction (Task 6.6, ADR-012): an architectural capability, off.
    # ColBERT-style per-token vectors in a Qdrant multivector collection, scored
    # by MaxSim over the retrieved pool. Routing policy stays experimental.
    ENABLE_LATE_INTERACTION: bool = Field(
        default=False,
        description="Reorder the retrieved pool by ColBERT MaxSim before any reranker",
    )
    LATE_INTERACTION_MODEL: str = "jina-colbert-v2"
    LATE_INTERACTION_DIMENSIONS: int = 128
    LATE_INTERACTION_COLLECTION: str = "enterprise_rag_chunks_colbert"
    LATE_INTERACTION_CANDIDATES: int = Field(
        default=20, ge=1, description="Pool scored by MaxSim; all must be indexed"
    )
    RERANK_CANDIDATES: int = Field(
        default=20,
        ge=1,
        description=(
            "Pool handed to the reranker. Wider finds more, but reranker cost and "
            "latency grow linearly with it (~350 tokens per chunk on the hosted tier)"
        ),
    )
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

    @model_validator(mode="after")
    def validate_chunking_version_matches_strategy(self) -> "AppSettings":
        """Keep two strategies from ever sharing a chunk identity.

        Chunk IDs are UUIDv5 over (version_id, element_ids, chunk_index,
        chunking_version) — the strategy name is deliberately not an input, so
        two strategies run under the same CHUNKING_VERSION would produce
        colliding IDs for the same elements. The rows would overwrite each other
        and Stage 5's comparison would silently measure a mixture of both.
        Requiring the version string to name its strategy makes that
        unrepresentable rather than merely discouraged.
        """
        # The trailing hyphen matters: "hierarchical_contextual-s256-o32" also
        # starts with "hierarchical", so a bare prefix test would let those two
        # strategies share a version string and collide on chunk identity.
        if not self.CHUNKING_VERSION.startswith(f"{self.CHUNKING_STRATEGY}-"):
            raise ValueError(
                f"CHUNKING_VERSION must start with CHUNKING_STRATEGY so strategies cannot "
                f"collide on chunk identity. Got strategy={self.CHUNKING_STRATEGY!r} "
                f"version={self.CHUNKING_VERSION!r}; try "
                f"{self.CHUNKING_STRATEGY}-v1."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached singleton application settings.

    No module should read os.environ directly; use get_settings() instead.
    """
    return AppSettings()
