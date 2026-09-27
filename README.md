# Enterprise Multimodal RAG Platform

A production-grade enterprise knowledge platform whose first application is an
HR Policy Assistant: answers are grounded in retrieved policy text, every citation
is checked against the evidence the model was given, and every configuration
choice is set by a measured experiment.

## What works today (roadmap Stages 0–6)

- **Document intelligence** — Docling-first parsing with fallbacks, a canonical
  Document → Version → Page → Element model, deduplication and boilerplate detection.
- **Retrieval** — neural sparse by default (OpenSearch), with dense (Qdrant), BM25
  and RRF hybrid selectable; optional metadata narrowing, reranking, ColBERT late
  interaction and parent-section expansion behind flags.
- **Grounded generation** — fenced evidence, versioned prompts, citation validation,
  a context-leak guard, and abstention when evidence is insufficient.
- **Evaluation** — a 181-question golden dataset over a real HR corpus, retrieval /
  answer / LLM-judge metrics, resumable experiment runs, and a CI regression gate.

Planned next, not yet built: the async Celery ingestion plane (Stage 7), document
ACLs (Stage 8), multimodal retrieval (Stage 9), and query understanding (Stage 10).

## Quick start

```bash
make up                                   # PostgreSQL, Qdrant, OpenSearch, MinIO, Redis, RabbitMQ
make migrate                              # database schema
python -m scripts.setup_neural_sparse     # OpenSearch ML models for the default retrieval channel
python -m scripts.ingest_corpus           # parse, chunk and index the benchmark corpus
make verify                               # lint + typecheck + tests
```

Configuration is environment-driven (`app/core/config.py`); the hosted inference
profile needs `GEMINI_API_KEY`, `GROQ_API_KEY` and `JINA_API_KEY`.

## Documentation

- [Architecture](ARCHITECTURE.md) — system overview, query path, technology and why
- [Architecture diagrams](docs/diagrams/architecture-current-state.html)
- [Current state](docs/roadmap/CURRENT_STATE.md) — progress, module map, experiment record
- [Implementation roadmap](docs/roadmap/IMPLEMENTATION_ROADMAP.md)
- [Engineering blueprint](docs/roadmap/ENTERPRISE_RAG_FINAL_ENGINEERING_BLUEPRINT_V1.md)
- [Technology baseline and ADRs](docs/architecture/README.md)
- [Master plan](docs/roadmap/production_grade_multimodal_rag_master_plan.md)
