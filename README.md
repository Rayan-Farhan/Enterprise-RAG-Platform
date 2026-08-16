# Enterprise Multimodal RAG Platform

A production-grade, multimodal enterprise knowledge platform whose first application is an HR Policy Assistant.

## Overview
Built with FastAPI, PostgreSQL, MinIO, Qdrant, OpenSearch, RabbitMQ, and Celery, this platform provides enterprise-grade document intelligence, hybrid retrieval (BM25 + Neural Sparse + Dense + Multimodal), grounded generation, precise citations, pre-retrieval document ACLs, and automated evaluation.

## Quick Start
```bash
# 1. Start stateful infrastructure
make up

# 2. Run test suite
pytest tests

# 3. Run linter and type checker
make lint
make typecheck
```

## Documentation & Architecture
- [System Architecture (Current State)](ARCHITECTURE.md)
- [Implementation Roadmap](docs/IMPLEMENTATION_ROADMAP.md)
- [Engineering Blueprint](docs/ENTERPRISE_RAG_FINAL_ENGINEERING_BLUEPRINT_V1.md)
- [Technology Baseline & 52 ADRs](docs/architecture/README.md)
- [Master Plan](docs/production_grade_multimodal_rag_master_plan.md)

