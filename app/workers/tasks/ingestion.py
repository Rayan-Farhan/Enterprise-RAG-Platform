"""The ingestion chain as Celery tasks (Task 7.3, ADR-018).

Each task is one step from `app.ingestion.pipeline`, bound to its queue and to
the step after it. A step learns its document and version from its job row and
receives the upload's metadata as its only argument, the same for every step,
so any one can be re-run on its own.
`start_ingestion` enqueues the first; each later step is enqueued by the one
before it once that has succeeded, so a failed or cancelled step stops the
chain there.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from app.core.logging import get_logger
from app.db.models.job import Job, JobType
from app.ingestion import pipeline
from app.ingestion.pipeline import PipelineServices, StepOutcome
from app.jobs.service import JobService
from app.workers.job_task import JobContext, enqueue, job_task
from app.workers.queues import QueueDomain

logger = get_logger("app.workers.ingestion")

# Indirection so tests can run the chain against in-memory doubles.
services_factory: Callable[[], PipelineServices] = pipeline.build_pipeline_services


async def _run(step: Callable[..., Any], ctx: JobContext, **extra: Any) -> None:
    if ctx.version_id is None:
        raise ValueError(f"ingestion job {ctx.job_id} has no version")
    services = services_factory()
    async with ctx.sessions() as session:
        outcome: StepOutcome = await step(
            services, session, ctx.version_id, progress=ctx.progress, **extra
        )
    logger.info(
        "ingestion_step_complete",
        job_id=str(ctx.job_id),
        step=step.__name__,
        version_id=str(ctx.version_id),
        noop=outcome.noop,
        summary=outcome.summary,
    )
    await ctx.progress(1.0, outcome.summary)


@job_task(
    name="ingestion.parse_document",
    queue=QueueDomain.PARSING,
    task_type=JobType.PARSE_DOCUMENT,
    then="ingestion.extract_pages",
)
async def parse_document(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.parse_document, ctx)


@job_task(
    name="ingestion.extract_pages",
    queue=QueueDomain.PARSING,
    task_type=JobType.EXTRACT_PAGES,
    then="ingestion.ocr_pages",
)
async def extract_pages(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.extract_pages, ctx)


@job_task(
    name="ingestion.ocr_pages",
    queue=QueueDomain.OCR,
    task_type=JobType.OCR_PAGES,
    then="ingestion.normalize_document",
)
async def ocr_pages(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.ocr_pages, ctx)


@job_task(
    name="ingestion.normalize_document",
    queue=QueueDomain.PARSING,
    task_type=JobType.NORMALIZE_DOCUMENT,
    then="ingestion.chunk_document",
)
async def normalize_document(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.normalize_document, ctx, metadata=metadata)


@job_task(
    name="ingestion.chunk_document",
    queue=QueueDomain.CHUNKING,
    task_type=JobType.CHUNK_DOCUMENT,
    then="ingestion.generate_embeddings",
)
async def chunk_document(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.chunk_document, ctx)


@job_task(
    name="ingestion.generate_embeddings",
    queue=QueueDomain.EMBEDDING,
    task_type=JobType.GENERATE_EMBEDDINGS,
    then="ingestion.index_opensearch",
)
async def generate_embeddings(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.generate_embeddings, ctx)


@job_task(
    name="ingestion.index_opensearch",
    queue=QueueDomain.INDEXING,
    task_type=JobType.INDEX_OPENSEARCH,
    then="ingestion.index_qdrant",
)
async def index_opensearch(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.index_opensearch, ctx)


@job_task(
    name="ingestion.index_qdrant",
    queue=QueueDomain.INDEXING,
    task_type=JobType.INDEX_QDRANT,
    then="ingestion.validate_index",
)
async def index_qdrant(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.index_qdrant, ctx)


@job_task(
    name="ingestion.validate_index",
    queue=QueueDomain.INDEXING,
    task_type=JobType.VALIDATE_INDEX,
    then="ingestion.publish_version",
)
async def validate_index(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.validate_index, ctx)


@job_task(
    name="ingestion.publish_version",
    queue=QueueDomain.INGESTION,
    task_type=JobType.PUBLISH_VERSION,
)
async def publish_version(ctx: JobContext, *, metadata: dict[str, Any]) -> None:
    await _run(pipeline.publish_version, ctx)


CHAIN = (
    parse_document,
    extract_pages,
    ocr_pages,
    normalize_document,
    chunk_document,
    generate_embeddings,
    index_opensearch,
    index_qdrant,
    validate_index,
    publish_version,
)


async def start_ingestion(
    document_id: uuid.UUID,
    version_id: uuid.UUID,
    metadata: dict[str, Any] | None = None,
    jobs: JobService | None = None,
) -> Job:
    """Enqueue the chain's first step for an accepted upload."""
    return await enqueue(
        parse_document,
        document_id=document_id,
        version_id=version_id,
        jobs=jobs,
        metadata=metadata or {},
    )
