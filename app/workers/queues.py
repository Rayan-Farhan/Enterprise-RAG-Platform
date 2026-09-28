"""Queue domains and the worker pools that consume them (Task 7.1, ADR-018, ADR-043).

A queue names a kind of work; a pool names a resource class. Queues are the
routing vocabulary tasks use, and pools decide which processes do the work, so
the two can be regrouped without touching any task. Every queue belongs to
exactly one pool - a queue no pool consumes would accept work that never runs.

The query path is not here: answering runs in the API process, so ingestion
load can only ever compete with it for the machine, never for a worker slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from kombu import Exchange, Queue


class QueueDomain(StrEnum):
    """The locked queue domains (Task 7.1)."""

    INGESTION = "ingestion"
    OCR = "ocr"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    VISUAL = "visual"
    REINDEX = "reindex"
    CLEANUP = "cleanup"
    EVALUATION = "evaluation"


# Declared now, not when Task 7.7 adds priorities: RabbitMQ fixes a queue's
# arguments at declaration, and redeclaring an existing queue with a different
# x-max-priority is refused (PRECONDITION_FAILED). Adding it later would mean
# deleting every queue and whatever is in it. Ten levels leave room for the
# five ADR-042 classes without the per-level cost of the 255 maximum.
MAX_PRIORITY = 10


@dataclass(frozen=True)
class WorkerPool:
    """One resource class: a worker process group with its own concurrency."""

    name: str
    queues: tuple[QueueDomain, ...]
    concurrency: int
    resource: str


WORKER_POOLS: tuple[WorkerPool, ...] = (
    WorkerPool(
        name="orchestration",
        queues=(QueueDomain.INGESTION, QueueDomain.CLEANUP),
        concurrency=4,
        resource="database and object-storage I/O; starts chains, sweeps artifacts",
    ),
    WorkerPool(
        name="document",
        queues=(QueueDomain.PARSING, QueueDomain.OCR, QueueDomain.CHUNKING),
        concurrency=2,
        resource="CPU and memory: parsers and OCR hold whole documents in memory",
    ),
    WorkerPool(
        name="inference",
        queues=(QueueDomain.EMBEDDING, QueueDomain.VISUAL),
        concurrency=2,
        resource="model calls, bounded by provider rate limits rather than local cores",
    ),
    WorkerPool(
        name="indexing",
        queues=(QueueDomain.INDEXING, QueueDomain.REINDEX),
        concurrency=2,
        resource="writes to OpenSearch and Qdrant",
    ),
    WorkerPool(
        name="evaluation",
        queues=(QueueDomain.EVALUATION,),
        concurrency=1,
        resource="hosted generation and judge calls under a daily free-tier budget",
    ),
)


def pool_by_name(name: str) -> WorkerPool:
    for pool in WORKER_POOLS:
        if pool.name == name:
            return pool
    known = ", ".join(p.name for p in WORKER_POOLS)
    raise KeyError(f"Unknown worker pool {name!r}; known pools: {known}")


def build_queues() -> tuple[Queue, ...]:
    """Durable kombu declarations, one direct exchange per queue domain."""
    return tuple(
        Queue(
            domain.value,
            Exchange(domain.value, type="direct", durable=True),
            routing_key=domain.value,
            durable=True,
            queue_arguments={"x-max-priority": MAX_PRIORITY},
        )
        for domain in QueueDomain
    )
