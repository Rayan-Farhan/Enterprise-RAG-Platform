"""Diagnostic tasks (Tasks 7.1, 7.2).

`ping` proves a queue is consumed end to end. It has no default queue on
purpose: it is sent to each queue explicitly, and the answer says which worker
took it from where. `exercise_job` drives the job lifecycle without real work.
"""

from __future__ import annotations

import asyncio
from typing import Any

from celery import Task

from app.workers.celery_app import celery_app
from app.workers.job_task import JobContext, job_task
from app.workers.queues import QueueDomain


@celery_app.task(name="diagnostics.ping", bind=True, ignore_result=False)
def ping(self: Task) -> dict[str, Any]:
    delivery = self.request.delivery_info or {}
    return {
        "worker": self.request.hostname,
        "queue": delivery.get("routing_key"),
    }


@job_task(name="diagnostics.exercise_job", queue=QueueDomain.INGESTION)
async def exercise_job(ctx: JobContext, *, steps: int = 5, step_seconds: float = 1.0) -> None:
    """A job that does nothing but report progress, for exercising the lifecycle.

    Watch it with `GET /api/v1/jobs/{id}` or cancel it mid-way; it stops at the
    next step after a cancel is requested.
    """
    for step in range(1, steps + 1):
        await asyncio.sleep(step_seconds)
        await ctx.progress(step / steps, f"step {step} of {steps}")
