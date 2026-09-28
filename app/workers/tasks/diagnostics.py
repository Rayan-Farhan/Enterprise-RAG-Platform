"""Diagnostic tasks: prove a queue is consumed end to end (Task 7.1).

`ping` has no default queue on purpose. It is sent to each queue explicitly, and
the answer says which worker took it from where.
"""

from __future__ import annotations

from typing import Any

from celery import Task

from app.workers.celery_app import celery_app


@celery_app.task(name="diagnostics.ping", bind=True, ignore_result=False)
def ping(self: Task) -> dict[str, Any]:
    delivery = self.request.delivery_info or {}
    return {
        "worker": self.request.hostname,
        "queue": delivery.get("routing_key"),
    }
