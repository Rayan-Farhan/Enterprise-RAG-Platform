"""The Celery app, queue topology, and worker launcher (Task 7.1)."""

from __future__ import annotations

import subprocess

import pytest

from app.core.config import AppSettings
from app.workers import cli
from app.workers.celery_app import TASK_MODULES, celery_app, create_celery_app
from app.workers.cli import resolve_concurrency, resolve_pool_implementation, worker_command
from app.workers.queues import (
    MAX_PRIORITY,
    WORKER_POOLS,
    QueueDomain,
    build_queues,
    pool_by_name,
)

LOCKED_DOMAINS = {
    "ingestion",
    "ocr",
    "parsing",
    "chunking",
    "embedding",
    "indexing",
    "visual",
    "reindex",
    "cleanup",
    "evaluation",
}


class TestTopology:
    def test_queue_domains_are_the_locked_set(self) -> None:
        assert {d.value for d in QueueDomain} == LOCKED_DOMAINS

    def test_every_queue_is_consumed_by_exactly_one_pool(self) -> None:
        consumed = [q for pool in WORKER_POOLS for q in pool.queues]

        assert sorted(consumed) == sorted(QueueDomain)

    def test_pool_names_are_unique(self) -> None:
        names = [p.name for p in WORKER_POOLS]
        assert len(names) == len(set(names))

    def test_declared_queues_are_durable_and_carry_priority(self) -> None:
        queues = {q.name: q for q in build_queues()}

        assert set(queues) == LOCKED_DOMAINS
        for name, queue in queues.items():
            assert queue.durable
            assert queue.exchange.name == name
            assert queue.routing_key == name
            assert queue.queue_arguments == {"x-max-priority": MAX_PRIORITY}

    def test_unknown_pool_is_rejected_with_the_known_names(self) -> None:
        with pytest.raises(KeyError, match="orchestration"):
            pool_by_name("gpu")


class TestCeleryApp:
    def test_broker_and_backend_come_from_settings(self) -> None:
        settings = AppSettings(RABBITMQ_HOST="mq.internal", REDIS_HOST="cache.internal")

        app = create_celery_app(settings)

        assert app.conf.broker_url.startswith("amqp://")
        assert "mq.internal" in app.conf.broker_url
        assert "cache.internal" in app.conf.result_backend

    def test_delivery_is_at_least_once(self) -> None:
        conf = celery_app.conf

        assert conf.task_acks_late is True
        assert conf.task_reject_on_worker_lost is True
        assert conf.worker_prefetch_multiplier == 1
        assert conf.task_create_missing_queues is False

    def test_default_queue_is_a_consumed_domain(self) -> None:
        assert celery_app.conf.task_default_queue in LOCKED_DOMAINS

    def test_results_are_off_unless_a_task_opts_in(self) -> None:
        celery_app.loader.import_default_modules()

        assert celery_app.conf.task_ignore_result is True
        assert celery_app.tasks["diagnostics.ping"].ignore_result is False

    def test_every_application_task_routes_to_a_locked_queue(self) -> None:
        # Diagnostics are sent to an explicit queue by the caller; every other
        # task must name its queue, not fall through to the default.
        for module in TASK_MODULES:
            __import__(module)
        application_tasks = {
            name: task
            for name, task in celery_app.tasks.items()
            if not name.startswith(("celery.", "diagnostics."))
        }
        for name, task in application_tasks.items():
            assert task.queue in LOCKED_DOMAINS, f"{name} has no locked queue"


class TestLauncher:
    def test_auto_pool_is_threads_on_windows_and_prefork_elsewhere(self) -> None:
        settings = AppSettings()

        assert resolve_pool_implementation(settings, platform="win32") == "threads"
        assert resolve_pool_implementation(settings, platform="linux") == "prefork"

    def test_explicit_pool_implementation_wins(self) -> None:
        settings = AppSettings(WORKER_POOL_IMPLEMENTATION="solo")

        assert resolve_pool_implementation(settings, platform="linux") == "solo"

    def test_command_binds_the_pool_to_its_own_queues(self) -> None:
        pool = pool_by_name("document")

        command = worker_command(pool, AppSettings(), platform="linux")

        assert command[command.index("--queues") + 1] == "parsing,ocr,chunking"
        assert command[command.index("--hostname") + 1] == "document@%h"
        assert command[command.index("--concurrency") + 1] == str(pool.concurrency)
        assert command[-2:] == ["-O", "fair"]

    def test_threads_pool_gets_no_prefork_scheduling_flag(self) -> None:
        command = worker_command(pool_by_name("document"), AppSettings(), platform="win32")

        assert command[command.index("--pool") + 1] == "threads"
        assert "-O" not in command

    def test_concurrency_override_applies_to_the_named_pool_only(self) -> None:
        settings = AppSettings(WORKER_CONCURRENCY={"document": 6})

        assert resolve_concurrency(pool_by_name("document"), settings) == 6
        assert resolve_concurrency(pool_by_name("inference"), settings) == 2

    def test_misspelt_pool_override_fails_loudly(self) -> None:
        settings = AppSettings(WORKER_CONCURRENCY={"documents": 6})

        with pytest.raises(ValueError, match="documents"):
            resolve_concurrency(pool_by_name("document"), settings)

    def test_zero_concurrency_is_rejected(self) -> None:
        settings = AppSettings(WORKER_CONCURRENCY={"document": 0})

        with pytest.raises(ValueError, match=">= 1"):
            resolve_concurrency(pool_by_name("document"), settings)


class FakeProcess:
    def __init__(self, exits_with: int | None = None) -> None:
        self.returncode = exits_with
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("celery", timeout or 0)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def patch_popen(monkeypatch: pytest.MonkeyPatch, processes: list[FakeProcess]) -> None:
    queue = iter(processes)
    monkeypatch.setattr(cli.subprocess, "Popen", lambda _command: next(queue))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "SHUTDOWN_GRACE_SECONDS", 0.0)


class TestSupervision:
    def test_one_pool_dying_stops_the_rest_and_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        survivor, casualty = FakeProcess(), FakeProcess(exits_with=3)
        patch_popen(monkeypatch, [survivor, casualty])

        code = cli.start(["document", "inference"], dry_run=False, settings=AppSettings())

        assert code == 3
        assert survivor.terminated
        assert not casualty.terminated

    def test_interrupt_leaves_the_warm_shutdown_to_the_workers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Ctrl-C already reached the workers; the launcher only waits.
        worker = FakeProcess()
        patch_popen(monkeypatch, [worker])

        def interrupt(_seconds: float) -> None:
            worker.returncode = 0
            raise KeyboardInterrupt

        monkeypatch.setattr(cli.time, "sleep", interrupt)

        assert cli.start(["document"], dry_run=False, settings=AppSettings()) == 0
        assert not worker.terminated
        assert not worker.killed

    def test_a_worker_that_ignores_shutdown_is_killed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stuck = FakeProcess()
        stuck.terminate = lambda: None  # type: ignore[method-assign]
        patch_popen(monkeypatch, [stuck, FakeProcess(exits_with=1)])

        cli.start(["document", "inference"], dry_run=False, settings=AppSettings())

        assert stuck.killed
