"""Start the worker pools and check that every queue is consumed (Task 7.1).

    python -m app.workers.cli start              # every pool (what `make workers` runs)
    python -m app.workers.cli start --pool document
    python -m app.workers.cli start --dry-run    # print the worker commands only
    python -m app.workers.cli check              # ping every queue, report who answered

`start` launches one `celery worker` process per pool, each bound to its own
queues and concurrency, and stops them all together. It is a development
convenience; in a deployment each pool is its own service, scaled on its own.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Sequence

from app.core.config import AppSettings, get_settings
from app.workers.queues import WORKER_POOLS, QueueDomain, WorkerPool, pool_by_name

# How long `start` waits for workers to exit on shutdown before killing them.
SHUTDOWN_GRACE_SECONDS = 15.0


def resolve_pool_implementation(settings: AppSettings, platform: str = sys.platform) -> str:
    if settings.WORKER_POOL_IMPLEMENTATION != "auto":
        return settings.WORKER_POOL_IMPLEMENTATION
    return "threads" if platform == "win32" else "prefork"


def resolve_concurrency(pool: WorkerPool, settings: AppSettings) -> int:
    unknown = set(settings.WORKER_CONCURRENCY) - {p.name for p in WORKER_POOLS}
    if unknown:
        # A misspelt pool name would otherwise leave the real pool at its default
        # with nothing to say the override was ignored.
        raise ValueError(f"WORKER_CONCURRENCY names unknown pools: {sorted(unknown)}")
    concurrency = settings.WORKER_CONCURRENCY.get(pool.name, pool.concurrency)
    if concurrency < 1:
        raise ValueError(f"WORKER_CONCURRENCY for {pool.name!r} must be >= 1")
    return concurrency


def worker_command(
    pool: WorkerPool, settings: AppSettings, platform: str = sys.platform
) -> list[str]:
    implementation = resolve_pool_implementation(settings, platform)
    command = [
        sys.executable,
        "-m",
        "celery",
        "--app",
        "app.workers.celery_app:celery_app",
        "worker",
        "--queues",
        ",".join(q.value for q in pool.queues),
        "--hostname",
        f"{pool.name}@%h",
        "--concurrency",
        str(resolve_concurrency(pool, settings)),
        "--pool",
        implementation,
        "--loglevel",
        "INFO",
    ]
    if implementation == "prefork":
        # Hand a task only to a child that is free, not one mid-way through a
        # long parse; the default distributes by round robin.
        command += ["-O", "fair"]
    return command


def start(pool_names: Sequence[str], dry_run: bool, settings: AppSettings) -> int:
    pools = [pool_by_name(n) for n in pool_names] if pool_names else list(WORKER_POOLS)
    commands = [(pool, worker_command(pool, settings)) for pool in pools]
    if dry_run:
        for pool, command in commands:
            print(f"[{pool.name}] {subprocess.list2cmdline(command)}")
        return 0

    processes = [(pool, subprocess.Popen(command)) for pool, command in commands]
    exit_code = 0
    interrupted = False
    try:
        # Workers run until interrupted. One dying takes the set down with it:
        # a silently missing pool leaves its queues filling with nobody reading.
        while exit_code == 0:
            for pool, process in processes:
                code = process.poll()
                if code is not None:
                    print(f"worker pool {pool.name!r} exited with {code}", file=sys.stderr)
                    exit_code = code or 1
                    break
            time.sleep(1)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        _stop(processes, already_signalled=interrupted)
    return exit_code


def _stop(
    processes: list[tuple[WorkerPool, subprocess.Popen[bytes]]], already_signalled: bool
) -> None:
    """Give every worker a warm shutdown, then kill whatever is still running.

    Ctrl-C reaches the whole console process group, so after an interrupt the
    workers are already finishing their current tasks and must not be
    terminated on top of it. Otherwise they are sent SIGTERM, which is a warm
    shutdown on POSIX; on Windows it is a hard kill, and acks_late puts any
    in-flight task back on its queue.
    """
    if not already_signalled:
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
    deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
    for pool, process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            print(f"worker pool {pool.name!r} did not stop in time; killing", file=sys.stderr)
            process.kill()


def check(timeout: float) -> int:
    """Send a ping to every queue and report which worker answered it."""
    from app.workers.celery_app import celery_app

    owner = {q: p.name for p in WORKER_POOLS for q in p.queues}
    # An unanswered ping expires, so a worker started later discards it rather
    # than answering a check that has already reported failure.
    pending = {
        domain: celery_app.send_task("diagnostics.ping", queue=domain.value, expires=timeout)
        for domain in QueueDomain
    }
    deadline = time.monotonic() + timeout
    failures = 0
    for domain, result in pending.items():
        try:
            reply = result.get(timeout=max(0.1, deadline - time.monotonic()))
            print(f"ok    {domain.value:<11} -> {reply['worker']} (pool {owner[domain]})")
        except Exception as exc:  # noqa: BLE001 - report every queue, not the first failure
            failures += 1
            print(f"FAIL  {domain.value:<11} -> no reply ({type(exc).__name__}: {exc})")
    return 1 if failures else 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.workers.cli")
    commands = parser.add_subparsers(dest="command", required=True)

    start_parser = commands.add_parser("start", help="start worker pools")
    start_parser.add_argument(
        "--pool",
        action="append",
        default=[],
        choices=[p.name for p in WORKER_POOLS],
        help="pool to start; repeatable; default is every pool",
    )
    start_parser.add_argument("--dry-run", action="store_true")

    check_parser = commands.add_parser("check", help="ping every queue")
    check_parser.add_argument("--timeout", type=float, default=10.0)

    args = parser.parse_args(argv)
    if args.command == "start":
        return start(args.pool, args.dry_run, get_settings())
    return check(args.timeout)


if __name__ == "__main__":
    sys.exit(main())
