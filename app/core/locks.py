"""Redis distributed locks with a TTL and an ownership token (Task 7.6, ADR-020, master §39).

    async with distributed_lock(f"version-activation:{document_id}", ttl=30, wait=60):
        ...  # exactly one holder across every worker and API process

Acquiring is `SET key token NX PX ttl`: it succeeds for one caller only, and
the TTL means a holder that dies - a killed worker - cannot hold the lock
forever. The token is random per acquisition, and release and extension are
Lua scripts that act only if the key still holds *this* token, so a holder
whose lock expired and was taken by someone else cannot delete or prolong the
new holder's lock.

A lock held through a context manager renews itself every third of its TTL,
from a thread of its own so that a holder blocking its event loop still keeps it.
If a renewal finds the lock gone (Redis restarted, the process stalled past
the TTL) the lock is marked lost; `ensure_held()` raises before a holder
commits work it may no longer own. Locks narrow races, they do not replace
the database: every write they guard is still a conditional update or sits
behind a uniqueness constraint (ADR-036), so a lost lock can at worst cause a
redundant attempt, never a duplicate.
"""

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Protocol

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger("app.core.locks")

KEY_PREFIX = "lock:"

# Delete / extend only while the key still holds our token.
RELEASE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('del', KEYS[1])
end
return 0
"""
EXTEND_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
    return redis.call('pexpire', KEYS[1], ARGV[2])
end
return 0
"""


class LockClient(Protocol):
    async def set(self, name: str, value: str, *, nx: bool, px: int) -> Any: ...

    async def get(self, name: str) -> Any: ...

    async def eval(self, script: str, numkeys: int, *keys_and_args: Any) -> Any: ...

    async def aclose(self) -> None: ...


def _redis_client() -> LockClient:
    import redis.asyncio as aioredis

    return aioredis.from_url(get_settings().redis_url, decode_responses=True)  # type: ignore[return-value]


# Indirection so tests can supply an in-memory client. A redis.asyncio client
# belongs to the event loop that opened it, and every job runs on its own
# loop, so a client is opened per lock use rather than shared.
lock_client_factory: Callable[[], LockClient] = _redis_client


class LockNotAcquired(Exception):
    """Someone else holds the lock and waiting for it was not allowed or ran out."""


class LockLost(Exception):
    """The lock expired or was taken while we believed we held it."""


class DistributedLock:
    def __init__(self, client: LockClient, name: str, ttl: float) -> None:
        if ttl <= 0:
            raise ValueError("lock ttl must be positive")
        self.client = client
        self.name = name
        self.key = KEY_PREFIX + name
        self.ttl_ms = int(ttl * 1000)
        self.token = secrets.token_hex(16)
        self.lost = False

    async def acquire(self, wait: float = 0.0, poll: float = 0.1) -> bool:
        """Take the lock, waiting up to ``wait`` seconds for the holder to let go."""
        deadline = time.monotonic() + wait
        while True:
            if await self.client.set(self.key, self.token, nx=True, px=self.ttl_ms):
                self.lost = False
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(min(poll, max(0.0, deadline - time.monotonic())))

    async def release(self) -> bool:
        """Let go; False when the lock had already expired or changed hands."""
        return bool(await self.client.eval(RELEASE_SCRIPT, 1, self.key, self.token))

    async def extend(self) -> bool:
        """Reset the TTL; False (and marked lost) when the lock is no longer ours."""
        if await self.client.eval(EXTEND_SCRIPT, 1, self.key, self.token, self.ttl_ms):
            return True
        self.lost = True
        return False

    async def held(self) -> bool:
        return bool(await self.client.get(self.key) == self.token)

    def ensure_held(self) -> None:
        if self.lost:
            raise LockLost(f"lock '{self.name}' was lost while held")


class _Renewer(threading.Thread):
    """Keeps a held lock alive from its own thread, on its own event loop.

    Not an asyncio task on the holder's loop: pipeline steps call synchronous
    clients (the parser, OpenSearch bulk writes) from async code and can stall
    their loop for minutes, and a renewal waiting on that loop would let the
    lock expire under a holder that is still working - found in the Task 7.6
    live check, where a 90-second sparse-encoding step lost its job lock.
    """

    def __init__(self, lock: DistributedLock) -> None:
        super().__init__(name=f"lock-renewer:{lock.name}", daemon=True)
        self.lock = lock
        self.stopped = threading.Event()

    def run(self) -> None:
        asyncio.run(self._renew())

    async def _renew(self) -> None:
        held = self.lock
        client = lock_client_factory()
        twin = DistributedLock(client, held.name, held.ttl_ms / 1000)
        twin.token = held.token
        interval = held.ttl_ms / 1000 / 3
        try:
            # A blocking wait is fine here: this loop belongs to this thread.
            while not self.stopped.wait(interval):
                try:
                    renewed = await twin.extend()
                except Exception as exc:
                    # Redis unreachable: assume the worst, since the TTL keeps running.
                    logger.warning("lock_renewal_failed", lock=held.name, error=str(exc))
                    renewed = False
                if not renewed:
                    held.lost = True
                    logger.warning("lock_lost", lock=held.name)
                    return
        finally:
            await client.aclose()

    def stop(self) -> None:
        self.stopped.set()
        self.join()


@asynccontextmanager
async def distributed_lock(
    name: str, *, ttl: float = 30.0, wait: float = 0.0
) -> AsyncIterator[DistributedLock]:
    """Hold ``name`` for the block, renewing it; raise LockNotAcquired if it is taken."""
    client = lock_client_factory()
    lock = DistributedLock(client, name, ttl)
    try:
        if not await lock.acquire(wait=wait):
            raise LockNotAcquired(f"lock '{name}' is held by another process")
        renewer = _Renewer(lock)
        renewer.start()
        try:
            yield lock
        finally:
            await asyncio.to_thread(renewer.stop)
            if not await lock.release():
                logger.warning("lock_expired_before_release", lock=name)
    finally:
        await client.aclose()
