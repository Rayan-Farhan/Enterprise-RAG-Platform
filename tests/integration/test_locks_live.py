"""The distributed lock against a real Redis: the Lua scripts, TTLs, renewal (Task 7.6).

Unit tests interpret the scripts by identity; only a real Redis proves that
compare-and-delete and compare-and-extend do what they claim. Skips when
Redis is not running, unless RAG_REQUIRE_SERVICES=1.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import pytest
import redis.asyncio as aioredis

from app.core import locks
from app.core.config import get_settings
from app.core.locks import DistributedLock, LockNotAcquired, distributed_lock


@pytest.fixture
async def real_redis(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    url = get_settings().redis_url
    probe = aioredis.from_url(url)
    try:
        await probe.ping()
    except Exception:  # noqa: BLE001 - any failure means "not available"
        if os.getenv("RAG_REQUIRE_SERVICES") == "1":
            pytest.fail("Redis is required (RAG_REQUIRE_SERVICES=1)")
        pytest.skip("Redis not reachable; start it with `make up`")
    finally:
        await probe.aclose()
    monkeypatch.setattr(
        locks, "lock_client_factory", lambda: aioredis.from_url(url, decode_responses=True)
    )
    yield


def name() -> str:
    return f"test:{uuid.uuid4()}"


async def test_compare_and_delete_spares_another_holders_lock(real_redis: None) -> None:
    key = name()
    stale = DistributedLock(locks.lock_client_factory(), key, ttl=0.2)
    assert await stale.acquire()
    await asyncio.sleep(0.3)  # expired in Redis
    fresh = DistributedLock(locks.lock_client_factory(), key, ttl=5)
    assert await fresh.acquire()

    assert not await stale.release()
    assert not await stale.extend()
    assert await fresh.held()
    assert await fresh.release()
    assert not await fresh.held()


async def test_extend_resets_the_ttl(real_redis: None) -> None:
    lock = DistributedLock(locks.lock_client_factory(), name(), ttl=0.3)
    await lock.acquire()
    for _ in range(4):
        await asyncio.sleep(0.15)
        assert await lock.extend()
    assert await lock.held()
    await lock.release()


async def test_the_context_renews_and_excludes_others(real_redis: None) -> None:
    key = name()
    async with distributed_lock(key, ttl=0.3) as lock:
        await asyncio.sleep(0.8)  # well past the TTL; renewal keeps it
        assert await lock.held()
        with pytest.raises(LockNotAcquired):
            async with distributed_lock(key, ttl=5):
                pass
    async with distributed_lock(key, ttl=5):  # released on exit
        pass


async def test_concurrent_contenders_get_exactly_one_holder(real_redis: None) -> None:
    key = name()
    attempts = [DistributedLock(locks.lock_client_factory(), key, ttl=5) for _ in range(20)]

    won = await asyncio.gather(*(a.acquire() for a in attempts))

    assert sum(won) == 1
    await next(a for a, w in zip(attempts, won, strict=True) if w).release()
