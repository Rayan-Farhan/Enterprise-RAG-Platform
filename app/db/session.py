"""Database engine and asynchronous session management (ADR-002)."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.db.models.base import Base

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return singleton async SQLAlchemy engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        # Connect args / pool settings
        kwargs: dict = {
            "echo": settings.DEBUG,
            "future": True,
        }
        if "sqlite" not in settings.async_database_url:
            kwargs.update(
                {
                    "pool_size": settings.POSTGRES_POOL_SIZE,
                    "max_overflow": settings.POSTGRES_MAX_OVERFLOW,
                    "pool_pre_ping": True,
                }
            )
        _engine = create_async_engine(settings.async_database_url, **kwargs)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return singleton async sessionmaker."""
    global _session_factory
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an AsyncSession."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def transaction_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for standalone async database transactions."""
    factory = get_session_factory()
    async with factory() as session:
        async with session.begin():
            yield session


async def create_tables(engine: AsyncEngine | None = None) -> None:
    """Helper to create all tables (primarily used in local test suites)."""
    target_engine = engine or get_engine()
    async with target_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables(engine: AsyncEngine | None = None) -> None:
    """Helper to drop all tables (primarily used in local test suites)."""
    target_engine = engine or get_engine()
    async with target_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
