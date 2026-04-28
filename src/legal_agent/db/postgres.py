"""PostgreSQL connection pool using asyncpg."""

import asyncpg

from legal_agent.core.config import get_settings

_pool: asyncpg.Pool | None = None


async def init_postgres_pool() -> asyncpg.Pool:
    """Initialize the global connection pool. Call once at startup."""
    global _pool
    if _pool is not None:
        return _pool

    settings = get_settings()
    _pool = await asyncpg.create_pool(
        dsn=settings.postgres_dsn,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    if _pool is None:
        raise RuntimeError("Failed to create PostgreSQL pool")
    return _pool


async def close_postgres_pool() -> None:
    """Close the pool. Call at shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_postgres_pool() -> asyncpg.Pool:
    """Return the initialized pool. Raises if not initialized."""
    if _pool is None:
        raise RuntimeError("PostgreSQL pool not initialized; call init_postgres_pool first")
    return _pool
