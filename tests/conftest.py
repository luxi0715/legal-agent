"""Pytest 全局配置 — session-scoped asyncpg pool + Redis client.

pytest-asyncio 1.x 通过 pyproject.toml 配置 loop scope:
    asyncio_default_fixture_loop_scope = "session"
    asyncio_default_test_loop_scope = "session"

本文件只负责 资源生命周期.每个 memory 单元层测试自动复用同一 pool/redis.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest_asyncio

from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.db.redis_client import close_redis, init_redis


@pytest_asyncio.fixture(scope="session", autouse=True)
async def db_pool() -> AsyncIterator[None]:
    """Session-level asyncpg pool."""
    await init_postgres_pool()
    yield
    await close_postgres_pool()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def redis_client() -> AsyncIterator[None]:
    """Session-level Redis client."""
    await init_redis()
    yield
    await close_redis()
