"""Test database connection and basic operations."""

from uuid import uuid4

import pytest

from legal_agent.db.messages import (
    get_or_create_session,
    list_session_messages,
    save_message,
)
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.db.redis_client import close_redis, get_redis, init_redis


@pytest.fixture
async def db_pool():
    """Set up and tear down DB pool for each test."""
    await init_postgres_pool()
    yield
    await close_postgres_pool()


@pytest.fixture
async def redis_client():
    """Set up and tear down Redis client for each test."""
    await init_redis()
    yield
    await close_redis()


async def test_postgres_session_message_roundtrip(db_pool: None) -> None:
    """Create session, save messages, read them back."""
    session_id = await get_or_create_session()
    assert session_id is not None

    await save_message(session_id, "user", "你好")
    await save_message(session_id, "assistant", "您好,有什么法律问题吗?")

    messages = await list_session_messages(session_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "你好"
    assert messages[1]["role"] == "assistant"


async def test_redis_set_get(redis_client: None) -> None:
    """Redis basic set/get should work."""
    redis = get_redis()
    test_key = f"test:{uuid4()}"
    await redis.set(test_key, "hello", ex=10)
    value = await redis.get(test_key)
    assert value == "hello"
    await redis.delete(test_key)
