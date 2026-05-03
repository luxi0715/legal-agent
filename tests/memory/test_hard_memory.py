"""单元测试 — Hard Memory (M8.2)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from legal_agent.memory.hard_memory import (
    delete_all_user_facts,
    delete_user_fact,
    get_user_facts,
    get_user_facts_with_meta,
    upsert_user_fact,
)


@pytest_asyncio.fixture
async def fresh_user_id() -> AsyncIterator[uuid.UUID]:
    """每个测试函数:给一个全新 user_id,测试结束清理数据.

    pool 由 conftest.py 的 session-scoped autouse fixture 提供.
    """
    uid = uuid.uuid4()
    try:
        yield uid
    finally:
        await delete_all_user_facts(uid)


@pytest.mark.asyncio
async def test_get_facts_empty(fresh_user_id: uuid.UUID) -> None:
    """新用户,facts 应为空 dict."""
    facts = await get_user_facts(fresh_user_id)
    assert facts == {}


@pytest.mark.asyncio
async def test_upsert_then_get(fresh_user_id: uuid.UUID) -> None:
    """写入 fact,再读出来."""
    await upsert_user_fact(fresh_user_id, "location", "北京")
    facts = await get_user_facts(fresh_user_id)
    assert facts == {"location": "北京"}


@pytest.mark.asyncio
async def test_upsert_overwrite(fresh_user_id: uuid.UUID) -> None:
    """同 key 重复 upsert,新值覆盖旧值."""
    await upsert_user_fact(fresh_user_id, "location", "北京")
    await upsert_user_fact(fresh_user_id, "location", "上海")
    facts = await get_user_facts(fresh_user_id)
    assert facts == {"location": "上海"}


@pytest.mark.asyncio
async def test_multiple_keys(fresh_user_id: uuid.UUID) -> None:
    """多个不同 key 共存."""
    await upsert_user_fact(fresh_user_id, "location", "北京")
    await upsert_user_fact(fresh_user_id, "occupation", "程序员")
    await upsert_user_fact(fresh_user_id, "age_range", "25-30")
    facts = await get_user_facts(fresh_user_id)
    assert facts == {
        "location": "北京",
        "occupation": "程序员",
        "age_range": "25-30",
    }


@pytest.mark.asyncio
async def test_confidence_default(fresh_user_id: uuid.UUID) -> None:
    """confidence 不传时默认 1.0."""
    await upsert_user_fact(fresh_user_id, "location", "北京")
    meta = await get_user_facts_with_meta(fresh_user_id)
    assert len(meta) == 1
    assert meta[0]["confidence"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_confidence_custom(fresh_user_id: uuid.UUID) -> None:
    """confidence 自定义."""
    await upsert_user_fact(fresh_user_id, "location", "北京", confidence=0.7)
    meta = await get_user_facts_with_meta(fresh_user_id)
    assert meta[0]["confidence"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_delete_existing(fresh_user_id: uuid.UUID) -> None:
    """删除存在的 key 返回 True."""
    await upsert_user_fact(fresh_user_id, "location", "北京")
    assert await delete_user_fact(fresh_user_id, "location") is True
    assert await get_user_facts(fresh_user_id) == {}


@pytest.mark.asyncio
async def test_delete_nonexistent(fresh_user_id: uuid.UUID) -> None:
    """删除不存在的 key 返回 False."""
    assert await delete_user_fact(fresh_user_id, "ghost_key") is False


@pytest.mark.asyncio
async def test_delete_all(fresh_user_id: uuid.UUID) -> None:
    """清空所有 facts."""
    await upsert_user_fact(fresh_user_id, "location", "北京")
    await upsert_user_fact(fresh_user_id, "occupation", "程序员")
    deleted = await delete_all_user_facts(fresh_user_id)
    assert deleted == 2
    assert await get_user_facts(fresh_user_id) == {}


@pytest.mark.asyncio
async def test_isolation_between_users() -> None:
    """不同 user_id 互不影响."""
    user_a = uuid.uuid4()
    user_b = uuid.uuid4()

    try:
        await upsert_user_fact(user_a, "location", "北京")
        await upsert_user_fact(user_b, "location", "上海")

        assert await get_user_facts(user_a) == {"location": "北京"}
        assert await get_user_facts(user_b) == {"location": "上海"}
    finally:
        await delete_all_user_facts(user_a)
        await delete_all_user_facts(user_b)
