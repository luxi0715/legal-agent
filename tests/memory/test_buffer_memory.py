"""单元测试 — Buffer Memory (M8.3)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from legal_agent.memory.buffer_memory import (
    BUFFER_MAX_ITEMS,
    append_to_buffer,
    clear_buffer,
    get_buffer,
    get_buffer_size,
    get_oldest_n,
    trim_oldest_n,
)


@pytest_asyncio.fixture
async def fresh_session_id() -> AsyncIterator[uuid.UUID]:
    """每个测试函数:给一个全新 session_id,测试结束清理 buffer."""
    sid = uuid.uuid4()
    try:
        yield sid
    finally:
        await clear_buffer(sid)


@pytest.mark.asyncio
async def test_get_empty_buffer(fresh_session_id: uuid.UUID) -> None:
    """新 session,buffer 应为空."""
    buffer = await get_buffer(fresh_session_id)
    assert buffer == []
    assert await get_buffer_size(fresh_session_id) == 0


@pytest.mark.asyncio
async def test_append_then_read(fresh_session_id: uuid.UUID) -> None:
    """追加一条消息,再读出来."""
    await append_to_buffer(fresh_session_id, "user", "你好")
    buffer = await get_buffer(fresh_session_id)
    assert len(buffer) == 1
    assert buffer[0]["role"] == "user"
    assert buffer[0]["content"] == "你好"
    assert "ts" in buffer[0]


@pytest.mark.asyncio
async def test_time_order(fresh_session_id: uuid.UUID) -> None:
    """get_buffer 返回时间正序(先发的在前)."""
    await append_to_buffer(fresh_session_id, "user", "msg1")
    await append_to_buffer(fresh_session_id, "assistant", "msg2")
    await append_to_buffer(fresh_session_id, "user", "msg3")

    buffer = await get_buffer(fresh_session_id)
    assert [m["content"] for m in buffer] == ["msg1", "msg2", "msg3"]


@pytest.mark.asyncio
async def test_auto_trim_over_capacity(fresh_session_id: uuid.UUID) -> None:
    """超过 BUFFER_MAX_ITEMS 时自动驱逐最旧的."""
    # 写入 BUFFER_MAX_ITEMS + 5 条
    for i in range(BUFFER_MAX_ITEMS + 5):
        await append_to_buffer(fresh_session_id, "user", f"msg-{i}")

    buffer = await get_buffer(fresh_session_id)
    assert len(buffer) == BUFFER_MAX_ITEMS

    # 最早的 5 条应该被驱逐(msg-0 ... msg-4)
    # 留下 msg-5 ... msg-(MAX+4)
    assert buffer[0]["content"] == "msg-5"
    assert buffer[-1]["content"] == f"msg-{BUFFER_MAX_ITEMS + 4}"


@pytest.mark.asyncio
async def test_get_oldest_n(fresh_session_id: uuid.UUID) -> None:
    """取最旧 N 条不影响 buffer 内容."""
    for i in range(10):
        await append_to_buffer(fresh_session_id, "user", f"msg-{i}")

    oldest_3 = await get_oldest_n(fresh_session_id, 3)
    assert [m["content"] for m in oldest_3] == ["msg-0", "msg-1", "msg-2"]

    # 不应影响 buffer
    assert await get_buffer_size(fresh_session_id) == 10


@pytest.mark.asyncio
async def test_get_oldest_zero(fresh_session_id: uuid.UUID) -> None:
    """请求 0 条返回空 list."""
    await append_to_buffer(fresh_session_id, "user", "msg")
    assert await get_oldest_n(fresh_session_id, 0) == []


@pytest.mark.asyncio
async def test_get_oldest_more_than_size(fresh_session_id: uuid.UUID) -> None:
    """请求超过实际数量,返回所有."""
    await append_to_buffer(fresh_session_id, "user", "only-msg")
    result = await get_oldest_n(fresh_session_id, 100)
    assert len(result) == 1
    assert result[0]["content"] == "only-msg"


@pytest.mark.asyncio
async def test_trim_oldest_n(fresh_session_id: uuid.UUID) -> None:
    """删最旧 N 条,留下最新的."""
    for i in range(10):
        await append_to_buffer(fresh_session_id, "user", f"msg-{i}")

    deleted = await trim_oldest_n(fresh_session_id, 3)
    assert deleted == 3

    buffer = await get_buffer(fresh_session_id)
    assert len(buffer) == 7
    assert buffer[0]["content"] == "msg-3"
    assert buffer[-1]["content"] == "msg-9"


@pytest.mark.asyncio
async def test_trim_oldest_more_than_size(fresh_session_id: uuid.UUID) -> None:
    """要求删超过实际数量,buffer 清空,返回实际删数."""
    await append_to_buffer(fresh_session_id, "user", "msg")
    deleted = await trim_oldest_n(fresh_session_id, 100)
    assert deleted == 1
    assert await get_buffer(fresh_session_id) == []


@pytest.mark.asyncio
async def test_trim_zero(fresh_session_id: uuid.UUID) -> None:
    """删 0 条 = no-op."""
    await append_to_buffer(fresh_session_id, "user", "msg")
    assert await trim_oldest_n(fresh_session_id, 0) == 0
    assert await get_buffer_size(fresh_session_id) == 1


@pytest.mark.asyncio
async def test_clear_buffer(fresh_session_id: uuid.UUID) -> None:
    """清空 buffer."""
    await append_to_buffer(fresh_session_id, "user", "msg1")
    await append_to_buffer(fresh_session_id, "user", "msg2")

    cleared = await clear_buffer(fresh_session_id)
    assert cleared is True
    assert await get_buffer(fresh_session_id) == []


@pytest.mark.asyncio
async def test_clear_empty_buffer(fresh_session_id: uuid.UUID) -> None:
    """清空本来就空的 buffer 返回 False."""
    cleared = await clear_buffer(fresh_session_id)
    assert cleared is False


@pytest.mark.asyncio
async def test_isolation_between_sessions() -> None:
    """不同 session_id 互不影响."""
    sid_a = uuid.uuid4()
    sid_b = uuid.uuid4()

    try:
        await append_to_buffer(sid_a, "user", "from_a")
        await append_to_buffer(sid_b, "user", "from_b")

        buf_a = await get_buffer(sid_a)
        buf_b = await get_buffer(sid_b)

        assert len(buf_a) == 1 and buf_a[0]["content"] == "from_a"
        assert len(buf_b) == 1 and buf_b[0]["content"] == "from_b"
    finally:
        await clear_buffer(sid_a)
        await clear_buffer(sid_b)
