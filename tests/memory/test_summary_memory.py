"""单元测试 — Summary Memory (M8.4)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from legal_agent.db.postgres import get_postgres_pool
from legal_agent.memory.summary_memory import (
    compress_and_update,
    delete_summary,
    get_summary,
    upsert_summary,
)


@pytest_asyncio.fixture
async def fresh_session() -> AsyncIterator[tuple[uuid.UUID, uuid.UUID]]:
    """每个测试:建一个真实 session(满足外键),返回 (session_id, user_id).

    清理:删 summary + 删 session.
    """
    user_id = uuid.uuid4()
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        # 建一个临时 session(user_id 字段 BIGINT,这里塞 NULL 即可)
        row = await conn.fetchrow(
            "INSERT INTO sessions (user_id, title) VALUES (NULL, 'test') RETURNING id"
        )
        session_id = row["id"]

    try:
        yield session_id, user_id
    finally:
        # 删 summary(如果存在)
        await delete_summary(session_id)
        # 删 session(级联会清掉 summary,双保险)
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)


@pytest.mark.asyncio
async def test_get_empty_summary(fresh_session: tuple[uuid.UUID, uuid.UUID]) -> None:
    """新 session 没 summary,返回空字符串."""
    session_id, _ = fresh_session
    summary = await get_summary(session_id)
    assert summary == ""


@pytest.mark.asyncio
async def test_upsert_then_get(fresh_session: tuple[uuid.UUID, uuid.UUID]) -> None:
    """写入 summary,再读出来."""
    session_id, user_id = fresh_session
    await upsert_summary(session_id, user_id, "用户咨询了拖欠工资问题")
    assert await get_summary(session_id) == "用户咨询了拖欠工资问题"


@pytest.mark.asyncio
async def test_upsert_overwrite(fresh_session: tuple[uuid.UUID, uuid.UUID]) -> None:
    """同 session 重复 upsert,新值覆盖旧值."""
    session_id, user_id = fresh_session
    await upsert_summary(session_id, user_id, "v1")
    await upsert_summary(session_id, user_id, "v2")
    assert await get_summary(session_id) == "v2"


@pytest.mark.asyncio
async def test_delete_existing(fresh_session: tuple[uuid.UUID, uuid.UUID]) -> None:
    """删除存在的 summary 返回 True."""
    session_id, user_id = fresh_session
    await upsert_summary(session_id, user_id, "test")
    assert await delete_summary(session_id) is True
    assert await get_summary(session_id) == ""


@pytest.mark.asyncio
async def test_delete_nonexistent(fresh_session: tuple[uuid.UUID, uuid.UUID]) -> None:
    """删除不存在的 summary 返回 False."""
    session_id, _ = fresh_session
    assert await delete_summary(session_id) is False


@pytest.mark.asyncio
async def test_compress_first_round(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """⭐ 调真实 LLM 压缩首轮对话."""
    session_id, user_id = fresh_session
    new_msgs = [
        {"role": "user", "content": "我在北京做程序员,被老板拖欠了 2 个月工资"},
        {
            "role": "assistant",
            "content": "建议先收集劳动合同、工资流水作为证据,然后向劳动监察大队投诉.",
        },
        {"role": "user", "content": "我没有书面合同怎么办?"},
        {
            "role": "assistant",
            "content": "可以用工资流水、聊天记录、考勤记录、同事证言等证明事实劳动关系.",
        },
    ]

    summary = await compress_and_update(
        session_id=session_id,
        user_id=user_id,
        new_messages=new_msgs,
        turn_count=2,
    )

    # LLM 压缩后非空
    assert len(summary) > 0
    # 长度合理(< 500 字 + 一些 buffer)
    assert len(summary) < 800
    # 数据库里读出来一致
    assert await get_summary(session_id) == summary


@pytest.mark.asyncio
async def test_compress_empty_messages(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """空消息列表:不调 LLM,返回当前 summary."""
    session_id, user_id = fresh_session
    await upsert_summary(session_id, user_id, "已有内容")

    result = await compress_and_update(
        session_id=session_id,
        user_id=user_id,
        new_messages=[],
    )
    assert result == "已有内容"


@pytest.mark.asyncio
async def test_compress_rolling_update(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """⭐ 滚动压缩:已有 summary + 新对话 → 融合新 summary."""
    session_id, user_id = fresh_session
    # 第一次压缩
    await compress_and_update(
        session_id=session_id,
        user_id=user_id,
        new_messages=[
            {"role": "user", "content": "我是北京的程序员"},
            {"role": "assistant", "content": "好的,记住了."},
        ],
        turn_count=1,
    )
    summary_v1 = await get_summary(session_id)
    assert len(summary_v1) > 0

    # 第二次压缩(基于 v1)
    summary_v2 = await compress_and_update(
        session_id=session_id,
        user_id=user_id,
        new_messages=[
            {"role": "user", "content": "我现在被拖欠工资,求助"},
            {"role": "assistant", "content": "建议向劳动监察投诉."},
        ],
        turn_count=2,
    )

    # 新摘要不为空,且与 v1 不同(因为合并了新内容)
    assert len(summary_v2) > 0
    assert summary_v2 != summary_v1
