"""单元测试 — Memory Manager (M8.6)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from legal_agent.db.postgres import get_postgres_pool
from legal_agent.memory.buffer_memory import (
    BUFFER_MAX_ITEMS,
    append_to_buffer,
    clear_buffer,
    get_buffer_size,
)
from legal_agent.memory.hard_memory import (
    delete_all_user_facts,
    upsert_user_fact,
)
from legal_agent.memory.memory_manager import (
    build_memory_context,
    inject_memory_into_messages,
    record_turn,
)
from legal_agent.memory.summary_memory import (
    delete_summary,
    get_summary,
    upsert_summary,
)


@pytest_asyncio.fixture
async def fresh_session() -> AsyncIterator[tuple[uuid.UUID, uuid.UUID]]:
    """建一个真实 session(满足外键),返回 (session_id, user_id).

    清理:删 buffer + summary + facts + session.
    """
    user_id = uuid.uuid4()
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "INSERT INTO sessions (user_id, title) VALUES (NULL, 'test') RETURNING id"
        )
        session_id = row["id"]

    try:
        yield session_id, user_id
    finally:
        await clear_buffer(session_id)
        await delete_summary(session_id)
        await delete_all_user_facts(user_id)
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM sessions WHERE id = $1", session_id)


# ─────────── build_memory_context ───────────


@pytest.mark.asyncio
async def test_build_context_empty(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """全空 → 返回空字符串."""
    session_id, user_id = fresh_session
    ctx = await build_memory_context(user_id, session_id)
    assert ctx == ""


@pytest.mark.asyncio
async def test_build_context_facts_only(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """只有 facts → 返回用户档案块."""
    session_id, user_id = fresh_session
    await upsert_user_fact(user_id, "location", "北京")
    await upsert_user_fact(user_id, "occupation", "程序员")

    ctx = await build_memory_context(user_id, session_id)
    assert "用户档案" in ctx
    assert "location: 北京" in ctx
    assert "occupation: 程序员" in ctx
    assert "历史对话摘要" not in ctx


@pytest.mark.asyncio
async def test_build_context_summary_only(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """只有 summary → 返回摘要块."""
    session_id, user_id = fresh_session
    await upsert_summary(session_id, user_id, "用户咨询了劳动纠纷")

    ctx = await build_memory_context(user_id, session_id)
    assert "历史对话摘要" in ctx
    assert "用户咨询了劳动纠纷" in ctx
    assert "用户档案" not in ctx


@pytest.mark.asyncio
async def test_build_context_both(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """facts + summary 都有 → 同时返回."""
    session_id, user_id = fresh_session
    await upsert_user_fact(user_id, "location", "北京")
    await upsert_summary(session_id, user_id, "用户咨询了劳动纠纷")

    ctx = await build_memory_context(user_id, session_id)
    assert "用户档案" in ctx
    assert "历史对话摘要" in ctx


# ─────────── inject_memory_into_messages ───────────


@pytest.mark.asyncio
async def test_inject_no_memory(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """无任何记忆 → 只有 system + 当前 user message."""
    session_id, user_id = fresh_session
    msgs = await inject_memory_into_messages(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
        system_prompt="你是法律助手",
    )
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "你是法律助手"
    assert msgs[1] == {"role": "user", "content": "你好"}


@pytest.mark.asyncio
async def test_inject_with_facts(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """有 facts → system prompt 包含用户档案."""
    session_id, user_id = fresh_session
    await upsert_user_fact(user_id, "location", "北京")

    msgs = await inject_memory_into_messages(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
        system_prompt="你是法律助手",
    )
    assert "你是法律助手" in msgs[0]["content"]
    assert "location: 北京" in msgs[0]["content"]


@pytest.mark.asyncio
async def test_inject_with_buffer(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """有 buffer → 展开成多轮历史."""
    session_id, user_id = fresh_session
    await append_to_buffer(session_id, "user", "之前的问题")
    await append_to_buffer(session_id, "assistant", "之前的回答")

    msgs = await inject_memory_into_messages(
        user_message="新问题",
        user_id=user_id,
        session_id=session_id,
        system_prompt="你是法律助手",
    )
    # system + 2 条历史 + 当前 user = 4 条
    assert len(msgs) == 4
    assert msgs[1] == {"role": "user", "content": "之前的问题"}
    assert msgs[2] == {"role": "assistant", "content": "之前的回答"}
    assert msgs[3] == {"role": "user", "content": "新问题"}


# ─────────── record_turn ───────────


@pytest.mark.asyncio
async def test_record_turn_appends_to_buffer(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """⭐ 一轮对话写入 Buffer."""
    session_id, user_id = fresh_session
    result = await record_turn(
        user_id=user_id,
        session_id=session_id,
        user_message="我在北京",
        assistant_reply="好的,记住了",
    )

    assert await get_buffer_size(session_id) == 2
    # 不触发 summary(buffer 才 2 条)
    assert result["summary_triggered"] is False


@pytest.mark.asyncio
async def test_record_turn_triggers_summary(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """⭐ Buffer 满 → 触发 Summary 压缩."""
    session_id, user_id = fresh_session

    # 预填 BUFFER_MAX_ITEMS - 2 条(直接 append,不触发抽取省时间)
    for i in range(BUFFER_MAX_ITEMS - 2):
        await append_to_buffer(
            session_id,
            "user" if i % 2 == 0 else "assistant",
            f"老消息 {i}",
        )

    # record_turn 再加 2 条 → buffer 正好 = MAX → 触发
    result = await record_turn(
        user_id=user_id,
        session_id=session_id,
        user_message="触发摘要的消息",
        assistant_reply="收到",
    )

    assert result["summary_triggered"] is True
    # Summary 写入了
    summary = await get_summary(session_id)
    assert len(summary) > 0
    # Buffer 被 trim
    final_size = await get_buffer_size(session_id)
    assert final_size < BUFFER_MAX_ITEMS
