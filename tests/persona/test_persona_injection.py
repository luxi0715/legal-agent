"""M10.3 集成测试 — Persona 注入到 inject_memory_into_messages."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from legal_agent.db.postgres import get_postgres_pool
from legal_agent.memory.buffer_memory import clear_buffer
from legal_agent.memory.hard_memory import (
    delete_all_user_facts,
    upsert_user_fact,
)
from legal_agent.memory.memory_manager import inject_memory_into_messages
from legal_agent.memory.summary_memory import delete_summary


@pytest_asyncio.fixture
async def fresh_session() -> AsyncIterator[tuple[uuid.UUID, uuid.UUID]]:
    """建一个真实 session,返回 (session_id, user_id)."""
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


# ─────────── persona_mode=None (M8 兼容路径) ───────────


@pytest.mark.asyncio
async def test_no_persona_mode_uses_system_prompt(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """persona_mode=None → 用传入的 system_prompt."""
    session_id, user_id = fresh_session

    msgs = await inject_memory_into_messages(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
        system_prompt="你是法律助手",
        persona_mode=None,
    )

    assert msgs[0]["role"] == "system"
    assert "你是法律助手" in msgs[0]["content"]


@pytest.mark.asyncio
async def test_no_persona_mode_uses_raw_kv_facts(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """persona_mode=None + 有 facts → 输出原始 KV 格式(M8 兼容)."""
    session_id, user_id = fresh_session
    await upsert_user_fact(user_id, "location", "北京")

    msgs = await inject_memory_into_messages(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
        system_prompt="你是法律助手",
        persona_mode=None,
    )
    sys_content = msgs[0]["content"]
    # M8 格式
    assert "用户档案" in sys_content
    assert "- location: 北京" in sys_content
    # 不应该有 M10 画像格式
    assert "用户画像" not in sys_content


# ─────────── persona_mode 启用 (M10.3 路径) ───────────


@pytest.mark.asyncio
async def test_default_persona_replaces_system_prompt(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """persona_mode='default' → 用 default persona 替换 system_prompt."""
    session_id, user_id = fresh_session

    msgs = await inject_memory_into_messages(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
        system_prompt="任何旧 prompt(应该被忽略)",
        persona_mode="default",
    )
    sys_content = msgs[0]["content"]

    # default persona 的标识
    assert "# 你的角色" in sys_content
    assert "# 回答风格" in sys_content
    assert "法律顾问" in sys_content
    # 旧的 system_prompt 不应出现
    assert "应该被忽略" not in sys_content


@pytest.mark.asyncio
async def test_strict_persona_distinct_from_friendly(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """不同 persona_mode 输出明显不同的 system prompt."""
    session_id, user_id = fresh_session

    msgs_strict = await inject_memory_into_messages(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
        system_prompt="",
        persona_mode="strict",
    )
    msgs_friendly = await inject_memory_into_messages(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
        system_prompt="",
        persona_mode="friendly",
    )

    s_content = msgs_strict[0]["content"]
    f_content = msgs_friendly[0]["content"]

    # 至少应有差异
    assert s_content != f_content
    # strict 的关键词
    assert "严谨" in s_content or "克制" in s_content
    # friendly 的关键词
    assert "温和" in f_content or "亲切" in f_content


@pytest.mark.asyncio
async def test_persona_mode_uses_user_persona_text(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """persona_mode 启用 + 有 facts → 用自然语言画像(不是 KV)."""
    session_id, user_id = fresh_session
    await upsert_user_fact(user_id, "location", "上海")
    await upsert_user_fact(user_id, "occupation", "教师")

    msgs = await inject_memory_into_messages(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
        system_prompt="",
        persona_mode="default",
    )
    sys_content = msgs[0]["content"]

    # M10.3 格式
    assert "用户画像" in sys_content
    assert "上海" in sys_content
    assert "教师" in sys_content
    # 不应该出现 M8 的 KV 格式
    assert "- location: 上海" not in sys_content
    assert "- occupation: 教师" not in sys_content


@pytest.mark.asyncio
async def test_unknown_persona_falls_back_to_default(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """未知 persona_mode → loader 内部降级 default."""
    session_id, user_id = fresh_session

    msgs = await inject_memory_into_messages(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
        system_prompt="",
        persona_mode="nonexistent_mode",
    )
    sys_content = msgs[0]["content"]

    # 应该走到 default
    assert "# 你的角色" in sys_content
    assert "法律顾问" in sys_content
