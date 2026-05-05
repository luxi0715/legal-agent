"""集成测试 — ReAct + Memory (M8.7 + M9.1 异步化).

端到端测试:验证记忆真的生效.
M9.1 后:entity 抽取异步,测试需 sleep 等任务完成 + 直接查 user_facts 表.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from legal_agent.agent.react_agent_with_memory import run_react_agent_with_memory
from legal_agent.db.postgres import get_postgres_pool
from legal_agent.memory.buffer_memory import clear_buffer
from legal_agent.memory.hard_memory import (
    delete_all_user_facts,
    get_user_facts,
)
from legal_agent.memory.summary_memory import delete_summary


# M9.1: 异步 entity 抽取 ~10-25s(LLM 调用),给宽裕时间
ASYNC_EXTRACT_WAIT_SEC = 30


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


@pytest.mark.asyncio
async def test_first_turn_basic_flow(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """⭐ 首轮对话:能正常跑通,记忆元信息正确返回."""
    session_id, user_id = fresh_session

    result = await run_react_agent_with_memory(
        user_message="你好",
        user_id=user_id,
        session_id=session_id,
    )

    assert "final_reply" in result
    assert len(result["final_reply"]) > 0
    assert "iterations" in result
    assert "memory_meta" in result
    # M9.1: entities_extracted 改为 "async" 字符串
    assert result["memory_meta"]["entities_extracted"] == "async"
    # 首轮 buffer 应有 2 条(user + assistant)
    assert result["memory_meta"]["buffer_size_after"] == 2


@pytest.mark.asyncio
async def test_entity_extracted_into_facts(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """⭐ 用户陈述身份 → entity 异步抽取写入 user_facts.

    M9.1: 抽取异步,需 sleep 等任务完成,再查 user_facts 表.
    """
    session_id, user_id = fresh_session

    await run_react_agent_with_memory(
        user_message="我在北京做程序员,想咨询劳动法相关问题",
        user_id=user_id,
        session_id=session_id,
    )

    # ⭐ M9.1: 等异步抽取完成
    await asyncio.sleep(ASYNC_EXTRACT_WAIT_SEC)

    facts = await get_user_facts(user_id)
    assert len(facts) >= 1, f"期望至少 1 个 fact,实际 {facts}"


@pytest.mark.asyncio
async def test_multi_turn_context_continuity(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """⭐ 多轮对话:第二轮 LLM 能看到第一轮的 buffer 历史.

    Buffer 写入是同步的,所以这个测试不受 M9.1 影响.
    """
    session_id, user_id = fresh_session

    await run_react_agent_with_memory(
        user_message="我刚被公司辞退,没有书面合同",
        user_id=user_id,
        session_id=session_id,
    )

    result2 = await run_react_agent_with_memory(
        user_message="那我该怎么维权?",
        user_id=user_id,
        session_id=session_id,
    )

    reply = result2["final_reply"]
    assert any(
        kw in reply
        for kw in ["辞退", "解雇", "劳动", "合同", "仲裁", "事实劳动关系", "工资"]
    ), f"第二轮没解析'那'的指代,回复:{reply[:200]}"


@pytest.mark.asyncio
async def test_facts_persist_across_turns(
    fresh_session: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """⭐ 第一轮抽到的 facts,第二轮注入到 system prompt.

    M9.1: 第一轮抽取异步,sleep 等完成后再查 + 跑第二轮.
    """
    session_id, user_id = fresh_session

    await run_react_agent_with_memory(
        user_message="我是上海的老师,30 岁",
        user_id=user_id,
        session_id=session_id,
    )

    # ⭐ M9.1: 等第一轮的异步抽取完成
    await asyncio.sleep(ASYNC_EXTRACT_WAIT_SEC)

    facts = await get_user_facts(user_id)
    assert len(facts) >= 1, f"facts 应被抽取并持久化: {facts}"

    result = await run_react_agent_with_memory(
        user_message="我能问个法律问题吗?",
        user_id=user_id,
        session_id=session_id,
    )

    assert len(result["final_reply"]) > 0
