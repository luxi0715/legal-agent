"""Memory Manager: 编排 4 个 memory 单元层 (M8 ⭐⭐⭐ + M9.1 异步化).

对外暴露 2 个高级 API:
  • inject_memory_into_messages — 喂 LLM 前注入记忆
  • record_turn                 — 一轮对话后更新所有记忆

设计哲学:
  • 不直接持有数据,只编排 4 个单元层
  • Buffer 满才触发 Summary 压缩(最贵操作)
  • Entity 抽取异步执行(M9.1 — 用 asyncio.create_task fire-and-forget)
  • 单元层独立可测,本层只关心编排顺序

M9.1 改动:
  • Entity 抽取从同步 await 改为 asyncio.create_task
  • record_turn 立刻返回,不等 entity 抽取完成
  • 牺牲实时一致性,换主流程不阻塞
  • Trade-off:返回值的 entities_extracted 改为 "async" 字符串
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import UUID

from legal_agent.memory.buffer_memory import (
    BUFFER_MAX_ITEMS,
    append_to_buffer,
    get_buffer,
    get_buffer_size,
    get_oldest_n,
    trim_oldest_n,
)
from legal_agent.memory.entity_extractor import extract_and_persist
from legal_agent.memory.hard_memory import get_user_facts
from legal_agent.memory.summary_memory import compress_and_update, get_summary

logger = logging.getLogger(__name__)

# Buffer 满时,压缩多少条进 Summary
SUMMARY_BATCH_SIZE = 6


def _format_facts(facts: dict[str, str]) -> str:
    """把 user_facts dict 格式化成 prompt 友好文本."""
    if not facts:
        return ""
    lines = "\n".join(f"- {k}: {v}" for k, v in facts.items())
    return f"## 用户档案\n{lines}"


def _format_summary(summary: str) -> str:
    """把 summary 格式化成 prompt 友好文本."""
    if not summary:
        return ""
    return f"## 历史对话摘要\n{summary}"


async def build_memory_context(
    user_id: UUID,
    session_id: UUID,
) -> str:
    """构建 facts + summary 文本块(给 system prompt 用).

    不包含 buffer — buffer 在 messages 里展开.
    """
    facts = await get_user_facts(user_id)
    summary = await get_summary(session_id)

    parts = []
    if facts:
        parts.append(_format_facts(facts))
    if summary:
        parts.append(_format_summary(summary))

    return "\n\n".join(parts)


async def inject_memory_into_messages(
    user_message: str,
    user_id: UUID,
    session_id: UUID,
    system_prompt: str,
) -> list[dict[str, Any]]:
    """⭐ 构造完整 messages,注入三层记忆."""
    memory_ctx = await build_memory_context(user_id, session_id)
    full_system = f"{system_prompt}\n\n{memory_ctx}" if memory_ctx else system_prompt

    messages: list[dict[str, Any]] = [{"role": "system", "content": full_system}]

    buffer = await get_buffer(session_id)
    for turn in buffer:
        messages.append(
            {
                "role": turn["role"],
                "content": turn["content"],
            }
        )

    messages.append({"role": "user", "content": user_message})

    return messages


async def _async_extract_entities(
    user_id: UUID,
    new_msgs: list[dict[str, str]],
) -> None:
    """⭐ M9.1 — 后台 entity 抽取任务.

    错误隔离:任何异常都被 swallow,不影响主流程.
    出错只记 logger,因为没人 await 这个 task.
    """
    try:
        persisted = await extract_and_persist(user_id, new_msgs)
        logger.info(
            "Entity extraction (async) succeeded for user_id=%s: %d facts",
            user_id, len(persisted),
        )
    except Exception:
        logger.exception(
            "Entity extraction (async) failed for user_id=%s — silently dropped",
            user_id,
        )


async def record_turn(
    user_id: UUID,
    session_id: UUID,
    user_message: str,
    assistant_reply: str,
) -> dict[str, Any]:
    """⭐ 记录一轮完整对话(M9.1: entity 抽取异步化).

    流程:
        1. user/assistant 写入 Buffer(同步,主流程必须)
        2. Entity 抽取 异步发出 — 立刻返回不等
        3. 检查 Buffer 是否满 → 触发 Summary 压缩

    Returns:
        dict: 这一轮的元信息
            • entities_extracted: "async" — 异步执行,实际值未知
            • summary_triggered: 是否压缩了 summary
            • buffer_size_after: buffer 当前大小
    """
    # 1. 写 Buffer(主流程,必须等)
    await append_to_buffer(session_id, "user", user_message)
    await append_to_buffer(session_id, "assistant", assistant_reply)

    # 2. ⭐ M9.1 — 异步抽 entity,fire-and-forget
    new_msgs_for_entity = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_reply},
    ]
    asyncio.create_task(_async_extract_entities(user_id, new_msgs_for_entity))

    # 3. 检查 Buffer 是否满,满则压缩(主流程,因为下一轮 inject 要用)
    summary_triggered = False
    buffer_size = await get_buffer_size(session_id)
    if buffer_size >= BUFFER_MAX_ITEMS:
        oldest = await get_oldest_n(session_id, SUMMARY_BATCH_SIZE)
        if oldest:
            await compress_and_update(
                session_id=session_id,
                user_id=user_id,
                new_messages=[
                    {"role": m["role"], "content": m["content"]} for m in oldest
                ],
                turn_count=buffer_size,
            )
            await trim_oldest_n(session_id, len(oldest))
            summary_triggered = True

    return {
        "entities_extracted": "async",  # ⚠️ M9.1 异步化,无法立刻知道数量
        "summary_triggered": summary_triggered,
        "buffer_size_after": await get_buffer_size(session_id),
    }


__all__ = [
    "SUMMARY_BATCH_SIZE",
    "build_memory_context",
    "inject_memory_into_messages",
    "record_turn",
]
