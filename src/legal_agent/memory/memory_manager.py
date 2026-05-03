"""Memory Manager: 编排 4 个 memory 单元层 (M8 ⭐⭐⭐).

对外暴露 2 个高级 API:
  • inject_memory_into_messages — 喂 LLM 前注入记忆
  • record_turn                 — 一轮对话后更新所有记忆

设计哲学:
  • 不直接持有数据,只编排 4 个单元层
  • Buffer 满才触发 Summary 压缩(最贵操作)
  • Entity 抽取每轮都做(同步等待,M8.0 决策 5: A)
  • 单元层独立可测,本层只关心编排顺序
"""

from __future__ import annotations

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

    Returns:
        str: 拼接好的 markdown 文本.无记忆时返回空串.
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
    """⭐ 构造完整 messages,注入三层记忆.

    Args:
        user_message: 当前用户输入
        user_id: 用户标识
        session_id: 会话标识
        system_prompt: 基础 system prompt(由调用方提供)

    Returns:
        list[dict]: 喂给 LLM 的完整 messages.
                    结构:[system, ...buffer..., user]
    """
    # 1. 拼 system:基础 prompt + facts + summary
    memory_ctx = await build_memory_context(user_id, session_id)
    full_system = f"{system_prompt}\n\n{memory_ctx}" if memory_ctx else system_prompt

    messages: list[dict[str, Any]] = [{"role": "system", "content": full_system}]

    # 2. 展开 buffer 作为多轮历史
    buffer = await get_buffer(session_id)
    for turn in buffer:
        messages.append(
            {
                "role": turn["role"],
                "content": turn["content"],
            }
        )

    # 3. 当前用户消息
    messages.append({"role": "user", "content": user_message})

    return messages


async def record_turn(
    user_id: UUID,
    session_id: UUID,
    user_message: str,
    assistant_reply: str,
) -> dict[str, Any]:
    """⭐ 记录一轮完整对话,触发 entity 抽取 + summary 压缩.

    流程:
        1. user/assistant 写入 Buffer
        2. 同步调 LLM 抽取 entities,写 Hard Memory
        3. 检查 Buffer 是否满 → 触发 Summary 压缩

    Returns:
        dict: 这一轮的元信息(给 M8.7 调试用)
            • entities_extracted: 抽到的 facts 数
            • summary_triggered: 是否压缩了 summary
    """
    # 1. 写 Buffer
    await append_to_buffer(session_id, "user", user_message)
    await append_to_buffer(session_id, "assistant", assistant_reply)

    # 2. 同步抽取 entity(只看这一轮的两条)
    new_msgs_for_entity = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_reply},
    ]
    persisted = await extract_and_persist(user_id, new_msgs_for_entity)

    # 3. 检查 Buffer 是否满,满则压缩
    summary_triggered = False
    buffer_size = await get_buffer_size(session_id)
    if buffer_size >= BUFFER_MAX_ITEMS:
        # 取最旧 N 条压缩
        oldest = await get_oldest_n(session_id, SUMMARY_BATCH_SIZE)
        if oldest:
            await compress_and_update(
                session_id=session_id,
                user_id=user_id,
                new_messages=[{"role": m["role"], "content": m["content"]} for m in oldest],
                turn_count=buffer_size,
            )
            await trim_oldest_n(session_id, len(oldest))
            summary_triggered = True

    return {
        "entities_extracted": len(persisted),
        "summary_triggered": summary_triggered,
        "buffer_size_after": await get_buffer_size(session_id),
    }


__all__ = [
    "SUMMARY_BATCH_SIZE",
    "build_memory_context",
    "inject_memory_into_messages",
    "record_turn",
]
