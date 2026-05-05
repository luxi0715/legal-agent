"""Memory Manager: 编排 4 个 memory 单元层 (M8 + M9.1 + M10.3).

对外暴露 2 个高级 API:
  • inject_memory_into_messages — 喂 LLM 前注入记忆 + Persona
  • record_turn                 — 一轮对话后更新所有记忆
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
from legal_agent.persona.loader import build_persona_system_prompt, get_persona
from legal_agent.persona.user_persona import build_user_persona_text

logger = logging.getLogger(__name__)

SUMMARY_BATCH_SIZE = 6


def _format_facts(facts: dict[str, str]) -> str:
    """把 user_facts dict 格式化成 prompt 友好文本(M8 兼容)."""
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
    use_user_persona: bool = False,
) -> str:
    """构建 facts/persona + summary 文本块.

    Args:
        use_user_persona: M10.3 — True 时用自然语言画像替代原始 KV.
    """
    facts = await get_user_facts(user_id)
    summary = await get_summary(session_id)

    parts = []
    if facts:
        if use_user_persona:
            persona_text = build_user_persona_text(facts)
            if persona_text:
                parts.append(f"## 用户画像\n{persona_text}")
        else:
            parts.append(_format_facts(facts))
    if summary:
        parts.append(_format_summary(summary))

    return "\n\n".join(parts)


async def inject_memory_into_messages(
    user_message: str,
    user_id: UUID,
    session_id: UUID,
    system_prompt: str,
    persona_mode: str | None = None,
) -> list[dict[str, Any]]:
    """⭐ 构造完整 messages,注入 Persona + 三层记忆.

    Args:
        persona_mode: M10.3 — None 时走 M8 老路(向后兼容).
                     非 None 时用 persona 替换 system_prompt + 用画像替代 KV.
    """
    if persona_mode is not None:
        persona = get_persona(persona_mode)
        full_system_base = build_persona_system_prompt(persona)
        memory_ctx = await build_memory_context(
            user_id, session_id, use_user_persona=True
        )
    else:
        full_system_base = system_prompt
        memory_ctx = await build_memory_context(
            user_id, session_id, use_user_persona=False
        )

    full_system = (
        f"{full_system_base}\n\n{memory_ctx}" if memory_ctx else full_system_base
    )

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
    """⭐ M9.1 — 后台 entity 抽取任务,错误隔离."""
    try:
        persisted = await extract_and_persist(user_id, new_msgs)
        logger.info(
            "Entity extraction (async) succeeded for user_id=%s: %d facts",
            user_id, len(persisted),
        )
    except Exception:
        logger.exception(
            "Entity extraction (async) failed for user_id=%s",
            user_id,
        )


async def record_turn(
    user_id: UUID,
    session_id: UUID,
    user_message: str,
    assistant_reply: str,
) -> dict[str, Any]:
    """⭐ 记录一轮对话(M9.1: entity 异步)."""
    await append_to_buffer(session_id, "user", user_message)
    await append_to_buffer(session_id, "assistant", assistant_reply)

    new_msgs_for_entity = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_reply},
    ]
    asyncio.create_task(_async_extract_entities(user_id, new_msgs_for_entity))

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
        "entities_extracted": "async",
        "summary_triggered": summary_triggered,
        "buffer_size_after": await get_buffer_size(session_id),
    }


__all__ = [
    "SUMMARY_BATCH_SIZE",
    "build_memory_context",
    "inject_memory_into_messages",
    "record_turn",
]
