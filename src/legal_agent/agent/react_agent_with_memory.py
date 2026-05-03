"""ReAct Agent + Memory 集成 (M8.7).

薄封装:把 M8 Memory Manager 编排层和 M7 ReAct Agent 拼起来.

设计哲学:
  • react_agent.py 几乎不改(只加 initial_messages 旁路参数)
  • 本文件 不持有 任何记忆逻辑,只编排
  • 记忆生命周期完全由 Memory Manager 管理
  • 体现 \"M7 推理 + M8 记忆\" 通过 messages 接口解耦
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from legal_agent.agent.react_agent import REACT_SYSTEM_PROMPT, run_react_agent
from legal_agent.memory.memory_manager import (
    inject_memory_into_messages,
    record_turn,
)


async def run_react_agent_with_memory(
    user_message: str,
    user_id: UUID,
    session_id: UUID,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """⭐ 带三层记忆的 ReAct Agent.

    Args:
        user_message: 当前用户输入
        user_id: 用户标识(M8 起步 = session.id 派生)
        session_id: 会话标识
        system_prompt: 基础 system prompt(默认用 REACT_SYSTEM_PROMPT)

    Returns:
        dict 包含原 run_react_agent 全部字段 + memory_meta:
            • final_reply: str
            • iterations: int
            • tool_calls_log: list[dict]
            • memory_meta: dict — entities_extracted / summary_triggered / buffer_size_after

    流程:
        1. Memory Manager 注入记忆 → 完整 messages
        2. 调原 ReAct graph 跑推理(走 initial_messages 旁路)
        3. record_turn 更新所有记忆(buffer / entities / summary)
    """
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT

    # 1. 注入记忆,得到完整 messages
    enriched_messages = await inject_memory_into_messages(
        user_message=user_message,
        user_id=user_id,
        session_id=session_id,
        system_prompt=base_prompt,
    )

    # 2. 跑 ReAct(用 initial_messages 旁路绕过内部组装)
    result = await run_react_agent(
        user_message=user_message,
        initial_messages=enriched_messages,
    )

    # 3. 更新记忆
    memory_meta = await record_turn(
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        assistant_reply=result["final_reply"],
    )

    result["memory_meta"] = memory_meta
    return result


__all__ = ["run_react_agent_with_memory"]
