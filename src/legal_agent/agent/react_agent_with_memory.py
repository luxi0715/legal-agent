"""ReAct Agent + Memory + Persona + Guard 集成 (M8.7 + M10.3 + M10.4).

薄封装:把 M8 Memory + M10 Persona + Guard + M7 ReAct 拼起来.

设计哲学:
  • react_agent.py 几乎不改
  • 本文件不持有任何记忆/Persona 逻辑,只编排
  • persona_mode=None → M8 行为(向后兼容,无 Guard)
  • persona_mode 指定 → M10 行为 + 自动 Guard 检测
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from legal_agent.agent.react_agent import REACT_SYSTEM_PROMPT, run_react_agent
from legal_agent.memory.memory_manager import (
    inject_memory_into_messages,
    record_turn,
)
from legal_agent.persona.guard import log_drift_if_any


async def run_react_agent_with_memory(
    user_message: str,
    user_id: UUID,
    session_id: UUID,
    system_prompt: str | None = None,
    persona_mode: str | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """⭐ 带记忆 + Persona + Guard 的 ReAct Agent."""
    base_prompt = system_prompt or REACT_SYSTEM_PROMPT

    enriched_messages = await inject_memory_into_messages(
        user_message=user_message,
        user_id=user_id,
        session_id=session_id,
        system_prompt=base_prompt,
        persona_mode=persona_mode,
    )

    react_kwargs: dict[str, Any] = {
        "user_message": user_message,
        "initial_messages": enriched_messages,
    }
    if thread_id is not None:
        react_kwargs["thread_id"] = thread_id

    result = await run_react_agent(**react_kwargs)

    # M10.4 — Persona Guard 检测
    if persona_mode is not None:
        guard = log_drift_if_any(
            reply=result["final_reply"],
            persona_mode=persona_mode,
            user_id=str(user_id),
            session_id=str(session_id),
        )
        result["guard"] = guard.to_dict()
    else:
        result["guard"] = {"is_drift": False, "triggered_phrases": [], "severity": "none"}

    memory_meta = await record_turn(
        user_id=user_id,
        session_id=session_id,
        user_message=user_message,
        assistant_reply=result["final_reply"],
    )

    result["memory_meta"] = memory_meta
    result["persona_mode"] = persona_mode or "none"
    return result


__all__ = ["run_react_agent_with_memory"]
