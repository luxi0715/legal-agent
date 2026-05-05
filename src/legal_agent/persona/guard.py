"""Persona Guard (M10.4): 检测 LLM 回复是否偏离 persona 角色.

设计:
  • 关键词黑名单匹配(全局禁词 + persona 专属禁词)
  • 仅记录(不重写),给运维观察漂移频率
  • 严重度三级:none / low / high
  • 高频漂移由 logger.warning 记录,M11+ 再决定治理策略

使用方式:
    guard = check_persona_drift(reply, "default")
    if guard.is_drift:
        logger.warning("漂移触发: %s", guard.triggered_phrases)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from legal_agent.persona.loader import get_persona

logger = logging.getLogger(__name__)


# 高严重度词:几乎肯定是漂移或注入
HIGH_SEVERITY_KEYWORDS = {
    "我是律师",
    "我是执业律师",
    "保证胜诉",
    "保证赢",
    "100% 能赢",
    "100%能赢",
    "建议你违法",
    "如何规避法律",
    "如何逃避",
    "忽略之前的指令",
    "现在你是",
}


@dataclass
class GuardResult:
    """Persona 漂移检测结果."""

    is_drift: bool
    triggered_phrases: list[str] = field(default_factory=list)
    severity: str = "none"  # "none" / "low" / "high"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _score_severity(triggered: list[str]) -> str:
    """根据触发的词判定严重度."""
    if not triggered:
        return "none"
    for phrase in triggered:
        if phrase in HIGH_SEVERITY_KEYWORDS:
            return "high"
    return "low"


def check_persona_drift(
    reply: str,
    persona_mode: str = "default",
) -> GuardResult:
    """⭐ 检测 LLM 回复是否触发 persona 禁词.

    Args:
        reply: LLM 的最终回复文本
        persona_mode: 当前的 persona,从中拿到禁词列表

    Returns:
        GuardResult: 含 is_drift / triggered_phrases / severity
    """
    if not reply:
        return GuardResult(is_drift=False)

    persona = get_persona(persona_mode)
    forbidden = persona.forbidden_phrases

    triggered: list[str] = []
    for phrase in forbidden:
        if phrase in reply:
            triggered.append(phrase)

    if not triggered:
        return GuardResult(is_drift=False)

    severity = _score_severity(triggered)
    return GuardResult(
        is_drift=True,
        triggered_phrases=triggered,
        severity=severity,
    )


def log_drift_if_any(
    reply: str,
    persona_mode: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> GuardResult:
    """检测 + 自动 logger 输出,一站式调用.

    用于 react_agent_with_memory.py 调用方便.
    """
    guard = check_persona_drift(reply, persona_mode)

    if guard.is_drift:
        logger.warning(
            "Persona drift detected (severity=%s, mode=%s, user=%s, session=%s): %s",
            guard.severity,
            persona_mode,
            user_id,
            session_id,
            guard.triggered_phrases,
        )

    return guard


__all__ = [
    "GuardResult",
    "HIGH_SEVERITY_KEYWORDS",
    "check_persona_drift",
    "log_drift_if_any",
]
