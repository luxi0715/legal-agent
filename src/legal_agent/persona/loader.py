"""Persona Loader (M10.1): 从 yaml 加载 5 套 Agent persona.

设计:
  • 模块加载时一次性读取(yaml 不会运行时变)
  • 每个 persona 包含 name / description / style / forbidden_phrases
  • _global_forbidden_phrases 合并到每个 persona 的禁词
  • 找不到 mode 时默认 default,记 logger 不抛异常
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class AgentPersona:
    """单个 Agent Persona 配置."""

    mode: str
    name: str
    description: str
    style: str
    forbidden_phrases: list[str] = field(default_factory=list)


PERSONAS_FILE = Path(__file__).parent / "personas.yaml"

_personas: dict[str, AgentPersona] = {}
_default_mode = "default"


def _load_personas() -> dict[str, AgentPersona]:
    """从 yaml 加载所有 persona,首次调用时执行."""
    if _personas:
        return _personas

    if not PERSONAS_FILE.exists():
        raise FileNotFoundError(f"personas.yaml not found at {PERSONAS_FILE}")

    with open(PERSONAS_FILE, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    global_forbidden: list[str] = raw.pop("_global_forbidden_phrases", [])

    for mode, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        persona_forbidden = list(cfg.get("forbidden_phrases", []))
        merged_forbidden = list(set(global_forbidden + persona_forbidden))

        _personas[mode] = AgentPersona(
            mode=mode,
            name=cfg["name"],
            description=cfg["description"],
            style=cfg["style"],
            forbidden_phrases=merged_forbidden,
        )

    logger.info("Loaded %d personas: %s", len(_personas), list(_personas.keys()))
    return _personas


def get_persona(mode: str | None = None) -> AgentPersona:
    """获取指定 mode 的 persona,找不到则降级 default."""
    personas = _load_personas()

    if mode is None or mode not in personas:
        if mode is not None:
            logger.warning("Unknown persona mode %r, fallback to default", mode)
        return personas[_default_mode]

    return personas[mode]


def list_persona_modes() -> list[str]:
    """列出所有可用的 persona mode."""
    return list(_load_personas().keys())


def build_persona_system_prompt(persona: AgentPersona) -> str:
    """把 AgentPersona 拼成 system prompt 文本."""
    return f"# 你的角色\n{persona.description}\n\n# 回答风格\n{persona.style}"


__all__ = [
    "AgentPersona",
    "get_persona",
    "list_persona_modes",
    "build_persona_system_prompt",
]
