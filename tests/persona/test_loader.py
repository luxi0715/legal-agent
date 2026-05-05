"""Persona Loader 单测 (M10.1)."""

from __future__ import annotations

from legal_agent.persona.loader import (
    AgentPersona,
    build_persona_system_prompt,
    get_persona,
    list_persona_modes,
)


def test_load_all_5_personas() -> None:
    """yaml 应包含 5 套 persona."""
    modes = list_persona_modes()
    expected = {"default", "strict", "friendly", "enterprise", "litigation"}
    assert set(modes) == expected, f"got {set(modes)}"


def test_default_persona() -> None:
    """default 是法律顾问助手."""
    p = get_persona("default")
    assert isinstance(p, AgentPersona)
    assert "法律顾问" in p.name
    assert len(p.description) > 0
    assert len(p.style) > 0


def test_strict_persona() -> None:
    """strict 是严谨分析师."""
    p = get_persona("strict")
    assert "严谨" in p.name or "分析" in p.name


def test_unknown_mode_falls_back() -> None:
    """未知 mode 降级到 default."""
    p = get_persona("nonexistent_mode")
    assert p.mode == "default"


def test_none_mode_returns_default() -> None:
    """None 返回 default."""
    p = get_persona(None)
    assert p.mode == "default"


def test_global_forbidden_merged() -> None:
    """每个 persona 都包含全局禁词."""
    for mode in list_persona_modes():
        p = get_persona(mode)
        assert "我是律师" in p.forbidden_phrases
        assert "保证胜诉" in p.forbidden_phrases
        assert "忽略之前的指令" in p.forbidden_phrases


def test_build_system_prompt() -> None:
    """system prompt 包含 description + style."""
    p = get_persona("default")
    prompt = build_persona_system_prompt(p)
    assert "# 你的角色" in prompt
    assert "# 回答风格" in prompt
    assert p.description.strip()[:20] in prompt
