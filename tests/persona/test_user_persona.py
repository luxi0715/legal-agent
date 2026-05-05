"""User Persona Builder 单测 (M10.2)."""

from __future__ import annotations

from legal_agent.persona.user_persona import build_user_persona_text


def test_empty_facts() -> None:
    """空 facts → 空字符串."""
    assert build_user_persona_text({}) == ""


def test_only_location() -> None:
    """只有 location → 简单画像 + 无沟通建议."""
    text = build_user_persona_text({"location": "上海"})
    assert "上海" in text
    assert "用户画像" in text
    # 没有职业 / 家庭信息 → 无沟通建议
    assert "沟通建议" not in text


def test_location_and_occupation() -> None:
    """location + occupation → 画像 + 教师沟通建议."""
    text = build_user_persona_text(
        {
            "location": "上海",
            "occupation": "教师",
        }
    )
    assert "上海" in text
    assert "教师" in text
    assert "沟通建议" in text
    assert "教育背景" in text


def test_full_facts() -> None:
    """完整 facts → 完整画像."""
    text = build_user_persona_text(
        {
            "location": "北京",
            "occupation": "程序员",
            "age_range": "25-30",
            "family_status": "已婚有娃",
            "legal_concern_type": "劳动纠纷",
        }
    )
    assert "北京" in text
    assert "程序员" in text
    assert "25-30" in text
    assert "已婚有娃" in text
    assert "劳动纠纷" in text
    # 程序员沟通建议
    assert "逻辑思维" in text or "结构化" in text
    # 已婚有娃关怀维度
    assert "子女权益" in text


def test_lawyer_no_explain_basics() -> None:
    """职业=律师 → 不需要解释基础概念."""
    text = build_user_persona_text({"occupation": "律师"})
    assert "无需解释基础概念" in text or "法律术语" in text


def test_unknown_occupation_no_hint() -> None:
    """未知职业 → 没有沟通建议(只输出事实)."""
    text = build_user_persona_text({"occupation": "宇航员"})
    assert "宇航员" in text
    assert "沟通建议" not in text


def test_empty_string_values_skipped() -> None:
    """空字符串值应被跳过."""
    text = build_user_persona_text(
        {
            "location": "上海",
            "occupation": "  ",  # 全空白
            "age_range": "",
        }
    )
    assert "上海" in text
    # 空白职业不应出现在文本里
    lines = text.split("\n")
    factual_line = lines[0]
    assert "职业" not in factual_line


def test_output_contains_user_persona_label() -> None:
    """输出格式以\"用户画像:\"开头."""
    text = build_user_persona_text({"location": "上海"})
    assert text.startswith("用户画像")
