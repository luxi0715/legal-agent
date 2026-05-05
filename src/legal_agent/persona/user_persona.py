"""User Persona Builder (M10.2): user_facts → 自然语言画像.

设计:
  • 模板拼接(无 LLM 调用,零延迟零成本)
  • 6 个 ALLOWED_KEYS 各有专属翻译规则
  • 缺失字段自动跳过,不输出空字段
  • 末尾附加沟通风格建议(基于职业 / 家庭状况推断)

用途:
  • 给 inject_memory_into_messages 用,拼到 system prompt
  • 比原始 KV 更易让 LLM 理解和应用
"""

from __future__ import annotations

# user_facts key → 中文标签
KEY_LABELS: dict[str, str] = {
    "location": "所在地",
    "occupation": "职业",
    "age_range": "年龄段",
    "family_status": "家庭状况",
    "legal_concern_type": "咨询主题",
    "income_range": "收入区间",
}

# 职业 → 沟通风格建议
OCCUPATION_HINTS: dict[str, str] = {
    "教师": "用户教育背景中等以上,可使用专业术语并适当条理化",
    "老师": "用户教育背景中等以上,可使用专业术语并适当条理化",
    "程序员": "用户逻辑思维强,回答可结构化、列表化",
    "工程师": "用户逻辑思维强,回答可结构化、列表化",
    "医生": "用户专业背景强,可直接用专业术语",
    "律师": "用户专业背景强,可直接用法律术语,无需解释基础概念",
    "销售": "用户偏务实,直接给可操作建议",
    "学生": "用户社会经验有限,术语要解释,流程要详细",
    "退休": "用户可能不熟悉新技术,建议用通俗语言",
}

# 家庭状况 → 关怀维度
FAMILY_HINTS: dict[str, str] = {
    "已婚有娃": "涉及家庭决策时可关注子女权益维度",
    "已婚": "涉及家庭决策时可考虑配偶共同利益",
    "单身": "决策维度以个人权益为主",
    "离异": "涉及家庭话题时保持中性、不预设立场",
}


def _format_factual_part(facts: dict[str, str]) -> str:
    """把 user_facts 翻译成事实陈述句.

    例:
      {"location": "上海", "occupation": "教师"}
      → "用户位于上海,职业教师."
    """
    parts: list[str] = []
    for key in [
        "location",
        "occupation",
        "age_range",
        "family_status",
        "legal_concern_type",
        "income_range",
    ]:
        if key not in facts or not facts[key].strip():
            continue
        label = KEY_LABELS.get(key, key)
        parts.append(f"{label} {facts[key]}")

    if not parts:
        return ""
    return "用户画像:" + ",".join(parts) + "."


def _infer_communication_hints(facts: dict[str, str]) -> list[str]:
    """根据 facts 推断沟通风格建议.

    返回多条建议,可能为空 list.
    """
    hints: list[str] = []

    occ = facts.get("occupation", "").strip()
    for keyword, hint in OCCUPATION_HINTS.items():
        if keyword in occ:
            hints.append(hint)
            break  # 只匹配第一条职业建议

    family = facts.get("family_status", "").strip()
    for keyword, hint in FAMILY_HINTS.items():
        if keyword in family:
            hints.append(hint)
            break

    return hints


def build_user_persona_text(facts: dict[str, str]) -> str:
    """⭐ 主入口:user_facts → 自然语言画像文本.

    Args:
        facts: M8 Hard Memory 抽出的用户事实 dict.

    Returns:
        str: 自然语言画像文本.facts 为空时返回空字符串.

    Example:
        facts = {"location": "上海", "occupation": "教师"}
        → "用户画像:所在地 上海,职业 教师.
           沟通建议:用户教育背景中等以上,可使用专业术语并适当条理化."
    """
    if not facts:
        return ""

    factual = _format_factual_part(facts)
    if not factual:
        return ""

    hints = _infer_communication_hints(facts)
    if not hints:
        return factual

    hints_part = "沟通建议:" + ";".join(hints) + "."
    return f"{factual}\n{hints_part}"


__all__ = [
    "build_user_persona_text",
    "KEY_LABELS",
    "OCCUPATION_HINTS",
    "FAMILY_HINTS",
]
