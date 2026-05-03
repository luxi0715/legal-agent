"""Entity Extractor: 从对话中抽取用户身份事实 (M8 ⭐).

LLM 抽取 → confidence 过滤 → 自动写 Hard Memory.

设计:
  • 受限 key 白名单(防 LLM 自创无意义 fact)
  • JSON-only 输出(response_format)
  • 解析失败 silent fallback(不阻塞主流程)
  • confidence 阈值过滤(默认 0.7)
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from legal_agent.agent.llm_client import get_llm_client
from legal_agent.core.config import get_settings
from legal_agent.memory.hard_memory import upsert_user_fact

# 受限 key 白名单(LLM 只能从这里选,防自创字段)
ALLOWED_KEYS = {
    "location",  # 用户所在地(北京/上海/...)
    "occupation",  # 职业(程序员/教师/...)
    "age_range",  # 年龄段(20-25/25-30/...)
    "family_status",  # 家庭状况(已婚/单身/有娃/...)
    "legal_concern_type",  # 法律关注主题(劳动/婚姻/合同/...)
    "income_range",  # 收入区间(月薪 1-2 万/...)
}

# 低于此置信度的 fact 不入库
DEFAULT_CONFIDENCE_THRESHOLD = 0.7


ENTITY_EXTRACTION_PROMPT = """你是用户事实抽取专家.从对话中提取用户的明确个人事实.

【严格规则】
1. 只提取用户 明确说出 的事实,不要推测、不要猜测
2. 用户提到 \"我朋友\"\"我同事\"\"我老婆\" 等他人信息 不要 抽取
3. 只能使用以下 6 个 key 之一,不许自创:
   - location:用户所在地(如 北京、上海)
   - occupation:用户职业(如 程序员、教师)
   - age_range:年龄段(如 25-30)
   - family_status:家庭状况(如 已婚有娃)
   - legal_concern_type:咨询主题(如 劳动纠纷、合同违约)
   - income_range:收入区间(如 月薪 1-2 万)
4. 没有可提取的事实 → 返回空数组
5. confidence:用户说得越明确分数越高,模糊推测分数越低
6. 必须输出 严格的 JSON 对象,无 markdown,无前缀,无后缀

【输出格式】
{{
  "facts": [
    {{"key": "<上面 6 个 key 之一>", "value": "<具体值>", "confidence": 0.0~1.0}},
    ...
  ]
}}

【对话片段】
{messages}

【输出 JSON】"""


def _format_messages(messages: list[dict[str, str]]) -> str:
    """把对话格式化成 LLM 友好的多行文本."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


async def extract_entities(
    messages: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """从对话中抽取 facts(只抽不写).

    Args:
        messages: 对话片段,每条含 role / content

    Returns:
        list[dict]: 每个 dict 含 key / value / confidence.
                    抽取失败或无 fact 时返回空 list.
    """
    if not messages:
        return []

    prompt = ENTITY_EXTRACTION_PROMPT.format(messages=_format_messages(messages))

    client = get_llm_client()
    settings = get_settings()

    try:
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,  # 求稳,抽取不要发散
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = json.loads(raw)
    except (json.JSONDecodeError, KeyError, AttributeError):
        # LLM 输出不规范 → 静默跳过,不阻塞主流程
        return []
    except Exception:
        # 网络抖动 / API 错误 → 同样静默
        return []

    facts = parsed.get("facts", [])
    if not isinstance(facts, list):
        return []

    # 过滤无效项:不在白名单 / 缺字段 / 类型不对
    valid: list[dict[str, Any]] = []
    for f in facts:
        if not isinstance(f, dict):
            continue
        key = f.get("key")
        value = f.get("value")
        conf = f.get("confidence", 0.0)
        if (
            isinstance(key, str)
            and key in ALLOWED_KEYS
            and isinstance(value, str)
            and value.strip()
            and isinstance(conf, (int, float))
        ):
            valid.append(
                {
                    "key": key,
                    "value": value.strip(),
                    "confidence": float(conf),
                }
            )

    return valid


async def extract_and_persist(
    user_id: UUID,
    messages: list[dict[str, str]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """⭐ 完整链路:LLM 抽取 + 阈值过滤 + 写 Hard Memory.

    Args:
        user_id: 用户标识
        messages: 对话片段
        confidence_threshold: 低于此分数不入库

    Returns:
        list[dict]: 实际写入的 facts(已过 confidence 阈值)
    """
    facts = await extract_entities(messages)

    persisted: list[dict[str, Any]] = []
    for f in facts:
        if f["confidence"] < confidence_threshold:
            continue
        await upsert_user_fact(
            user_id=user_id,
            key=f["key"],
            value=f["value"],
            confidence=f["confidence"],
        )
        persisted.append(f)

    return persisted


__all__ = [
    "ALLOWED_KEYS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "extract_and_persist",
    "extract_entities",
]
