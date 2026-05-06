"""Tool: get_related_articles — 知识图谱引用查询 (M11).

用 Cypher 查 Neo4j 法条引用关系图.

vs M5 RAG 和 M6 SQL:
  • M5 RAG:语义检索"相似条文"
  • M6 SQL:精确查"民法典第577条"原文
  • M11 KG:查"577 条引用了哪些条 / 哪些条引用了 577"

功能:
  • outgoing: 该条引用的其他条
  • incoming: 引用该条的其他条
  • both:    双向(默认)
"""

from __future__ import annotations

import re
from typing import Any

from legal_agent.db.neo4j_client import get_neo4j

# 法律全称 → 简称(跟 M11.2 抽取脚本一致)
_LAW_NAME_MAP = {
    "中华人民共和国民法典": "民法典",
    "民法典": "民法典",
    "中华人民共和国劳动法": "劳动法",
    "劳动法": "劳动法",
    "中华人民共和国劳动合同法": "劳动合同法",
    "劳动合同法": "劳动合同法",
    "中华人民共和国消费者权益保护法": "消费者权益保护法",
    "消费者权益保护法": "消费者权益保护法",
    "中华人民共和国反不正当竞争法": "反不正当竞争法",
    "反不正当竞争法": "反不正当竞争法",
}

_CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
}


def _chinese_to_arabic(s: str) -> int:
    """中文数字 → 阿拉伯数字."""
    if not s:
        return 0
    if s in _CHINESE_DIGITS and _CHINESE_DIGITS[s] < 10:
        return _CHINESE_DIGITS[s]
    result = 0
    current = 0
    for ch in s:
        if ch not in _CHINESE_DIGITS:
            continue
        n = _CHINESE_DIGITS[ch]
        if n < 10:
            current = n
        else:
            if current == 0:
                current = 1
            result += current * n
            current = 0
    result += current
    return result


def _normalize_article_id(law_name: str, article_no: str) -> str | None:
    """把"民法典 第五百七十七条"标准化成"民法典:577"."""
    short_name = _LAW_NAME_MAP.get(law_name.strip())
    if short_name is None:
        # 模糊匹配:用户输了"民事典"这种错字
        for full, short in _LAW_NAME_MAP.items():
            if law_name in full or full in law_name:
                short_name = short
                break
    if short_name is None:
        return None

    # 解析 article_no:支持"第五百七十七条"或"577"
    article_no = article_no.strip()

    # 阿拉伯数字
    m = re.match(r"^(?:第)?(\d+)(?:条)?$", article_no)
    if m:
        return f"{short_name}:{int(m.group(1))}"

    # 中文数字
    m = re.match(r"^第([零一二三四五六七八九十百千万]+)条$", article_no)
    if m:
        n = _chinese_to_arabic(m.group(1))
        if n > 0:
            return f"{short_name}:{n}"

    return None


async def get_related_articles(
    law_name: str,
    article_no: str,
    direction: str = "both",
    limit: int = 10,
) -> str:
    """查询某条法条的关联条文(知识图谱).

    Args:
        law_name: 法律名(如"民法典")
        article_no: 条款号(中文或阿拉伯,如"第五百七十七条"或"577")
        direction: "outgoing" / "incoming" / "both"(默认)
        limit: 单方向最大返回数

    Returns:
        格式化文本.如果没有关联,返回明确提示.
    """
    article_id = _normalize_article_id(law_name, article_no)
    if article_id is None:
        return (
            f"[参数错误] 无法解析 '{law_name} {article_no}'.\n"
            f"支持的法律:民法典 / 劳动法 / 劳动合同法 / 消费者权益保护法 / 反不正当竞争法\n"
            f"条款号格式:'第五百七十七条' 或 '577'"
        )

    if direction not in ("outgoing", "incoming", "both"):
        return f"[参数错误] direction 必须是 outgoing/incoming/both,得到 '{direction}'"

    driver = get_neo4j()
    async with driver.session() as session:
        # 先验证节点存在
        r = await session.run(
            "MATCH (a:Article {article_id: $aid}) RETURN a.article_no AS no LIMIT 1",
            aid=article_id,
        )
        rec = await r.single()
        if rec is None:
            return (
                f"[未找到] 知识图谱中没有 '{article_id}'.\n"
                f"可能原因:该条款不在 MVP 收录的 5 部核心法律内,\n"
                f"或者条款号超出范围."
            )

        outgoing: list[dict[str, Any]] = []
        incoming: list[dict[str, Any]] = []

        if direction in ("outgoing", "both"):
            r = await session.run(
                """
                MATCH (a:Article {article_id: $aid})-[:REFERENCES]->(b:Article)
                RETURN b.article_id AS id, b.article_no AS no,
                       substring(b.content, 0, 80) AS preview
                LIMIT $lim
                """,
                aid=article_id,
                lim=limit,
            )
            outgoing = [dict(rec) async for rec in r]

        if direction in ("incoming", "both"):
            r = await session.run(
                """
                MATCH (a:Article)-[:REFERENCES]->(b:Article {article_id: $aid})
                RETURN a.article_id AS id, a.article_no AS no,
                       substring(a.content, 0, 80) AS preview
                LIMIT $lim
                """,
                aid=article_id,
                lim=limit,
            )
            incoming = [dict(rec) async for rec in r]

    # 格式化输出
    parts = [f"## {article_id} 的关联条文"]

    if direction in ("outgoing", "both"):
        if outgoing:
            parts.append(f"\n### 该条引用的条文({len(outgoing)} 条):")
            for item in outgoing:
                parts.append(f"- 【{item['id']}】{item['preview']}...")
        else:
            parts.append("\n### 该条没有引用其他条文")

    if direction in ("incoming", "both"):
        if incoming:
            parts.append(f"\n### 引用该条的条文({len(incoming)} 条):")
            for item in incoming:
                parts.append(f"- 【{item['id']}】{item['preview']}...")
        else:
            parts.append("\n### 没有其他条文引用该条")

    if not outgoing and not incoming:
        parts.append(
            "\n[提示] 该法条在知识图谱中是 孤立节点(没有进出引用).\n"
            "可能因为它是定义型条款,或引用关系未被规则抓到."
        )

    return "\n".join(parts)


__all__ = ["get_related_articles"]
