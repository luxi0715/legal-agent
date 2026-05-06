"""M11.3 — 测 get_related_articles."""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from legal_agent.db.neo4j_client import close_neo4j, init_neo4j
from legal_agent.tools.get_related_articles import get_related_articles

CASES = [
    # 入度高的明星条款,应该有 incoming
    ("民法典", "第五百一十条", "both"),
    ("民法典", "510", "both"),  # 阿拉伯数字应该也行
    # 出度高的
    ("劳动合同法", "第四十六条", "outgoing"),
    # 孤儿条款
    ("民法典", "第五百七十七条", "both"),
    # 错误参数
    ("不存在的法律", "第一条", "both"),
    ("民法典", "第九千九百九十九条", "both"),
]


async def main() -> None:
    await init_neo4j()
    try:
        for law, no, direction in CASES:
            print("=" * 70)
            print(f"Query: {law} {no} (direction={direction})")
            print("=" * 70)
            result = await get_related_articles(law, no, direction=direction)
            print(result[:500])
            if len(result) > 500:
                print("...(截断)")
            print()
    finally:
        await close_neo4j()


asyncio.run(main())
