"""M11.2 — 从 PG 抽法条引用关系,写入 Neo4j.

流程:
  1. 从 PG 读 5 部核心法律(民法典/劳动法/劳动合同法/消保法/反不正当竞争法)
  2. 正则扫每条法条文本,抽"依照本法第X条""参照第X条"等引用
  3. 中文数字 → 阿拉伯数字标准化
  4. UNWIND 批量写 Neo4j:
     - MERGE Article 节点(幂等)
     - MERGE REFERENCES 边
  5. 输出统计:节点数、边数

设计:
  • 只抓"本法"内部引用(MVP 简化,跳过跨法律)
  • 用 MERGE 而不是 CREATE 保证幂等
  • UNWIND 批量提升速度
"""

from __future__ import annotations

import asyncio
import re
import sys
from collections import defaultdict
from typing import Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from legal_agent.db.neo4j_client import close_neo4j, get_neo4j, init_neo4j
from legal_agent.db.postgres import (
    close_postgres_pool,
    get_postgres_pool,
    init_postgres_pool,
)

# MVP 范围:5 部核心法律
TARGET_LAWS = [
    "中华人民共和国民法典",
    "中华人民共和国劳动法",
    "中华人民共和国劳动合同法",
    "中华人民共和国消费者权益保护法",
    "中华人民共和国反不正当竞争法",
]

# 法律名 → 简称(用于 article_id)
LAW_SHORT_NAMES = {
    "中华人民共和国民法典": "民法典",
    "中华人民共和国劳动法": "劳动法",
    "中华人民共和国劳动合同法": "劳动合同法",
    "中华人民共和国消费者权益保护法": "消费者权益保护法",
    "中华人民共和国反不正当竞争法": "反不正当竞争法",
}


# ──────── 中文数字 → 阿拉伯数字 ────────


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


def chinese_to_arabic(s: str) -> int:
    """转中文数字到阿拉伯数字.

    支持:一、十、二十、一百、一千二百三十四 等.
    """
    if not s:
        return 0

    # 简单情况
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
        else:  # 单位:十/百/千/万
            if current == 0:
                current = 1  # "十" = "一十"
            result += current * n
            current = 0
    result += current
    return result


# ──────── 正则抽取引用 ────────


# 匹配"第X条"的所有变体,X 是中文数字
_REF_PATTERN = re.compile(
    r"(?:依照|参照|依据|适用|根据|按照)?\s*"
    r"(?:本法)?\s*"
    r"第([零一二三四五六七八九十百千万]+)条"
)


def extract_references(content: str) -> set[int]:
    """从法条文本里抽所有"第X条"引用,返回阿拉伯数字 set.

    Note: 这里抽的是文本里所有"第X条"提及.
    会包括法条自己的开头("第五百七十七条 当事人..."),
    我们后面会过滤掉自引用.
    """
    refs: set[int] = set()
    for match in _REF_PATTERN.finditer(content):
        cn = match.group(1)
        n = chinese_to_arabic(cn)
        if n > 0:
            refs.add(n)
    return refs


# ──────── 主流程 ────────


async def fetch_articles(law_titles: list[str]) -> list[dict[str, Any]]:
    """从 PG 读出指定法律的所有条款."""
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT metadata->>'law_title' AS law_title,
                   metadata->>'article_no' AS article_no,
                   content
            FROM embeddings
            WHERE source_type = 'law'
              AND metadata->>'law_title' = ANY($1::text[])
              AND metadata->>'article_no' IS NOT NULL
            """,
            law_titles,
        )
    return [dict(r) for r in rows]


def build_article_id(law_short: str, article_no_chinese: str) -> str:
    """生成节点唯一键.格式:'民法典:577'(用阿拉伯数字)."""
    # article_no_chinese 形如 "第五百七十七条"
    m = re.match(r"第([零一二三四五六七八九十百千万]+)条", article_no_chinese)
    if not m:
        return f"{law_short}:{article_no_chinese}"  # 兜底
    n = chinese_to_arabic(m.group(1))
    return f"{law_short}:{n}"


def parse_article_no_to_int(article_no_chinese: str) -> int:
    """'第五百七十七条' → 577."""
    m = re.match(r"第([零一二三四五六七八九十百千万]+)条", article_no_chinese)
    if not m:
        return 0
    return chinese_to_arabic(m.group(1))


async def write_to_neo4j(
    nodes: list[dict[str, Any]],
    edges: list[tuple[str, str]],
) -> None:
    """批量写 Neo4j."""
    driver = get_neo4j()
    async with driver.session() as session:
        # 1. 建 unique 约束(幂等)
        await session.run(
            "CREATE CONSTRAINT article_id_unique IF NOT EXISTS "
            "FOR (a:Article) REQUIRE a.article_id IS UNIQUE"
        )
        await session.run(
            "CREATE INDEX article_law_name IF NOT EXISTS FOR (a:Article) ON (a.law_name)"
        )

        # 2. UNWIND 批量 MERGE 节点
        await session.run(
            """
            UNWIND $nodes AS node
            MERGE (a:Article {article_id: node.article_id})
            SET a.law_name = node.law_name,
                a.article_no_int = node.article_no_int,
                a.article_no = node.article_no,
                a.content = node.content
            """,
            nodes=nodes,
        )
        print(f"  ✅ 写入 {len(nodes)} 个节点")

        # 3. UNWIND 批量 MERGE 边
        if edges:
            edge_pairs = [{"src": s, "dst": d} for s, d in edges]
            await session.run(
                """
                UNWIND $edges AS edge
                MATCH (a:Article {article_id: edge.src})
                MATCH (b:Article {article_id: edge.dst})
                MERGE (a)-[:REFERENCES]->(b)
                """,
                edges=edge_pairs,
            )
            print(f"  ✅ 写入 {len(edges)} 条 REFERENCES 边")


async def main() -> None:
    print("=" * 70)
    print("M11.2 法条引用抽取 + 写入 Neo4j")
    print("=" * 70)

    await init_postgres_pool()
    await init_neo4j()

    try:
        # 1. 读 PG
        print(f"\n📥 从 PG 读 {len(TARGET_LAWS)} 部法律...")
        articles = await fetch_articles(TARGET_LAWS)
        print(f"   读到 {len(articles)} 条法条")

        # 2. 准备节点(去重)
        nodes_dict: dict[str, dict[str, Any]] = {}
        # 法律名 → 该法律的所有 article_no_int 集合(用于过滤"本法引用")
        law_articles: dict[str, set[int]] = defaultdict(set)

        for art in articles:
            law_full = art["law_title"]
            law_short = LAW_SHORT_NAMES.get(law_full, law_full)
            no_int = parse_article_no_to_int(art["article_no"])
            if no_int == 0:
                continue
            article_id = f"{law_short}:{no_int}"
            nodes_dict[article_id] = {
                "article_id": article_id,
                "law_name": law_short,
                "article_no_int": no_int,
                "article_no": art["article_no"],
                "content": art["content"],
            }
            law_articles[law_short].add(no_int)

        nodes = list(nodes_dict.values())
        print(f"   去重后 {len(nodes)} 个节点")

        # 3. 抽引用边
        print("\n🔍 正则抽取引用...")
        edges_set: set[tuple[str, str]] = set()
        ref_count_per_article: list[int] = []

        for art in articles:
            law_full = art["law_title"]
            law_short = LAW_SHORT_NAMES.get(law_full, law_full)
            self_no = parse_article_no_to_int(art["article_no"])
            if self_no == 0:
                continue
            self_id = f"{law_short}:{self_no}"

            refs = extract_references(art["content"])
            # 只保留 "本法" 内引用,排除自引用
            in_law_refs = {n for n in refs if n != self_no and n in law_articles[law_short]}
            for n in in_law_refs:
                target_id = f"{law_short}:{n}"
                edges_set.add((self_id, target_id))

            if in_law_refs:
                ref_count_per_article.append(len(in_law_refs))

        edges = list(edges_set)
        print(f"   抽到 {len(edges)} 条 REFERENCES 边")
        if ref_count_per_article:
            avg_refs = sum(ref_count_per_article) / len(ref_count_per_article)
            print(f"   有引用的法条:{len(ref_count_per_article)}")
            print(f"   平均每条引用数:{avg_refs:.2f}")
            print(f"   引用最多的法条:{max(ref_count_per_article)} 条")

        # 4. 写 Neo4j
        print("\n📤 写入 Neo4j...")
        await write_to_neo4j(nodes, edges)

        # 5. 验证
        print("\n🔎 Neo4j 现状:")
        driver = get_neo4j()
        async with driver.session() as session:
            r = await session.run("MATCH (a:Article) RETURN count(a) AS n")
            rec = await r.single()
            print(f"   节点总数:{rec['n']}")

            r = await session.run("MATCH ()-[r:REFERENCES]->() RETURN count(r) AS n")
            rec = await r.single()
            print(f"   边总数:{rec['n']}")

            # 演示:民法典 577 引用了哪些条
            r = await session.run(
                """
                MATCH (a:Article {article_id: '民法典:577'})
                      -[:REFERENCES]->(b:Article)
                RETURN b.article_id AS target
                LIMIT 10
                """
            )
            targets = [rec["target"] async for rec in r]
            print(f"\n   示例:民法典:577 引用了 {targets}")

        print("\n🎉 M11.2 抽取完成")

    finally:
        await close_neo4j()
        await close_postgres_pool()


if __name__ == "__main__":
    asyncio.run(main())
