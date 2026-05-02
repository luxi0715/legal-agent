"""Tool: get_law_article — exact lookup by law title + article number.

这是 M6 的核心补救工具,解决 M5 暴露的精确条款号查询弱点.

M5 阶段对 "民法典第五百七十七条" 这种 query,
Vector / BM25 / Reranker 都失败 — Reranker 甚至自信地排了错的(0.4982).

本工具走 SQL 直接查 metadata,完全绕开 jieba 分词和 embedding,
精确命中"民法典 第577条"这一类查询.
"""

from legal_agent.db.postgres import get_postgres_pool


async def get_law_article(law_title: str, article_no: str) -> str:
    """按法律名 + 条款号精确查询.

    Args:
        law_title: 法律名,可以是简称('民法典')或全称.
        article_no: 条款号,中文数字格式(如'第五百七十七条').

    Returns:
        命中的条文原文.如果未找到,返回明确的"未找到"提示.

    SQL 策略:
        - law_title 用 ILIKE '%xxx%' 模糊匹配(支持简称)
        - article_no 用 = 精确匹配(因 metadata 里也是中文数字)
    """
    if not law_title or not article_no:
        return "[参数错误] law_title 和 article_no 都不能为空"

    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT metadata->>'law_title' AS law,
                   metadata->>'article_no' AS article,
                   content
            FROM embeddings
            WHERE source_type = 'law'
              AND metadata->>'law_title' ILIKE '%' || $1 || '%'
              AND metadata->>'article_no' = $2
            LIMIT 5
            """,
            law_title,
            article_no,
        )

    if not rows:
        return (
            f"[未找到] 在法律资料库中没有找到 '{law_title} {article_no}'.\n"
            f"可能原因:\n"
            f"- 法律名拼写不准确(请确认是否为正式名称)\n"
            f"- 条款号格式错误(应为中文数字,如'第五百七十七条')\n"
            f"- 该条款不在数据库收录范围内"
        )

    # 多条命中(罕见,例如多部法律都有"第一条")时全部返回
    parts = []
    for r in rows:
        parts.append(f"【{r['law']} {r['article']}】\n{r['content']}")
    return "\n\n".join(parts)


__all__ = ["get_law_article"]
