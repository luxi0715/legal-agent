"""BM25-like keyword retrieval using PostgreSQL tsvector + ts_rank.

Strategy:
- OR matching for recall
- ts_rank with length normalization (flag 32) to suppress short-doc bias
- BM25 returns rough candidates, Hybrid + RRF does final ranking.
"""

import json
from typing import TypedDict

from legal_agent.db.postgres import get_postgres_pool
from legal_agent.rag.tokenizer import tokenize_for_tsvector


class BM25Chunk(TypedDict):
    """A chunk returned from BM25 retrieval."""

    content: str
    score: float
    metadata: dict[str, object]


async def bm25_retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[BM25Chunk]:
    """Retrieve top-k chunks by BM25-like keyword scoring."""
    query_tokens = tokenize_for_tsvector(query)
    if not query_tokens.strip():
        return []

    tokens = query_tokens.split()
    if not tokens:
        return []

    # OR 召回:任一 token 命中即返回
    ts_query = " | ".join(tokens)

    # ts_rank 第三个参数 normalization 用 32:
    #   = "rank/(1 + log(unique_words))"
    #   → 长文档 不被偏袒
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                content,
                metadata,
                ts_rank(content_tsv, query, 32) AS score
            FROM embeddings, to_tsquery('simple', $1) query
            WHERE source_type = 'law'
              AND content_tsv @@ query
            ORDER BY score DESC
            LIMIT $2
            """,
            ts_query,
            top_k,
        )

    results: list[BM25Chunk] = []
    for row in rows:
        score = float(row["score"])
        if score < min_score:
            continue
        results.append(
            BM25Chunk(
                content=row["content"],
                score=score,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
        )
    return results
