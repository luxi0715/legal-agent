"""Retrieve relevant law chunks for a query."""

import json
from typing import TypedDict

from legal_agent.db.postgres import get_postgres_pool
from legal_agent.rag.embedder import embed_batch


class RetrievedChunk(TypedDict):
    """A chunk returned from retrieval."""

    content: str
    score: float
    metadata: dict[str, str | int | None]


async def retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[RetrievedChunk]:
    """Retrieve top-k similar chunks for a query.

    Args:
        query: User's natural language question.
        top_k: How many results to return.
        min_score: Filter out results below this similarity (0-1).

    Returns:
        List sorted by similarity descending.
        Score is cosine similarity in [-1, 1], 1 means most similar.
    """
    # 1. Query → vector
    vectors = await embed_batch([query])
    query_vector = vectors[0]
    vector_str = "[" + ",".join(str(x) for x in query_vector) + "]"

    # 2. SQL similarity search via HNSW
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                content,
                metadata,
                1 - (embedding <=> $1::vector) AS similarity
            FROM embeddings
            WHERE source_type = 'law'
            ORDER BY embedding <=> $1::vector
            LIMIT $2
            """,
            vector_str,
            top_k,
        )

    results: list[RetrievedChunk] = []
    for row in rows:
        score = float(row["similarity"])
        if score < min_score:
            continue
        results.append(
            RetrievedChunk(
                content=row["content"],
                score=score,
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )
        )
    return results
