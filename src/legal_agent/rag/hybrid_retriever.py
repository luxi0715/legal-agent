"""Hybrid retrieval: vector + BM25 fused via RRF."""

import asyncio
from typing import TypedDict

from legal_agent.rag.bm25_retriever import bm25_retrieve
from legal_agent.rag.fusion import reciprocal_rank_fusion
from legal_agent.rag.retriever import retrieve as vector_retrieve


class HybridChunk(TypedDict):
    """A chunk returned from hybrid retrieval."""

    content: str
    rrf_score: float
    sources: list[str]
    metadata: dict[str, object]


async def hybrid_retrieve(
    query: str,
    top_k: int = 5,
    candidates_per_method: int = 50,
    rrf_k: int = 60,
) -> list[HybridChunk]:
    """Retrieve via vector and BM25 in parallel, fuse with RRF.

    Args:
        query: User query.
        top_k: Final number of results.
        candidates_per_method: How many candidates each method returns
                               before fusion. Larger = better recall.
        rrf_k: RRF smoothing constant.

    Returns:
        Top-k fused results sorted by RRF score descending.
    """
    # 并发跑两路检索
    vector_task = vector_retrieve(query, top_k=candidates_per_method, min_score=0.0)
    bm25_task = bm25_retrieve(query, top_k=candidates_per_method, min_score=0.0)
    vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)

    # 转成 dict[str, Any] 列表喂给 RRF
    vector_dicts = [dict(r) for r in vector_results]
    bm25_dicts = [dict(r) for r in bm25_results]

    fused = reciprocal_rank_fusion([vector_dicts, bm25_dicts], k=rrf_k, top_k=top_k)

    results: list[HybridChunk] = []
    for item in fused:
        # 重命名 sources 让用户看得懂
        sources = item.get("sources", [])
        named_sources = []
        for s in sources:
            if s == "retriever_0":
                named_sources.append("vector")
            elif s == "retriever_1":
                named_sources.append("bm25")
        results.append(
            HybridChunk(
                content=item["content"],
                rrf_score=item["rrf_score"],
                sources=named_sources,
                metadata=item.get("metadata", {}),
            )
        )
    return results


# Export
__all__ = ["hybrid_retrieve", "HybridChunk"]
