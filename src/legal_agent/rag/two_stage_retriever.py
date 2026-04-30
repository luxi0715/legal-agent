"""Two-Stage Retrieval: Hybrid 召回 → Reranker 精排.

M5 核心编排层,串联完整的 RAG 检索链路:
  Stage 1: Hybrid (BM25 + 向量 + RRF) → recall_top_k 候选
  Stage 2: Reranker → final_top_k 精选

本模块只做编排,不写检索/精排具体逻辑.
"""

from legal_agent.rag.hybrid_retriever import hybrid_retrieve
from legal_agent.rag.reranker import RerankedChunk, rerank


async def two_stage_retrieve(
    query: str,
    recall_top_k: int = 50,
    final_top_k: int = 5,
) -> list[RerankedChunk]:
    """两阶段检索:Hybrid 召回 + Reranker 精排.

    Args:
        query: 用户查询
        recall_top_k: 召回阶段保留多少候选送给 reranker.
                      官方推荐 50,太多 reranker 慢,太少容易漏.
        final_top_k: 精排后返回前 N 条给 LLM(通常 3~5).

    Returns:
        Top-N 精排结果,按 rerank_score 降序.
        如果召回为空,返回空列表(由调用方决定是否拒答).
    """
    # Stage 1: Hybrid 召回 — 显式传两个 top_k 都为 recall_top_k
    # • top_k=recall_top_k:融合后保留这么多
    # • candidates_per_method=recall_top_k:每路也召这么多
    hybrid_results = await hybrid_retrieve(
        query=query,
        top_k=recall_top_k,
        candidates_per_method=recall_top_k,
    )

    if not hybrid_results:
        return []

    # 转成 dict 喂给 rerank(rerank 不耦合 HybridChunk 类型)
    candidates = [
        {
            "content": h["content"],
            "rrf_score": h["rrf_score"],
            "metadata": h["metadata"],
            "sources": h["sources"],
        }
        for h in hybrid_results
    ]

    # Stage 2: Reranker 精排
    return await rerank(
        query=query,
        candidates=candidates,
        top_n=final_top_k,
    )


__all__ = ["two_stage_retrieve"]
