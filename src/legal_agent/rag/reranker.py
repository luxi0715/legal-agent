"""Cross-encoder Reranker via DashScope gte-rerank-v2.

输入 query + 候选 docs,输出按相关度重排后的 docs(带 rerank_score).
单一职责:本模块只负责"打分 + 排序",不调数据库,不管检索来源.
"""

import asyncio
from typing import Any, TypedDict

import dashscope

from legal_agent.core.config import get_settings


class RerankedChunk(TypedDict):
    """精排后的文档结构."""

    content: str
    rerank_score: float
    original_score: float | None
    metadata: dict[str, Any]
    sources: list[str]


# DashScope gte-rerank-v2 单次最大输入 30K tokens.
# 中文每字 ~1.5 tokens,留余量到 25K,即每条 doc 最长 ~500 字 × 50 条.
MAX_DOC_CHARS = 1500
MAX_CANDIDATES = 50


def _truncate(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    """截断超长 chunk,防止整体输入超过 30K token 上限."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


async def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_n: int = 5,
) -> list[RerankedChunk]:
    """对召回候选做 Cross-encoder 精排.

    Args:
        query: 用户原始查询
        candidates: 召回阶段的候选,每个必须有 content 字段
        top_n: 精排后返回前 N 条

    Returns:
        按 rerank_score 降序排列的 Top-N 结果

    Raises:
        RuntimeError: API 调用失败时
    """
    if not candidates:
        return []

    settings = get_settings()

    # 防爆 1:候选数硬上限
    safe_candidates = candidates[:MAX_CANDIDATES]

    # 防爆 2:每条 doc 截断
    documents = [_truncate(c["content"]) for c in safe_candidates]

    # DashScope SDK 是同步的,用 to_thread 避免阻塞 FastAPI 事件循环
    response = await asyncio.to_thread(
        dashscope.TextReRank.call,
        model=settings.rerank_model,
        query=query,
        documents=documents,
        top_n=min(top_n, len(documents)),
        return_documents=False,
        api_key=settings.dashscope_api_key,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Rerank API 失败: code={response.code}, message={response.message}")

    results: list[RerankedChunk] = []
    for item in response.output["results"]:
        idx = int(item["index"])
        score = float(item["relevance_score"])
        original = safe_candidates[idx]
        results.append(
            RerankedChunk(
                content=original["content"],
                rerank_score=score,
                original_score=original.get("rrf_score") or original.get("score"),
                metadata=original.get("metadata", {}),
                sources=original.get("sources", []),
            )
        )
    return results
