"""M5 评测脚本:横向对比 Vector / Hybrid / Two-Stage 三种检索方式.

输出三种方式的 Top-5 命中情况、关键分数、耗时,
为 docs/notes/M5-evaluation.md 提供数据.
"""

import asyncio
import time
from typing import Any

from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.rag.abstention import decide_abstention
from legal_agent.rag.hybrid_retriever import hybrid_retrieve
from legal_agent.rag.retriever import retrieve as vector_retrieve
from legal_agent.rag.two_stage_retriever import two_stage_retrieve

# 6 类典型 query,覆盖不同场景
TEST_QUERIES = [
    ("概念性查询", "什么是合同违约责任?"),
    ("语义模糊查询", "老板拖欠工资怎么办?"),
    ("专业术语", "不当得利如何处理?"),
    ("精确条款号 (M4 弱点)", "民法典第五百七十七条规定了什么?"),
    ("多领域查询", "合同诈骗的法律责任"),
    ("无关查询", "今天北京天气怎么样?"),
]


def fmt_chunk_summary(chunk: dict[str, Any], score_key: str) -> str:
    """格式化一条 chunk 的摘要(法律名 + 条款 + 分数)."""
    meta = chunk.get("metadata", {})
    law = meta.get("law_title", "?")[:25]  # 截断长法律名
    article = meta.get("article_no", "")
    score = chunk.get(score_key, 0.0)
    return f"  {score:.4f}  {law} {article}"


async def evaluate_query(category: str, query: str) -> None:
    """对单个 query 跑三种检索方式并对比."""
    print("\n" + "=" * 75)
    print(f"[{category}] {query}")
    print("=" * 75)

    # 方式 1: Vector only(M3)
    t0 = time.perf_counter()
    vec_results = await vector_retrieve(query, top_k=5, min_score=0.0)
    vec_ms = (time.perf_counter() - t0) * 1000

    # 方式 2: Hybrid(M4)
    t0 = time.perf_counter()
    hybrid_results = await hybrid_retrieve(query, top_k=5, candidates_per_method=50)
    hybrid_ms = (time.perf_counter() - t0) * 1000

    # 方式 3: Two-Stage(M5)
    t0 = time.perf_counter()
    two_stage_results = await two_stage_retrieve(query=query, recall_top_k=50, final_top_k=5)
    two_stage_ms = (time.perf_counter() - t0) * 1000

    # M5 拒答决策
    decision = decide_abstention(two_stage_results)

    # 输出
    print(f"\n📊 Vector only (M3) — 耗时 {vec_ms:.0f}ms")
    for r in vec_results:
        d = dict(r)
        print(fmt_chunk_summary(d, "score"))

    print(f"\n📊 Hybrid (M4) — 耗时 {hybrid_ms:.0f}ms")
    for r in hybrid_results:
        d = dict(r)
        print(fmt_chunk_summary(d, "rrf_score"))

    print(f"\n📊 Two-Stage (M5) — 耗时 {two_stage_ms:.0f}ms")
    if not two_stage_results:
        print("  (空)")
    else:
        for r in two_stage_results:
            print(fmt_chunk_summary(dict(r), "rerank_score"))

    print(f"\n🎯 M5 决策: {'拒答' if decision.should_abstain else '通过'}")
    print(f"   理由: {decision.reason}")
    print(f"   Top1 rerank={decision.top_rerank:.4f}  recall={decision.top_recall:.4f}")


async def main() -> None:
    await init_postgres_pool()
    try:
        for category, query in TEST_QUERIES:
            await evaluate_query(category, query)
    finally:
        await close_postgres_pool()
    print("\n" + "=" * 75)
    print("评测完成 ✅")
    print("=" * 75)


if __name__ == "__main__":
    asyncio.run(main())
