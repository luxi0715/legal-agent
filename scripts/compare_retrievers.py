"""Compare vector-only vs BM25 vs Hybrid retrieval on representative queries."""

import asyncio

from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.rag.bm25_retriever import bm25_retrieve
from legal_agent.rag.hybrid_retriever import hybrid_retrieve
from legal_agent.rag.retriever import retrieve as vector_retrieve

TEST_QUERIES = [
    ("精确条款", "民法典第五百七十七条规定了什么?"),
    ("精确条款", "刑法第二百六十四条 盗窃罪"),
    ("语义查询", "老板拖欠工资不发,我能要求多少倍赔偿?"),
    ("语义查询", "邻居家狗咬伤了我,我能起诉谁?"),
    ("混合查询", "民法典中关于不当得利的规定"),
    ("概念性", "什么是合同违约责任?"),
]


async def main() -> None:
    await init_postgres_pool()
    try:
        for category, query in TEST_QUERIES:
            print("=" * 80)
            print(f"[{category}] {query}")
            print("=" * 80)

            v_results = await vector_retrieve(query, top_k=3)
            print("\n--- Vector (cosine) ---")
            for i, r in enumerate(v_results, 1):
                print(f"  {i}. [{r['score']:.3f}] {r['content'][:65]}")

            b_results = await bm25_retrieve(query, top_k=3)
            print("\n--- BM25 (ts_rank) ---")
            for i, r in enumerate(b_results, 1):
                print(f"  {i}. [{r['score']:.3f}] {r['content'][:65]}")

            h_results = await hybrid_retrieve(query, top_k=3)
            print("\n--- Hybrid (RRF) ---")
            for i, r in enumerate(h_results, 1):
                src = ",".join(r["sources"])
                print(f"  {i}. [{r['rrf_score']:.4f}] [{src}] {r['content'][:55]}")

            print()

    finally:
        await close_postgres_pool()


if __name__ == "__main__":
    asyncio.run(main())
