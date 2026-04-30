"""端到端测试 two_stage_retrieve.

真连数据库,真调 DashScope,真跑完整链路.
"""

import asyncio
import time

from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.rag.two_stage_retriever import two_stage_retrieve

# 4 类典型 query,覆盖不同场景
TEST_QUERIES = [
    ("概念性查询", "什么是合同违约责任?"),
    ("语义查询", "老板拖欠工资怎么办?"),
    ("专业术语", "不当得利如何处理?"),
    ("无关查询", "今天北京天气怎么样?"),
]


async def main() -> None:
    # 必须先初始化连接池(hybrid 内部 SQL 需要)
    await init_postgres_pool()

    try:
        for category, query in TEST_QUERIES:
            print("=" * 70)
            print(f"[{category}] {query}")
            print("=" * 70)

            t0 = time.perf_counter()
            results = await two_stage_retrieve(
                query=query,
                recall_top_k=50,
                final_top_k=5,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if not results:
                print(f"⚠️ 召回为空(耗时 {elapsed_ms:.0f}ms)\n")
                continue

            print(f"✅ 返回 {len(results)} 条(总耗时 {elapsed_ms:.0f}ms)\n")
            for i, r in enumerate(results, 1):
                meta = r["metadata"]
                law = meta.get("law_title", "?")
                article = meta.get("article_no", "")
                print(
                    f"  [{i}] rerank={r['rerank_score']:.4f}  "
                    f"召回={r['original_score']:.4f}  "
                    f"来源={r['sources']}"
                )
                print(f"      法条: {law} {article}")
                print(f"      内容: {r['content'][:50]}...")
            print()
    finally:
        await close_postgres_pool()


if __name__ == "__main__":
    asyncio.run(main())
