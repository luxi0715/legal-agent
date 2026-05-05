"""M9.2 — 测 LLMRouter 延迟分布."""

import asyncio
import time

from legal_agent.agent.router import LLMRouter, RuleBasedRouter
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.db.redis_client import close_redis, init_redis

TEST_QUERIES = [
    "你好",
    "我被公司辞退了",
    "民法典第577条",
    "今天天气怎么样",
    "你能做什么",
    "合同诈骗怎么处理",
    "推荐电影",
]


async def main() -> None:
    await init_postgres_pool()
    await init_redis()

    print("=" * 60)
    print("LLMRouter 延迟分布")
    print("=" * 60)

    router = LLMRouter()
    rule_router = RuleBasedRouter()

    total_llm = 0.0
    total_rule = 0.0

    for q in TEST_QUERIES:
        # LLMRouter
        t0 = time.perf_counter()
        llm_result = await router.route(q)
        llm_ms = (time.perf_counter() - t0) * 1000
        total_llm += llm_ms

        # RuleBasedRouter
        t0 = time.perf_counter()
        rule_result = await rule_router.route(q)
        rule_ms = (time.perf_counter() - t0) * 1000
        total_rule += rule_ms

        agree = "✅" if llm_result == rule_result else "⚠️"
        print(
            f"  {agree} {q[:30]:30s} "
            f"LLM={llm_result:10s} ({llm_ms:5.0f}ms)  "
            f"Rule={rule_result:10s} ({rule_ms:4.1f}ms)"
        )

    n = len(TEST_QUERIES)
    print()
    print(f"LLMRouter  平均延迟: {total_llm / n:.0f}ms")
    print(f"RuleRouter 平均延迟: {total_rule / n:.1f}ms")
    print(f"成本(LLM 多调一次): 每 query +{total_llm / n:.0f}ms")

    await close_postgres_pool()
    await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
