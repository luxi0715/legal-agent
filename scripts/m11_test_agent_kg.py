"""M11.4 — 测 ReAct Agent 能否自主调用 KG 工具."""

import asyncio
import sys
import time

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from legal_agent.agent.react_agent import run_react_agent
from legal_agent.db.neo4j_client import close_neo4j, init_neo4j
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.db.redis_client import close_redis, init_redis

# 4 个测试 query,关注 LLM 工具选择
TEST_CASES = [
    # 期望 LLM 选 get_related_articles
    ("KG 查询", "民法典第510条都有哪些条文引用了它?"),
    # 期望 LLM 先调 get_law_article 再调 get_related_articles(组合)
    ("混合查询", "民法典第510条说啥?以及还有哪些条文跟它有关?"),
    # 期望 LLM 选 get_law_article(纯原文)
    ("精确条款", "民法典第577条规定了什么?"),
    # 期望 LLM 选 legal_search(语义)
    ("语义检索", "老板拖欠工资怎么办"),
]


async def main() -> None:
    await init_postgres_pool()
    await init_redis()
    await init_neo4j()

    try:
        for category, query in TEST_CASES:
            print("\n" + "=" * 70)
            print(f"[{category}] {query}")
            print("=" * 70)

            t0 = time.perf_counter()
            result = await run_react_agent(user_message=query)
            elapsed = (time.perf_counter() - t0) * 1000

            tool_calls = result["tool_calls_log"]
            print(f"\n📋 工具调用 {len(tool_calls)} 次:")
            for tc in tool_calls:
                print(f"   [{tc['iteration']}] {tc['name']}({tc['arguments']})")
                print(f"       结果: {tc['result_preview'][:80]}...")

            print(f"\n📊 耗时 {elapsed:.0f}ms,迭代 {result['iterations']} 轮")
            print("\n💬 回复(前 200 字):")
            print(result["final_reply"][:200])
            if len(result["final_reply"]) > 200:
                print("...(截断)")
    finally:
        await close_neo4j()
        await close_postgres_pool()
        await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
