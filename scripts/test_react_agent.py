"""测试 ReAct Agent — 用真实 LLM + 工具."""

import asyncio
import time

from legal_agent.agent.react_agent import run_react_agent
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool

TEST_CASES = [
    ("精确条款", "民法典第五百七十七条规定了什么?"),
    ("语义检索", "老板拖欠工资怎么办?"),
    # ⭐ M6.4 case 3 弱点 — 跨工具混合
    ("跨工具混合", "民法典577条说什么?这种情况下我该怎么维权?"),
    ("闲聊", "你好,介绍一下你自己"),
]


async def main() -> None:
    await init_postgres_pool()
    try:
        for category, query in TEST_CASES:
            print("\n" + "=" * 70)
            print(f"[{category}] {query}")
            print("=" * 70 + "\n")

            t0 = time.perf_counter()
            result = await run_react_agent(user_message=query)
            elapsed = (time.perf_counter() - t0) * 1000

            print(f"\n📊 总耗时: {elapsed:.0f}ms")
            print(f"📊 迭代次数: {result['iterations']}")
            print(f"📊 工具调用: {len(result['tool_calls_log'])} 次")

            print("\n💬 最终回复(前 250 字):")
            print(result["final_reply"][:250])
            if len(result["final_reply"]) > 250:
                print("...(截断)")
    finally:
        await close_postgres_pool()


if __name__ == "__main__":
    asyncio.run(main())
