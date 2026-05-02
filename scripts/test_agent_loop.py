"""端到端测试 agent_loop.py(真实 LLM + 真实工具).

预计耗时:每个 query 5-30 秒,4 个 query 总计 ~2 分钟.
"""

import asyncio
import time

from legal_agent.agent.agent_loop import run_agent_loop
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool

TEST_CASES = [
    # 期望 LLM 选 legal_search
    ("语义检索", "老板拖欠工资怎么办?"),
    # 期望 LLM 选 get_law_article ⭐ M6 灵魂测试
    ("精确条款", "民法典第五百七十七条规定了什么?"),
    # 期望 LLM 不调工具
    ("闲聊", "你好,介绍一下你自己"),
    # 期望 LLM 选 legal_search 但触发 abstention
    ("无关查询", "今天北京天气怎么样?"),
]


async def main() -> None:
    await init_postgres_pool()

    try:
        for category, query in TEST_CASES:
            print("=" * 70)
            print(f"[{category}] User: {query}")
            print("=" * 70)

            t0 = time.perf_counter()
            trace = await run_agent_loop(user_message=query)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # 工具调用轨迹
            print(f"\n📋 工具调用轨迹(共 {len(trace.tool_calls)} 次):")
            if not trace.tool_calls:
                print("   (无 — LLM 直接回答)")
            else:
                for tc in trace.tool_calls:
                    print(f"   [迭代 {tc.iteration}] {tc.name}({tc.arguments})")
                    print(f"      结果预览: {tc.result_preview[:80]}...")

            # 元信息
            print(f"\n📊 总耗时:{elapsed_ms:.0f}ms")
            print(f"   迭代次数:{trace.iterations}")
            if trace.hit_max_iter:
                print("   ⚠️ 达到最大迭代上限")

            # 最终回复
            print("\n💬 最终回复:")
            preview = trace.final_reply[:300] + ("..." if len(trace.final_reply) > 300 else "")
            print(preview)
            print()
    finally:
        await close_postgres_pool()


if __name__ == "__main__":
    asyncio.run(main())
