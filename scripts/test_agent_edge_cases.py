"""M6.4 — Agent Loop 边界测试.

目的不是"通过",而是 暴露问题 + 决定是否修.
"""

import asyncio
import time

from legal_agent.agent.agent_loop import run_agent_loop
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool

EDGE_CASES = [
    {
        "category": "1. 模糊条款号",
        "query": "民法典 577 怎么说的?",
        "watch": "LLM 是否会把 '577' 转成 '第五百七十七条'?",
    },
    {
        "category": "2. 全称",
        "query": "中华人民共和国民法典第五百七十七条规定了什么?",
        "watch": "LLM 是否会把全称缩成简称?",
    },
    {
        "category": "3. 跨工具混合",
        "query": "民法典577条说什么?这种情况下我该怎么维权?",
        "watch": "LLM 是否会调 2 个不同工具?",
    },
    {
        "category": "4. 不存在的法",
        "query": "野生动物保护法第九百九十九条规定了什么?",
        "watch": "工具返回'未找到',LLM 怎么处理?",
    },
    {
        "category": "5. 半法律半闲聊",
        "query": "我女朋友最近压力大,我们离婚了,你能帮我看看吗?",
        "watch": "LLM 是否调工具?偏哪一个?",
    },
    {
        "category": "6. 错误条款号",
        "query": "民法典第十二亿条规定了什么?",
        "watch": "LLM 是否会把不合理的数字传过去?",
    },
    {
        "category": "7. 多条同时问",
        "query": "民法典第五百七十七条和第五百七十八条分别说什么?",
        "watch": "LLM 是否会一次调 2 个 get_law_article?",
    },
    {
        "category": "8. 自我反思",
        "query": "你刚才说的对吗?",
        "watch": "LLM 是否会乱调工具?",
    },
]


async def main() -> None:
    await init_postgres_pool()
    try:
        for case in EDGE_CASES:
            print("=" * 75)
            print(f"[{case['category']}] {case['query']}")
            print(f"💡 观察点: {case['watch']}")
            print("=" * 75)

            t0 = time.perf_counter()
            try:
                trace = await run_agent_loop(user_message=case["query"])
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if trace.tool_calls:
                    print(f"\n🔧 工具调用 {len(trace.tool_calls)} 次:")
                    for tc in trace.tool_calls:
                        print(f"   [{tc.iteration}] {tc.name}({tc.arguments})")
                        print(f"       结果: {tc.result_preview[:100]}...")
                else:
                    print("\n🚫 未调工具(LLM 直接回答)")

                print(f"\n📊 耗时 {elapsed_ms:.0f}ms,迭代 {trace.iterations} 轮")
                if trace.hit_max_iter:
                    print("⚠️ 达到最大迭代上限")

                preview = trace.final_reply[:250]
                print(f"\n💬 回复预览:\n{preview}")
                if len(trace.final_reply) > 250:
                    print("...(截断)")
            except Exception as e:
                print(f"\n❌ 异常: {type(e).__name__}: {e}")
            print()
    finally:
        await close_postgres_pool()


if __name__ == "__main__":
    asyncio.run(main())
