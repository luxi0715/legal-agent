"""端到端验证 abstention 决策:用 M5.3 的 4 个 query 看决策对不对."""

import asyncio

from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.rag.abstention import decide_abstention
from legal_agent.rag.two_stage_retriever import two_stage_retrieve

TEST_CASES = [
    ("概念性查询", "什么是合同违约责任?", False),  # 期望:通过
    ("语义查询", "老板拖欠工资怎么办?", False),  # 期望:通过
    ("专业术语", "不当得利如何处理?", False),  # 期望:通过
    ("无关查询", "今天北京天气怎么样?", True),  # 期望:拒答
]


async def main() -> None:
    await init_postgres_pool()

    pass_count = 0
    fail_count = 0

    try:
        for category, query, expected_abstain in TEST_CASES:
            print("=" * 70)
            print(f"[{category}] {query}")
            print(f"期望:{'拒答' if expected_abstain else '通过'}")
            print("=" * 70)

            results = await two_stage_retrieve(
                query=query,
                recall_top_k=50,
                final_top_k=5,
            )

            decision = decide_abstention(results)

            actual_abstain = decision.should_abstain
            match = actual_abstain == expected_abstain
            mark = "✅" if match else "❌"

            print(f"{mark} 实际:{'拒答' if actual_abstain else '通过'}")
            print(f"   Top1 rerank: {decision.top_rerank:.4f}")
            print(f"   Top1 recall: {decision.top_recall:.4f}")
            print(f"   理由: {decision.reason}")
            if decision.user_message:
                print(f"   用户提示: {decision.user_message}")
            print()

            if match:
                pass_count += 1
            else:
                fail_count += 1

        print("=" * 70)
        print(f"最终:{pass_count}/{len(TEST_CASES)} 决策正确")
        if fail_count > 0:
            print(f"⚠️ 有 {fail_count} 个误判,M5.7 阶段需要调阈值")
        else:
            print("🎉 所有决策正确")
    finally:
        await close_postgres_pool()


if __name__ == "__main__":
    asyncio.run(main())
