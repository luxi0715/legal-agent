"""端到端测试 tools/ 模块(用真实数据库 + 真实 RAG 链路)."""

import asyncio

from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.tools.dispatcher import dispatch_tool

TEST_CASES = [
    # 测试 legal_search(包装 M5)
    {
        "tool": "legal_search",
        "args": {"query": "老板拖欠工资怎么办"},
        "expect_contains": "劳动",
    },
    # 测试 get_law_article(M6 杀手锏)⭐
    {
        "tool": "get_law_article",
        "args": {"law_title": "民法典", "article_no": "第五百七十七条"},
        "expect_contains": "违约",
    },
    # 测试 get_law_article 简称
    {
        "tool": "get_law_article",
        "args": {"law_title": "劳动合同法", "article_no": "第三十条"},
        "expect_contains": "劳动报酬",
    },
    # 测试 get_law_article 未命中
    {
        "tool": "get_law_article",
        "args": {"law_title": "民法典", "article_no": "第九千九百九十九条"},
        "expect_contains": "未找到",
    },
    # 测试参数错误
    {
        "tool": "get_law_article",
        "args": {"law_title": "民法典"},  # 缺 article_no
        "expect_contains": "参数错误",
    },
    # 测试未知工具
    {
        "tool": "fake_tool",
        "args": {"x": 1},
        "expect_contains": "未知工具",
    },
]


async def main() -> None:
    await init_postgres_pool()

    pass_count = 0
    fail_count = 0

    try:
        for i, case in enumerate(TEST_CASES, 1):
            print("=" * 70)
            print(f"测试 {i}: {case['tool']}({case['args']})")
            print("=" * 70)

            result = await dispatch_tool(case["tool"], case["args"])

            preview = result[:200] + ("..." if len(result) > 200 else "")
            print(f"\n返回:\n{preview}\n")

            if case["expect_contains"] in result:
                print(f"✅ 包含期望关键字: '{case['expect_contains']}'")
                pass_count += 1
            else:
                print(f"❌ 未包含期望关键字: '{case['expect_contains']}'")
                fail_count += 1
            print()
    finally:
        await close_postgres_pool()

    print("=" * 70)
    print(f"最终: {pass_count}/{len(TEST_CASES)} 通过")
    if fail_count == 0:
        print("🎉 所有工具测试通过")
    else:
        print(f"⚠️ {fail_count} 个测试失败")


if __name__ == "__main__":
    asyncio.run(main())
