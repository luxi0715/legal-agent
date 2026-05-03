"""DeepSeek Tool Use 冒烟测试.

验证目标:
1. DeepSeek 能识别"何时调工具"
2. 输出标准 tool_calls 结构
3. 多轮闭环:tool_call → 模拟执行 → tool_result → 最终回答
4. 工具选择正确性(2 工具区分)

注意:这个脚本里的工具用模拟函数,不真调数据库,只验证协议层.
"""

import asyncio
import json
import time

from openai import AsyncOpenAI

from legal_agent.core.config import get_settings

# ──────────────────────────────────────────
# 工具定义(JSON Schema 格式)
# ──────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "legal_search",
            "description": (
                "查询中国法律条文(语义检索 + 精排).适用于:\n"
                "- 概念解释('什么是合同违约')\n"
                "- 维权咨询('老板拖欠工资怎么办')\n"
                "- 法律责任分析('合同诈骗的责任')\n"
                "不适合:已知具体法律名 + 条款号的精确查询(请用 get_law_article)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "用户的法律问题,自然语言形式",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_law_article",
            "description": (
                "按法律名 + 条款号精确查询条文原文.适用于:\n"
                "- 用户问'XX法第X条规定了什么'\n"
                "- 用户引用具体条款号问解读\n"
                "条款号要保留中文数字格式,例如'第五百七十七条'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "law_title": {
                        "type": "string",
                        "description": (
                            "法律名称,可以是简称('民法典')或全称('中华人民共和国民法典')"
                        ),
                    },
                    "article_no": {
                        "type": "string",
                        "description": "条款号,中文数字格式,如'第五百七十七条'",
                    },
                },
                "required": ["law_title", "article_no"],
            },
        },
    },
]


# ──────────────────────────────────────────
# 模拟工具(冒烟阶段不真调,只验证协议)
# ──────────────────────────────────────────
def fake_legal_search(query: str) -> str:
    return (
        f"[模拟检索] query='{query}'\n"
        f"找到 3 条相关法条:\n"
        f"[1] 劳动合同法 第三十条 用人单位应当按时足额支付劳动报酬...\n"
        f"[2] 劳动法 第九十一条 用人单位拖欠工资的,劳动行政部门责令支付...\n"
        f"[3] 刑法 第二百七十六条之一 恶意欠薪罪..."
    )


def fake_get_law_article(law_title: str, article_no: str) -> str:
    return (
        f"[模拟原文] {law_title} {article_no}\n"
        f"当事人一方不履行合同义务或者履行合同义务不符合约定的,"
        f"应当承担继续履行、采取补救措施或者赔偿损失等违约责任."
    )


# ──────────────────────────────────────────
# 测试用例(覆盖 3 类决策路径)
# ──────────────────────────────────────────
TEST_QUERIES = [
    ("语义检索期望", "老板拖欠工资怎么办?"),  # 期望调 legal_search
    ("精确查询期望", "民法典第五百七十七条规定了什么?"),  # 期望调 get_law_article
    ("不调工具期望", "你好,介绍一下你自己"),  # 期望不调工具
]


async def main() -> None:
    settings = get_settings()
    client = AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )

    for category, query in TEST_QUERIES:
        print("=" * 70)
        print(f"[{category}] User: {query}")
        print("=" * 70)

        messages: list[dict] = [
            {
                "role": "system",
                "content": (
                    "你是法律顾问助手,根据用户问题决定是否调用工具.\n"
                    "如果是法律问题,请调用合适的工具获取信息后再回答.\n"
                    "如果是闲聊或自我介绍,直接回答即可."
                ),
            },
            {"role": "user", "content": query},
        ]

        # 第一轮:让 LLM 决策
        t0 = time.perf_counter()
        response = await client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        first_round_ms = (time.perf_counter() - t0) * 1000

        msg = response.choices[0].message

        if msg.tool_calls:
            print(f"\n🔧 LLM 决定调工具(第一轮 {first_round_ms:.0f}ms):")
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                print(f"   • {fn_name}({fn_args})")

            # 模拟执行
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                if fn_name == "legal_search":
                    result = fake_legal_search(**fn_args)
                elif fn_name == "get_law_article":
                    result = fake_get_law_article(**fn_args)
                else:
                    result = f"[错误] 未知工具 {fn_name}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

            # 第二轮:LLM 看到工具结果生成最终回答
            t1 = time.perf_counter()
            response2 = await client.chat.completions.create(
                model=settings.deepseek_model,
                messages=messages,
                temperature=0.3,
            )
            second_round_ms = (time.perf_counter() - t1) * 1000

            print(
                f"\n💬 LLM 最终回答(第二轮 {second_round_ms:.0f}ms):\n"
                f"{response2.choices[0].message.content}"
            )
            print(
                f"\n📊 总耗时:{first_round_ms + second_round_ms:.0f}ms "
                f"(决策 {first_round_ms:.0f}ms + 生成 {second_round_ms:.0f}ms)"
            )
        else:
            print(f"\n💬 LLM 直接回答(未调工具,耗时 {first_round_ms:.0f}ms):")
            print(msg.content)

        print()


if __name__ == "__main__":
    asyncio.run(main())
