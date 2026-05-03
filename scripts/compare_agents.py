"""M7.5 — Agent 横向对比评测.

对比维度:
- M6 /chat/agent (while-loop based)
- M7 /chat/react (LangGraph based)

关注:
1. 工具调用次数与并行度
2. 响应延迟
3. 回答质量(vibe check)
"""

import asyncio
import json
import time
from typing import Any

import httpx

BASE_URL = "http://127.0.0.1:8000"

TEST_QUERIES = [
    ("1. 精确条款", "民法典第五百七十七条规定了什么?"),
    ("2. 语义检索", "老板拖欠工资怎么办?"),
    ("3. 跨工具混合 ⭐ M6 弱点", "民法典577条说什么?这种情况下我该怎么维权?"),
    ("4. 跨工具复杂", "我和公司签的合同被违约了,具体看哪条法律?该怎么办?"),
    ("5. 多条同时问", "民法典第五百七十七条和第五百七十八条分别说什么?"),
    ("6. 闲聊", "你好,介绍一下你自己"),
]


async def call_endpoint(client: httpx.AsyncClient, endpoint: str, message: str) -> dict[str, Any]:
    """调用一个 chat 端点."""
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE_URL}{endpoint}",
            json={"message": message},
            timeout=120.0,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()

        if endpoint == "/chat/react":
            return {
                "elapsed_ms": elapsed_ms,
                "iterations": data.get("iterations", 0),
                "tool_calls_count": data.get("tool_calls_count", 0),
                "tool_calls": data.get("tool_calls", []),
                "reply": data.get("reply", ""),
            }
        else:
            # M6 /chat/agent 不返回 trace 信息,只能拿 reply
            return {
                "elapsed_ms": elapsed_ms,
                "iterations": "N/A",
                "tool_calls_count": "N/A",
                "tool_calls": [],
                "reply": data.get("reply", ""),
            }
    except Exception as e:
        return {
            "elapsed_ms": (time.perf_counter() - t0) * 1000,
            "error": f"{type(e).__name__}: {e}",
            "iterations": 0,
            "tool_calls_count": 0,
            "tool_calls": [],
            "reply": "",
        }


async def main() -> None:
    print("=" * 78)
    print("M7.5 评测:M6 /chat/agent vs M7 /chat/react")
    print("=" * 78)

    async with httpx.AsyncClient() as client:
        # 健康检查
        try:
            await client.get(f"{BASE_URL}/health", timeout=5.0)
        except Exception as e:
            print(f"❌ uvicorn 没在跑?{e}")
            return

        for category, query in TEST_QUERIES:
            print(f"\n{'━' * 78}")
            print(f"【{category}】{query}")
            print("━" * 78)

            # M6
            print("\n📍 M6 /chat/agent")
            m6 = await call_endpoint(client, "/chat/agent", query)
            print(f"  耗时: {m6['elapsed_ms']:.0f}ms")
            print(f"  回复: {m6['reply'][:120]}...")

            # M7
            print("\n📍 M7 /chat/react")
            m7 = await call_endpoint(client, "/chat/react", query)
            print(f"  耗时: {m7['elapsed_ms']:.0f}ms")
            print(f"  迭代: {m7['iterations']} 轮")
            print(f"  工具调用: {m7['tool_calls_count']} 次")
            for tc in m7["tool_calls"]:
                args_short = json.dumps(tc["arguments"], ensure_ascii=False)[:80]
                print(f"    • [iter {tc['iteration']}] {tc['name']}({args_short})")
            print(f"  回复: {m7['reply'][:120]}...")

        print(f"\n{'═' * 78}")
        print("✅ 评测完成 — 把这份输出贴给 Claude,他会生成 M7-evaluation.md")
        print("═" * 78)


if __name__ == "__main__":
    asyncio.run(main())
