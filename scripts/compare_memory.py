"""M8.8 — Memory 对比评测.

对比维度:
- M7 /chat/react       (无记忆)
- M8 /chat/memory      (三层记忆)

核心场景:多轮对话指代解析
- 第 1 轮:陈述背景(\"我刚被公司辞退,没有书面合同\")
- 第 2 轮:用代词引用(\"那我该怎么维权?\")

M7:第 2 轮看不到第 1 轮 → 笼统回答
M8:第 2 轮 buffer 有上下文 → 精准聚焦"辞退"+"事实劳动关系"
"""

import asyncio
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"


async def call_endpoint(
    client: httpx.AsyncClient,
    endpoint: str,
    message: str,
    session_id: str | None = None,
) -> dict:
    """调一次端点."""
    payload = {"message": message}
    if session_id:
        payload["session_id"] = session_id

    t0 = time.perf_counter()
    resp = await client.post(f"{BASE_URL}{endpoint}", json=payload, timeout=120.0)
    elapsed = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    data["_elapsed_ms"] = elapsed
    return data


async def run_scenario(
    client: httpx.AsyncClient,
    endpoint: str,
    label: str,
    turn1: str,
    turn2: str,
) -> None:
    """跑同一个 2 轮场景,使用同一 session_id."""
    print(f"\n{'━' * 78}")
    print(f"📍 {label}  endpoint={endpoint}")
    print("━" * 78)

    # 第 1 轮
    print(f"\n🗣️  Turn 1:{turn1}")
    r1 = await call_endpoint(client, endpoint, turn1)
    sid = r1["session_id"]
    print(f"   ⏱ {r1['_elapsed_ms']:.0f}ms  session={sid[:8]}...")
    print(f"   💬 {r1['reply'][:150]}...")
    if "memory_meta" in r1:
        print(f"   📊 memory: {r1['memory_meta']}")

    # 第 2 轮(关键!用同一 session_id)
    print(f"\n🗣️  Turn 2:{turn2}")
    r2 = await call_endpoint(client, endpoint, turn2, session_id=sid)
    print(f"   ⏱ {r2['_elapsed_ms']:.0f}ms")
    print(f"   💬 {r2['reply'][:300]}...")
    if "memory_meta" in r2:
        print(f"   📊 memory: {r2['memory_meta']}")


async def main() -> None:
    print("=" * 78)
    print("M8.8 评测:M7 /chat/react vs M8 /chat/memory")
    print("=" * 78)

    async with httpx.AsyncClient() as client:
        try:
            await client.get(f"{BASE_URL}/health", timeout=5.0)
        except Exception as e:
            print(f"❌ uvicorn 没在跑?{e}")
            return

        scenarios = [
            (
                "场景 1:辞退 + 维权指代",
                "我刚被公司辞退,没有书面合同",
                "那我该怎么维权?",
            ),
            (
                "场景 2:身份陈述 + 后续追问",
                "我是上海的中学老师,30 岁",
                "我能问个法律问题吗?涉及孩子抚养权那种",
            ),
        ]

        for label, t1, t2 in scenarios:
            print(f"\n\n{'═' * 78}")
            print(f"  {label}")
            print("═" * 78)

            await run_scenario(client, "/chat/react", "M7 (无记忆)", t1, t2)
            await run_scenario(client, "/chat/memory", "M8 (三层记忆)", t1, t2)

        print(f"\n\n{'═' * 78}")
        print("✅ 评测完成 — 对比 M7 vs M8 第 2 轮回复质量")
        print("═" * 78)


if __name__ == "__main__":
    asyncio.run(main())
