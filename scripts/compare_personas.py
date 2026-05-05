"""M10.5 评测 — Persona 切换效果对比.

同一个 query 跑 5 个 persona,观察:
- 回复风格差异(default 平实 vs strict 严谨 vs friendly 温暖)
- 用户画像注入是否生效
- Guard 漂移检测是否触发
"""

import asyncio
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"


async def call_persona(
    client: httpx.AsyncClient,
    message: str,
    persona_mode: str,
    session_id: str | None = None,
) -> dict:
    """调一次 /chat/persona."""
    payload = {"message": message, "persona_mode": persona_mode}
    if session_id:
        payload["session_id"] = session_id

    t0 = time.perf_counter()
    resp = await client.post(f"{BASE_URL}/chat/persona", json=payload, timeout=120.0)
    elapsed = (time.perf_counter() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    data["_elapsed_ms"] = elapsed
    return data


async def main() -> None:
    print("=" * 78)
    print("M10.5 评测:Persona 切换 + Guard 检测")
    print("=" * 78)

    async with httpx.AsyncClient() as client:
        try:
            await client.get(f"{BASE_URL}/health", timeout=5.0)
        except Exception as e:
            print(f"❌ uvicorn 没在跑?{e}")
            return

        # ─── 评测 1:同一 query × 5 个 persona ───
        query = "我刚被公司辞退,没有签书面合同,该怎么办?"
        print("\n📍 评测 1:同一 query × 5 个 persona")
        print(f"   Query: {query}\n")

        for mode in ["default", "strict", "friendly", "enterprise", "litigation"]:
            print(f"━━━━━━━━━━━━━━━━ persona_mode = {mode} ━━━━━━━━━━━━━━━━")
            try:
                result = await call_persona(client, query, mode)
                print(f"⏱  {result['_elapsed_ms']:.0f}ms")
                print("📝 回复(前 250 字):")
                print(f"   {result['reply'][:250]}...")
                print(
                    f"🛡️  Guard: drift={result['guard']['is_drift']}, "
                    f"severity={result['guard']['severity']}"
                )
                if result["guard"]["triggered_phrases"]:
                    print(f"   ⚠️  触发词: {result['guard']['triggered_phrases']}")
                print()
            except Exception as e:
                print(f"❌ {mode} 失败: {e}\n")

        # ─── 评测 2:User Persona 注入(2 轮) ───
        print("\n\n📍 评测 2:User Persona 注入(教师身份保留)")

        # Turn 1:陈述身份
        t1 = "我是上海的中学老师,30 岁"
        print(f"\n🗣️  Turn 1: {t1}")
        r1 = await call_persona(client, t1, "default")
        sid = r1["session_id"]
        print(f"   ⏱ {r1['_elapsed_ms']:.0f}ms  session={sid[:8]}...")
        print(f"   📊 memory_meta: {r1['memory_meta']}")

        # 等 entity 抽取完成
        print("\n⏳ 等 5 秒让异步 entity 抽取完成...")
        await asyncio.sleep(5)

        # Turn 2:不再说身份,用 default persona 看是否体现教师画像
        t2 = "我能问个法律问题吗?涉及孩子抚养权那种"
        print(f"\n🗣️  Turn 2: {t2}")
        r2 = await call_persona(client, t2, "default", sid)
        print(f"   ⏱ {r2['_elapsed_ms']:.0f}ms")
        print("   📝 回复(前 350 字):")
        print(f"      {r2['reply'][:350]}...")
        print(f"   🛡️  Guard: {r2['guard']}")

        print(f"\n\n{'═' * 78}")
        print("✅ M10.5 评测完成")
        print("═" * 78)


if __name__ == "__main__":
    asyncio.run(main())
