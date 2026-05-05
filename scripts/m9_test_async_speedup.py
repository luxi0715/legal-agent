"""M9.1 — 验证 record_turn 主流程不再阻塞."""

import asyncio
import time
import uuid

from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.db.redis_client import close_redis, init_redis
from legal_agent.memory.buffer_memory import clear_buffer
from legal_agent.memory.hard_memory import delete_all_user_facts, get_user_facts
from legal_agent.memory.memory_manager import record_turn


async def main() -> None:
    await init_postgres_pool()
    await init_redis()

    user_id = uuid.uuid4()
    session_id = uuid.uuid4()

    try:
        t0 = time.perf_counter()
        meta = await record_turn(
            user_id=user_id,
            session_id=session_id,
            user_message="我在上海工作,是一名 35 岁的工程师",
            assistant_reply="好的,我已了解您的背景.",
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        print(f"✅ record_turn 主流程耗时: {elapsed_ms:.0f}ms")
        print(f"   memory_meta: {meta}")
        print("\n⏳ 等待异步 entity 抽取完成(30s)...")

        await asyncio.sleep(30)

        facts = await get_user_facts(user_id)
        print(f"\n✅ 异步抽取结果: {facts}")

        if elapsed_ms < 1500:
            print("\n🎉 M9.1 成功:主流程不再被 entity 抽取阻塞")
        else:
            print(f"\n⚠️ 主流程耗时 {elapsed_ms:.0f}ms 超过 1500ms")

    finally:
        await clear_buffer(session_id)
        await delete_all_user_facts(user_id)
        await close_postgres_pool()
        await close_redis()


if __name__ == "__main__":
    asyncio.run(main())
