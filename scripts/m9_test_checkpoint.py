"""M9.3 — 演示 ReAct 状态持久化 + 恢复."""

import asyncio
import sys
import uuid

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from legal_agent.agent.checkpointer import close_checkpointer, init_checkpointer
from legal_agent.agent.react_agent import get_react_graph, run_react_agent
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool


async def main() -> None:
    await init_postgres_pool()
    await init_checkpointer()

    thread_id = str(uuid.uuid4())
    print(f"📌 Thread ID: {thread_id}\n")

    try:
        # ─── 第 1 轮:正常推理,状态写入 PG ───
        print("=" * 60)
        print("第 1 轮:用户提问,ReAct 跑完,状态自动持久化")
        print("=" * 60)
        result1 = await run_react_agent(
            user_message="民法典第577条规定了什么?",
            thread_id=thread_id,
        )
        print(f"\n💬 回复: {result1['final_reply'][:150]}...")
        print(f"   迭代: {result1['iterations']} 轮")

        # ─── 模拟"服务重启" ───
        print("\n" + "=" * 60)
        print("⚠️  模拟服务重启:清空内存中的图")
        print("=" * 60)

        import legal_agent.agent.react_agent as ra

        ra._compiled_graph_with_checkpoint = None
        ra._compiled_graph_without_checkpoint = None

        # ─── 直接读 checkpoint ───
        print("\n查看 PG 里这个 thread 的历史状态:")
        app = get_react_graph(use_checkpoint=True)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await app.aget_state(config)

        print("\n✅ Checkpoint 加载成功")
        print(f"   消息数: {len(snapshot.values.get('messages', []))}")
        print(f"   tool_calls 历史: {len(snapshot.values.get('tool_calls_log', []))}")
        print(f"   下一节点: {snapshot.next}")

        # ─── 第 2 轮:同 thread,LangGraph 自动加载历史 ───
        print("\n" + "=" * 60)
        print("第 2 轮:同 thread 继续提问 — 自动延续历史")
        print("=" * 60)
        result2 = await run_react_agent(
            user_message="那这个责任要怎么承担?",
            thread_id=thread_id,
        )
        print(f"\n💬 回复: {result2['final_reply'][:200]}...")
        print(f"   迭代: {result2['iterations']} 轮")

        snapshot2 = await app.aget_state(config)
        print(f"\n✅ 累积消息数: {len(snapshot2.values.get('messages', []))}")
        print("   (应该 > 第 1 轮的消息数,证明历史持久化生效)")

    finally:
        await close_checkpointer()
        await close_postgres_pool()


if __name__ == "__main__":
    asyncio.run(main())
