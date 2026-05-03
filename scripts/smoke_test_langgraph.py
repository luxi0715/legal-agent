"""LangGraph 冒烟测试.

验证目标:
1. langgraph 安装成功
2. StateGraph 能定义、编译、执行
3. 条件边(conditional edge)能根据 state 路由
4. 把概念跑通,M7.2 才开始接真工具
"""

import asyncio
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    """图的全局状态.每个节点读 + 改 它."""

    messages: Annotated[list, add_messages]  # 消息历史(自动累加)
    iteration: int  # 当前迭代轮数
    should_continue: bool  # 是否继续循环


# ────────── 节点定义 ──────────
# 每个节点 = 一个函数,输入 state,输出 state 的部分更新


def thinker_node(state: State) -> dict:
    """思考节点 — 决定下一步."""
    iteration = state["iteration"] + 1
    print(f"  🧠 [thinker 第 {iteration} 轮] 思考中...")
    return {
        "iteration": iteration,
        "messages": [{"role": "assistant", "content": f"思考第 {iteration} 轮"}],
        "should_continue": iteration < 3,  # 跑 3 轮就停
    }


def actor_node(state: State) -> dict:
    """行动节点 — 模拟执行工具."""
    print(f"  🔧 [actor 第 {state['iteration']} 轮] 模拟执行工具")
    return {
        "messages": [{"role": "assistant", "content": f"工具结果 {state['iteration']}"}],
    }


def end_node(state: State) -> dict:
    """终结节点."""
    print(f"  ✅ [end] 总迭代 {state['iteration']} 轮")
    return {"messages": [{"role": "assistant", "content": "最终回答"}]}


# ────────── 路由函数 ──────────
# 决定走哪条边


def should_continue_router(state: State) -> str:
    """根据 state 决定下一步去哪."""
    if state["should_continue"]:
        return "actor"  # 继续行动
    return "end"  # 结束


# ────────── 构建图 ──────────


def build_graph():
    """构建 StateGraph.

    流程:
        START → thinker → (条件边) → actor → thinker → ... → end → END
    """
    graph = StateGraph(State)

    # 加节点
    graph.add_node("thinker", thinker_node)
    graph.add_node("actor", actor_node)
    graph.add_node("end", end_node)

    # 加边
    graph.add_edge(START, "thinker")
    graph.add_conditional_edges(
        "thinker",
        should_continue_router,
        {"actor": "actor", "end": "end"},
    )
    graph.add_edge("actor", "thinker")  # actor 执行完回到 thinker
    graph.add_edge("end", END)

    return graph.compile()


async def main() -> None:
    print("=" * 60)
    print("LangGraph 冒烟测试")
    print("=" * 60)

    app = build_graph()

    initial_state: State = {
        "messages": [{"role": "user", "content": "测试输入"}],
        "iteration": 0,
        "should_continue": True,
    }

    print("\n开始执行图:\n")
    final_state = await app.ainvoke(initial_state)

    print("\n" + "=" * 60)
    print("✅ 图执行完成")
    print("=" * 60)
    print(f"最终迭代次数: {final_state['iteration']}")
    print(f"消息数: {len(final_state['messages'])}")
    print("\n所有消息流:")
    for i, msg in enumerate(final_state["messages"], 1):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "?")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "?")
        print(f"  [{i}] [{role}] {content}")


if __name__ == "__main__":
    asyncio.run(main())
