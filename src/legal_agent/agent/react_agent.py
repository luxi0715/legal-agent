"""ReAct Agent based on LangGraph StateGraph.

vs M6 agent_loop.py:
- M6 用 while 循环硬写,控制流嵌在代码里
- M7 用图结构声明,thinker/actor 解耦,易扩展

ReAct 循环:
  thinker → (有工具调用?)
              是 → actor → thinker (再思考)
              否 → END (最终回答)

防死循环:state.iteration 上限.

M8.7 加了 initial_messages 旁路参数,让 Memory Manager
能注入历史上下文.向后兼容,M7 现有调用不受影响.

M9.3 加了 thread_id 参数 + checkpointer 支持,
LangGraph 自动持久化 ReAct 状态到 PostgreSQL.
"""

import json
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from legal_agent.agent.checkpointer import get_checkpointer
from legal_agent.agent.llm_client import DEFAULT_SYSTEM_PROMPT, get_llm_client
from legal_agent.core.config import get_settings
from legal_agent.tools.definitions import ALL_TOOLS
from legal_agent.tools.dispatcher import dispatch_tool

MAX_ITERATIONS = 5

REACT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT + (
    "\n\n你是 ReAct Agent,你可以多步思考 + 调用工具.\n"
    "每一步你可以:\n"
    "1. 调用工具获取信息\n"
    "2. 观察工具结果后,如果信息不够,再调用其他工具\n"
    "3. 信息足够时,生成最终回答\n"
    "可用工具:\n"
    "- legal_search:开放性法律问题\n"
    "- get_law_article:精确条款号查询"
)


class ReactState(TypedDict):
    """ReAct Agent 全局状态."""

    messages: Annotated[list[Any], add_messages]
    iteration: int
    tool_calls_log: list[dict[str, Any]]


# ────────── 节点函数 ──────────


def _to_openai_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """把 LangChain Message 对象转回 OpenAI 兼容的 dict."""
    type_to_role = {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}
    result = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
            continue
        msg: dict[str, Any] = {"role": type_to_role.get(getattr(m, "type", ""), "user")}
        if hasattr(m, "content") and m.content:
            msg["content"] = m.content
        if hasattr(m, "tool_calls") and m.tool_calls:
            tcs_out = []
            for tc in m.tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name", "")
                    args_val = tc.get("args", {})
                else:
                    tc_id = tc.id
                    tc_name = tc.function.name
                    args_val = tc.function.arguments
                args_str = (
                    args_val
                    if isinstance(args_val, str)
                    else json.dumps(args_val, ensure_ascii=False)
                )
                tcs_out.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": tc_name, "arguments": args_str},
                    }
                )
            msg["tool_calls"] = tcs_out
        if hasattr(m, "tool_call_id") and m.tool_call_id:
            msg["tool_call_id"] = m.tool_call_id
        result.append(msg)
    return result


async def thinker_node(state: ReactState) -> dict[str, Any]:
    """思考节点 — LLM 决定下一步."""
    iteration = state["iteration"] + 1
    print(f"  🧠 [thinker 第 {iteration} 轮] LLM 决策中...")

    if iteration > MAX_ITERATIONS:
        print(f"  ⚠️ 达到迭代上限 {MAX_ITERATIONS}")
        return {
            "iteration": iteration,
            "messages": [
                {
                    "role": "assistant",
                    "content": "(达到最大思考轮数,任务可能太复杂)",
                }
            ],
        }

    client = get_llm_client()
    settings = get_settings()
    openai_messages = _to_openai_messages(state["messages"])

    response = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=openai_messages,  # type: ignore[call-overload]
        tools=ALL_TOOLS,
        tool_choice="auto",
        temperature=0.3,
    )
    msg = response.choices[0].message

    return {
        "iteration": iteration,
        "messages": [msg.model_dump()],
    }


async def actor_node(state: ReactState) -> dict[str, Any]:
    """行动节点 — 执行 thinker 决定的所有工具调用."""
    last_msg = state["messages"][-1]

    tool_calls = (
        last_msg.tool_calls if hasattr(last_msg, "tool_calls") else last_msg.get("tool_calls", [])
    )

    if not tool_calls:
        return {}

    print(f"  🔧 [actor] 执行 {len(tool_calls)} 个工具")

    new_messages = []
    new_logs = []

    for tc in tool_calls:
        if isinstance(tc, dict):
            tc_id = tc.get("id", "")
            if "function" in tc:
                fn_name = tc["function"]["name"]
                fn_args_raw = tc["function"]["arguments"]
                fn_args = json.loads(fn_args_raw) if isinstance(fn_args_raw, str) else fn_args_raw
            else:
                fn_name = tc.get("name", "")
                fn_args = tc.get("args", {})
        else:
            tc_id = tc.id
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

        try:
            tool_result = await dispatch_tool(fn_name, fn_args)
        except Exception as e:
            tool_result = f"[工具执行错误] {type(e).__name__}: {e}"

        print(f"     • {fn_name}({fn_args}) → {tool_result[:60]}...")

        new_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": tool_result,
            }
        )
        new_logs.append(
            {
                "iteration": state["iteration"],
                "name": fn_name,
                "arguments": fn_args,
                "result_preview": tool_result[:200],
            }
        )

    return {
        "messages": new_messages,
        "tool_calls_log": state["tool_calls_log"] + new_logs,
    }


# ────────── 路由函数 ──────────


def route_after_thinker(state: ReactState) -> str:
    """thinker 后的路由."""
    last_msg = state["messages"][-1]
    tool_calls = (
        last_msg.tool_calls if hasattr(last_msg, "tool_calls") else last_msg.get("tool_calls", [])
    )
    if tool_calls and state["iteration"] < MAX_ITERATIONS:
        return "actor"
    return "end"


# ────────── 图构建 ──────────


def build_react_graph(checkpointer: Any = None) -> Any:
    """构建 ReAct 图.

    Args:
        checkpointer: 可选 LangGraph checkpointer.
                      传入则启用状态持久化(M9.3),否则不持久化.
    """
    graph = StateGraph(ReactState)
    graph.add_node("thinker", thinker_node)
    graph.add_node("actor", actor_node)

    graph.add_edge(START, "thinker")
    graph.add_conditional_edges(
        "thinker",
        route_after_thinker,
        {"actor": "actor", "end": END},
    )
    graph.add_edge("actor", "thinker")

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()


# 编译后的图(模块级单例)
_compiled_graph_with_checkpoint: Any = None
_compiled_graph_without_checkpoint: Any = None


def get_react_graph(use_checkpoint: bool = True) -> Any:
    """获取编译后的图(惰性初始化).

    Args:
        use_checkpoint: True 启用 PG 持久化(M9.3 默认),
                        False 不持久化(用于无 thread_id 调用).
    """
    global _compiled_graph_with_checkpoint, _compiled_graph_without_checkpoint

    if use_checkpoint:
        if _compiled_graph_with_checkpoint is None:
            cp = get_checkpointer()
            _compiled_graph_with_checkpoint = build_react_graph(checkpointer=cp)
        return _compiled_graph_with_checkpoint
    else:
        if _compiled_graph_without_checkpoint is None:
            _compiled_graph_without_checkpoint = build_react_graph()
        return _compiled_graph_without_checkpoint


async def run_react_agent(
    user_message: str,
    system_prompt: str | None = None,
    initial_messages: list[dict[str, Any]] | None = None,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """运行 ReAct Agent.

    Args:
        user_message: 当前用户消息
        system_prompt: 系统提示,initial_messages 为 None 时使用
        initial_messages: 可选 — 调用方预构造的 messages(M8 Memory 用)
        thread_id: 可选 — LangGraph thread_id(M9.3 状态持久化用,
                   通常传 session_id).不传则不持久化.

    Returns:
        dict: final_reply / iterations / tool_calls_log
    """
    use_checkpoint = thread_id is not None
    app = get_react_graph(use_checkpoint=use_checkpoint)

    if initial_messages is not None:
        messages: list[Any] = list(initial_messages)
    else:
        messages = [
            {"role": "system", "content": system_prompt or REACT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    initial_state: ReactState = {
        "messages": messages,
        "iteration": 0,
        "tool_calls_log": [],
    }

    if use_checkpoint:
        config = {"configurable": {"thread_id": thread_id}}
        final_state = await app.ainvoke(initial_state, config=config)
    else:
        final_state = await app.ainvoke(initial_state)

    # 提取最终回答
    final_reply = ""
    for msg in reversed(final_state["messages"]):
        content = msg.content if hasattr(msg, "content") else msg.get("content")
        msg_type = msg.type if hasattr(msg, "type") else msg.get("role")
        if content and msg_type in ("ai", "assistant"):
            final_reply = content
            break

    return {
        "final_reply": final_reply,
        "iterations": final_state["iteration"],
        "tool_calls_log": final_state["tool_calls_log"],
    }


__all__ = ["run_react_agent", "ReactState", "build_react_graph"]
