"""Tool Use Loop: LLM 决策 → 程序执行 → 喂回 → LLM 再决策..."""

import json
from dataclasses import dataclass, field
from typing import Any

from legal_agent.agent.llm_client import DEFAULT_SYSTEM_PROMPT, get_llm_client
from legal_agent.core.config import get_settings
from legal_agent.tools.definitions import ALL_TOOLS
from legal_agent.tools.dispatcher import dispatch_tool

# 安全阈值:防止 LLM 在工具间死循环.
# 法律咨询场景 5 轮足够,通常 1-2 轮就结束.
MAX_TOOL_ITERATIONS = 5


@dataclass
class ToolCallRecord:
    """单次工具调用的记录(用于可观测性)."""

    iteration: int
    name: str
    arguments: dict[str, Any]
    result_preview: str  # 截断的结果,完整结果太长不存


@dataclass
class AgentTrace:
    """Agent 完整执行轨迹.

    用途:
    - 调试:工具调用顺序、参数是否合理
    - 监控:平均每个 query 调几次工具
    - 评测:LLM 是否选对工具
    """

    final_reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    iterations: int = 0
    hit_max_iter: bool = False


# Agent 模式专用 system prompt(在默认基础上加 Tool Use 引导)
AGENT_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT + (
    "\n\n你有以下工具可以调用:\n"
    "- legal_search:用于回答开放性法律问题(概念解释、维权咨询、法律责任分析)\n"
    "- get_law_article:用于精确查询特定法律的特定条款(用户已知法律名 + 条款号)\n"
    "请根据用户问题选择合适的工具.对闲聊或自我介绍,无需调用工具,直接回答即可."
)


async def run_agent_loop(
    user_message: str,
    system_prompt: str | None = None,
) -> AgentTrace:
    """运行 Tool Use 循环直到 LLM 不再调工具.

    流程:
        1. 把 message + tools 喂给 LLM
        2. 如果 LLM 输出 tool_calls,执行所有工具,把结果喂回
        3. LLM 再决策:再调工具 / 输出最终回答
        4. 直到 LLM 输出纯文本回答(无 tool_calls)
        5. 或达到 MAX_TOOL_ITERATIONS 强制停止

    Args:
        user_message: 用户的原始问题.
        system_prompt: 可选自定义 system prompt(覆盖 AGENT_SYSTEM_PROMPT).

    Returns:
        AgentTrace 包含最终回复 + 完整工具调用轨迹.
    """
    client = get_llm_client()
    settings = get_settings()

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt or AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    trace = AgentTrace(final_reply="")

    for iteration in range(MAX_TOOL_ITERATIONS):
        trace.iterations = iteration + 1

        response = await client.chat.completions.create(  # type: ignore[call-overload]
            model=settings.deepseek_model,
            messages=messages,
            tools=ALL_TOOLS,
            tool_choice="auto",
            temperature=0.3,
        )
        msg = response.choices[0].message

        # 没有工具调用 → 终止循环,返回最终回答
        if not msg.tool_calls:
            trace.final_reply = msg.content or ""
            return trace

        # 有工具调用 → 执行所有工具,把结果喂回
        # 注意:DeepSeek 可能一次返回多个并行 tool_calls
        messages.append(msg.model_dump())  # 保留 LLM 输出的 tool_calls 块

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as e:
                fn_args = {}
                tool_result = f"[参数解析错误] {e}"
            else:
                tool_result = await dispatch_tool(fn_name, fn_args)

            # 记录到 trace
            trace.tool_calls.append(
                ToolCallRecord(
                    iteration=iteration + 1,
                    name=fn_name,
                    arguments=fn_args,
                    result_preview=tool_result[:200],
                )
            )

            # 喂回 LLM
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )

    # 达到上限,强制返回
    trace.hit_max_iter = True
    trace.final_reply = "(达到最大工具调用次数,任务可能太复杂.建议简化问题再试.)"
    return trace


__all__ = ["AgentTrace", "ToolCallRecord", "run_agent_loop"]
