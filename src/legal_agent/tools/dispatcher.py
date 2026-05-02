"""Tool dispatcher: maps tool name → actual async function."""

from collections.abc import Awaitable, Callable
from typing import Any

from legal_agent.tools.get_law_article import get_law_article
from legal_agent.tools.legal_search import legal_search

# 工具名 → 异步函数(必须跟 definitions.py 里的 name 字段对齐)
TOOL_REGISTRY: dict[str, Callable[..., Awaitable[str]]] = {
    "legal_search": legal_search,
    "get_law_article": get_law_article,
}


async def dispatch_tool(name: str, arguments: dict[str, Any]) -> str:
    """根据工具名调用对应函数.

    Args:
        name: LLM 请求的工具名(从 tool_use 块中解析得到).
        arguments: LLM 提供的参数(已 JSON 解析的 dict).

    Returns:
        工具返回的字符串结果(作为 tool_result 喂回 LLM).
        所有错误都被捕获并转成字符串,不会抛异常 — 因为 agent loop
        不应该被工具异常中断.
    """
    fn = TOOL_REGISTRY.get(name)
    if fn is None:
        return f"[错误] 未知工具: {name}.可用工具: {list(TOOL_REGISTRY.keys())}"

    try:
        return await fn(**arguments)
    except TypeError as e:
        # 参数对不上(例如缺必填、类型错)
        return f"[参数错误] 调用 {name} 失败: {e}"
    except Exception as e:
        # 其他异常(数据库挂、网络抖等)
        return f"[执行错误] {name} 执行失败: {type(e).__name__}: {e}"


__all__ = ["TOOL_REGISTRY", "dispatch_tool"]
