"""LangGraph Checkpointer (M9.3): PostgreSQL 持久化 ReAct 状态.

设计哲学:
  • 复用 PG 连接(独立连接池,跟 asyncpg pool 分开)
  • 模块级单例 + 惰性初始化
  • 首次启动自动建 4 张 checkpoint 表

应用场景:
  • 网络中断恢复
  • 服务重启不丢上下文
  • 长任务追踪每步状态
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from legal_agent.core.config import get_settings

logger = logging.getLogger(__name__)


_checkpointer: AsyncPostgresSaver | None = None
_checkpointer_ctx = None  # async context manager 句柄


async def init_checkpointer() -> AsyncPostgresSaver:
    """初始化 PG Checkpointer + 建表(首次).

    在 lifespan startup 调一次.
    """
    global _checkpointer, _checkpointer_ctx
    if _checkpointer is not None:
        return _checkpointer

    settings = get_settings()
    # langgraph-checkpoint-postgres 用 libpq URL 格式,不带 +asyncpg
    libpq_dsn = settings.postgres_dsn.replace("postgresql+asyncpg://", "postgresql://")

    _checkpointer_ctx = AsyncPostgresSaver.from_conn_string(libpq_dsn)
    _checkpointer = await _checkpointer_ctx.__aenter__()
    await _checkpointer.setup()  # 首次建 4 张表(幂等)

    logger.info("Checkpointer initialized")
    return _checkpointer


async def close_checkpointer() -> None:
    """关闭 Checkpointer 连接.lifespan shutdown 调."""
    global _checkpointer, _checkpointer_ctx
    if _checkpointer_ctx is not None:
        await _checkpointer_ctx.__aexit__(None, None, None)
        _checkpointer = None
        _checkpointer_ctx = None


def get_checkpointer() -> AsyncPostgresSaver:
    """返回已初始化的 Checkpointer."""
    if _checkpointer is None:
        raise RuntimeError("Checkpointer not initialized; call init_checkpointer first")
    return _checkpointer


__all__ = ["init_checkpointer", "close_checkpointer", "get_checkpointer"]
