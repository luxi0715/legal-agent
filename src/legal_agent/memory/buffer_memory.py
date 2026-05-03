"""Buffer Memory: short-term sliding window of recent messages (M8 ⭐).

Redis LIST 实现.
key: buffer:{session_id}
存储:JSON 序列化的 {role, content, ts} 字典.

设计:
  • LPUSH 头插新消息
  • LTRIM 0 (BUFFER_MAX_ITEMS-1) 保留最新 N 条,自动驱逐旧消息
  • LRANGE 0 -1 读取后 reverse → 时间正序
  • TTL 7 天防孤儿数据

注:redis-py 5.x 异步签名返回 Awaitable[T] | T,与同步模式共用.
   mypy 无法区分,需要 type: ignore[misc] 抑制误报.
"""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import UUID

from legal_agent.db.redis_client import get_redis

# 7 轮 = 14 条消息(user + assistant 各 7)
BUFFER_MAX_ITEMS = 14
BUFFER_TTL_SECONDS = 7 * 24 * 3600  # 7 天


def _key(session_id: UUID) -> str:
    """Redis key 格式:buffer:{session_id}."""
    return f"buffer:{session_id}"


async def append_to_buffer(
    session_id: UUID,
    role: str,
    content: str,
) -> None:
    """追加一条消息到 buffer 头部,自动 TRIM 保留最新 BUFFER_MAX_ITEMS 条."""
    redis = get_redis()
    payload = json.dumps(
        {"role": role, "content": content, "ts": time.time()},
        ensure_ascii=False,
    )
    key = _key(session_id)
    pipe = redis.pipeline()
    pipe.lpush(key, payload)
    pipe.ltrim(key, 0, BUFFER_MAX_ITEMS - 1)
    pipe.expire(key, BUFFER_TTL_SECONDS)
    await pipe.execute()


async def get_buffer(session_id: UUID) -> list[dict[str, Any]]:
    """读取 buffer 全部消息(时间正序).

    LPUSH 头插 → list[0] 是最新 → reverse 得到时间正序.
    """
    redis = get_redis()
    raw: list[str] = await redis.lrange(_key(session_id), 0, -1)  # type: ignore[misc]
    return [json.loads(item) for item in reversed(raw)]


async def get_oldest_n(session_id: UUID, n: int) -> list[dict[str, Any]]:
    """取出最旧的 n 条消息(给 Summary 压缩用,不删除).

    LRANGE -n -1 拿 list 尾部 n 条 = 最旧 n 条,顺序为 [新→旧],reverse 得时间正序.
    """
    if n <= 0:
        return []
    redis = get_redis()
    raw: list[str] = await redis.lrange(_key(session_id), -n, -1)  # type: ignore[misc]
    return [json.loads(item) for item in reversed(raw)]


async def trim_oldest_n(session_id: UUID, n: int) -> int:
    """删除最旧的 n 条消息.

    Returns:
        实际删除条数(min(n, 当前长度))
    """
    if n <= 0:
        return 0
    redis = get_redis()
    key = _key(session_id)
    length: int = await redis.llen(key)  # type: ignore[misc]
    if length == 0:
        return 0
    actually_deleted = min(n, length)
    keep = length - actually_deleted
    if keep == 0:
        await redis.delete(key)
    else:
        await redis.ltrim(key, 0, keep - 1)  # type: ignore[misc]
    return actually_deleted


async def clear_buffer(session_id: UUID) -> bool:
    """清空 buffer (GDPR 主动遗忘).

    Returns:
        True 表示真删了,False 表示 buffer 本来就空.
    """
    redis = get_redis()
    deleted = await redis.delete(_key(session_id))
    return bool(deleted)


async def get_buffer_size(session_id: UUID) -> int:
    """获取 buffer 当前长度."""
    redis = get_redis()
    size: int = await redis.llen(_key(session_id))  # type: ignore[misc]
    return size


__all__ = [
    "BUFFER_MAX_ITEMS",
    "append_to_buffer",
    "clear_buffer",
    "get_buffer",
    "get_buffer_size",
    "get_oldest_n",
    "trim_oldest_n",
]
