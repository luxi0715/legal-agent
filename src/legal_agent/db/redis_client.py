"""Redis async client."""

from redis.asyncio import Redis, from_url

from legal_agent.core.config import get_settings

_redis: Redis | None = None


async def init_redis() -> Redis:
    """Initialize the global Redis client.

    M9.1: 强制 ping() 完成首次握手,避免运行时第 1 次操作慢 10-20 秒.
    握手成本提前到启动时支付,而不是用户第一次请求时.
    """
    global _redis
    if _redis is not None:
        return _redis

    settings = get_settings()
    _redis = from_url(settings.redis_url, decode_responses=True)

    # ⭐ M9.1 — 强制完成 socket 握手
    await _redis.ping()  # type: ignore[misc]

    return _redis


async def close_redis() -> None:
    """Close the Redis client."""
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def get_redis() -> Redis:
    """Return the initialized Redis client."""
    if _redis is None:
        raise RuntimeError("Redis not initialized; call init_redis first")
    return _redis
