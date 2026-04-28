"""Redis async client."""

from redis.asyncio import Redis, from_url

from legal_agent.core.config import get_settings

_redis: Redis | None = None


async def init_redis() -> Redis:
    """Initialize the global Redis client."""
    global _redis
    if _redis is not None:
        return _redis

    settings = get_settings()
    _redis = from_url(settings.redis_url, decode_responses=True)
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
