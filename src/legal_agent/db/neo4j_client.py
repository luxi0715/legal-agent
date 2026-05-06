"""Neo4j async client (M11.1).

跟 postgres.py / redis_client.py 一致的单例模式.

⭐ M11 设计要点:
  - init 内调 verify_connectivity() 强制握手
  - 失败立即抛,避免冷启动延迟
  - 学到 M9.1 给 Redis 加 ping 的同样教训
"""

from neo4j import AsyncDriver, AsyncGraphDatabase

from legal_agent.core.config import get_settings

_driver: AsyncDriver | None = None


async def init_neo4j() -> AsyncDriver:
    """Initialize the global Neo4j driver."""
    global _driver
    if _driver is not None:
        return _driver

    settings = get_settings()
    _driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    # ⭐ 强制握手,启动时立即暴露连接错误
    await _driver.verify_connectivity()
    return _driver


async def close_neo4j() -> None:
    """Close the Neo4j driver."""
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


def get_neo4j() -> AsyncDriver:
    """Return the initialized driver."""
    if _driver is None:
        raise RuntimeError("Neo4j not initialized; call init_neo4j first")
    return _driver
