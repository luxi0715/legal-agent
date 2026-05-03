"""Hard Memory: persistent user identity facts (M8 ⭐).

KV 设计:user_facts 表存任意 (user_id, key, value) 三元组.
读快、写易、扩展强.
"""

from __future__ import annotations

from uuid import UUID

from legal_agent.db.postgres import get_postgres_pool


async def get_user_facts(user_id: UUID) -> dict[str, str]:
    """返回用户所有事实(key → value).

    Returns:
        dict: 例 {"location": "北京", "occupation": "程序员"}
              用户无任何 fact 时返回空 dict.
    """
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT key, value FROM user_facts WHERE user_id = $1",
            user_id,
        )
    return {r["key"]: r["value"] for r in rows}


async def get_user_facts_with_meta(user_id: UUID) -> list[dict[str, object]]:
    """返回用户所有事实 + 元信息(供调试 / Persona 防漂移用).

    Returns:
        list[dict]: 每个 dict 含 key / value / confidence / extracted_at.
    """
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT key, value, confidence, extracted_at
            FROM user_facts
            WHERE user_id = $1
            ORDER BY extracted_at DESC
            """,
            user_id,
        )
    return [dict(r) for r in rows]


async def upsert_user_fact(
    user_id: UUID,
    key: str,
    value: str,
    confidence: float = 1.0,
) -> None:
    """新增或更新一个用户事实.

    冲突时(同 user_id + key)新值覆盖旧值,
    extracted_at 更新为现在时间.
    """
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO user_facts (user_id, key, value, confidence)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (user_id, key)
            DO UPDATE SET
                value = EXCLUDED.value,
                confidence = EXCLUDED.confidence,
                extracted_at = NOW()
            """,
            user_id,
            key,
            value,
            confidence,
        )


async def delete_user_fact(user_id: UUID, key: str) -> bool:
    """删除一个用户事实(GDPR / 主动遗忘).

    Returns:
        bool: True 表示真的删掉了,False 表示该 key 本来就不存在.
    """
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM user_facts WHERE user_id = $1 AND key = $2",
            user_id,
            key,
        )
    # asyncpg 返回 'DELETE N',N 是删除行数
    return bool(result.endswith(" 1"))


async def delete_all_user_facts(user_id: UUID) -> int:
    """清空用户所有事实(GDPR 完整删除权).

    Returns:
        int: 删除的行数.
    """
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM user_facts WHERE user_id = $1",
            user_id,
        )
    # 'DELETE N' → 取最后的数字
    try:
        return int(result.rsplit(" ", 1)[-1])
    except (ValueError, IndexError):
        return 0


__all__ = [
    "get_user_facts",
    "get_user_facts_with_meta",
    "upsert_user_fact",
    "delete_user_fact",
    "delete_all_user_facts",
]
