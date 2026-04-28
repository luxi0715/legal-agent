"""Message persistence helpers."""

from uuid import UUID, uuid4

from legal_agent.db.postgres import get_postgres_pool


async def get_or_create_session(session_id: UUID | None = None) -> UUID:
    """Return existing session id, or create a new one."""
    pool = get_postgres_pool()

    if session_id is None:
        session_id = uuid4()

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO sessions (id) VALUES ($1)
            ON CONFLICT (id) DO NOTHING
            """,
            session_id,
        )
    return session_id


async def save_message(
    session_id: UUID,
    role: str,
    content: str,
) -> None:
    """Save a single message."""
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO messages (session_id, role, content)
            VALUES ($1, $2, $3)
            """,
            session_id,
            role,
            content,
        )


async def list_session_messages(session_id: UUID, limit: int = 20) -> list[dict[str, str]]:
    """Fetch the most recent N messages of a session, oldest first."""
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = $1
            ORDER BY created_at DESC
            LIMIT $2
            """,
            session_id,
            limit,
        )
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
