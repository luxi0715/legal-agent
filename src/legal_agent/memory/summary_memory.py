"""Summary Memory: rolling-compressed conversation summary (M8 ⭐).

PostgreSQL session_summaries 表 + LLM 滚动压缩.

何时触发:Buffer 满 BUFFER_MAX_ITEMS 时,M8.6 Memory Manager 调用
         compress_and_update,取最旧 N 条压缩进 Summary.

压缩策略:
  • 输入:已有 summary + 新对话片段
  • 输出:替换式新 summary(不是追加)
  • 长度:~500 字内,LLM 自我裁剪
"""

from __future__ import annotations

from uuid import UUID

from legal_agent.agent.llm_client import get_llm_client
from legal_agent.core.config import get_settings
from legal_agent.db.postgres import get_postgres_pool

# 摘要长度上限提示给 LLM
SUMMARY_MAX_CHARS = 500


SUMMARY_COMPRESSION_PROMPT = """你是对话压缩专家.把已有摘要 + 新对话片段融合成一份新摘要.

要求:
1. 保留用户的关键信息:身份(职业/地区/家庭)、咨询主题、关键决策
2. 删除寒暄、重复、客套内容
3. 总长度不超过 {max_chars} 字
4. 用第三人称叙述,不用对话格式
5. 输出 纯文本,不要任何标题或前缀

【已有摘要】
{old_summary}

【新对话片段】
{new_messages}

【新摘要】"""


async def get_summary(session_id: UUID) -> str:
    """读当前 session 摘要.

    Returns:
        str: 摘要文本.session 无摘要时返回空字符串.
    """
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT summary FROM session_summaries WHERE session_id = $1",
            session_id,
        )
    return row["summary"] if row else ""


async def upsert_summary(
    session_id: UUID,
    user_id: UUID,
    summary: str,
    turn_count: int = 0,
) -> None:
    """写入或更新摘要(纯数据库操作,不调 LLM).

    冲突时(同 session_id)新值覆盖旧值.
    """
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO session_summaries
                (session_id, user_id, summary, turn_count, last_updated)
            VALUES ($1, $2, $3, $4, NOW())
            ON CONFLICT (session_id)
            DO UPDATE SET
                summary = EXCLUDED.summary,
                turn_count = EXCLUDED.turn_count,
                last_updated = NOW()
            """,
            session_id,
            user_id,
            summary,
            turn_count,
        )


async def compress_and_update(
    session_id: UUID,
    user_id: UUID,
    new_messages: list[dict[str, str]],
    turn_count: int = 0,
) -> str:
    """⭐ 核心编排:LLM 压缩 + 更新 Summary.

    Args:
        session_id: 会话标识
        user_id: 用户标识
        new_messages: 要压缩的新对话片段,每条含 role / content
        turn_count: 累计轮数(给监控用)

    Returns:
        str: LLM 生成的新摘要

    流程:
        1. 读已有 summary
        2. 格式化 new_messages 为 prompt 友好文本
        3. 调 LLM 压缩(temperature=0.3 求稳)
        4. 写回数据库
        5. 返回新摘要
    """
    if not new_messages:
        return await get_summary(session_id)

    old_summary = await get_summary(session_id)

    # 把对话片段格式化成可读文本
    formatted_msgs = "\n".join(f"{m['role']}: {m['content']}" for m in new_messages)

    prompt = SUMMARY_COMPRESSION_PROMPT.format(
        max_chars=SUMMARY_MAX_CHARS,
        old_summary=old_summary if old_summary else "(暂无)",
        new_messages=formatted_msgs,
    )

    client = get_llm_client()
    settings = get_settings()
    response = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    new_summary = (response.choices[0].message.content or "").strip()

    await upsert_summary(
        session_id=session_id,
        user_id=user_id,
        summary=new_summary,
        turn_count=turn_count,
    )
    return new_summary


async def delete_summary(session_id: UUID) -> bool:
    """删除摘要(GDPR / 重置).

    Returns:
        True 表示真的删了,False 表示该 session 本来就没摘要.
    """
    pool = get_postgres_pool()
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM session_summaries WHERE session_id = $1",
            session_id,
        )
    return bool(result.endswith(" 1"))


__all__ = [
    "SUMMARY_MAX_CHARS",
    "compress_and_update",
    "delete_summary",
    "get_summary",
    "upsert_summary",
]
