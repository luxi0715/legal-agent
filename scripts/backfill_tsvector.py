"""Backfill content_tsv for all existing embeddings rows."""

import asyncio

import asyncpg
from tqdm import tqdm

from legal_agent.core.config import get_settings
from legal_agent.rag.tokenizer import tokenize_for_tsvector

BATCH_SIZE = 500


async def backfill() -> None:
    settings = get_settings()
    conn = await asyncpg.connect(settings.postgres_dsn)

    # 1. 统计需要回填的数量
    total = await conn.fetchval("SELECT COUNT(*) FROM embeddings WHERE content_tsv IS NULL")
    print(f"Rows to backfill: {total}")

    if total == 0:
        print("Nothing to do.")
        await conn.close()
        return

    # 2. 分批拉取 + 更新(BATCH_SIZE 为模块级常量,见文件顶部)
    offset = 0
    processed = 0
    pbar = tqdm(total=total, desc="Backfilling tsvector")

    while True:
        rows = await conn.fetch(
            "SELECT id, content FROM embeddings WHERE content_tsv IS NULL ORDER BY id LIMIT $1",
            BATCH_SIZE,
        )
        if not rows:
            break

        for row in rows:
            tokens = tokenize_for_tsvector(row["content"])
            await conn.execute(
                "UPDATE embeddings SET content_tsv = to_tsvector('simple', $1) WHERE id = $2",
                tokens,
                row["id"],
            )
            processed += 1
            pbar.update(1)

        offset += BATCH_SIZE

    pbar.close()
    print(f"\nDone. Processed {processed} rows.")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(backfill())
