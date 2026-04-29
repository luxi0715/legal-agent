"""Index embedded chunks into PostgreSQL pgvector."""

import asyncio
import json
from pathlib import Path

from tqdm import tqdm

from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool

BATCH_SIZE = 100


async def clear_embeddings_table() -> None:
    """Clear all rows in embeddings table for fresh indexing."""
    pool = await init_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute("TRUNCATE TABLE embeddings RESTART IDENTITY")
    print("Cleared embeddings table")


async def insert_embeddings(jsonl_path: Path) -> int:
    """Insert all embedded chunks into the database."""
    pool = await init_postgres_pool()

    with jsonl_path.open(encoding="utf-8") as f:
        total = sum(1 for _ in f)

    inserted = 0
    batch: list[tuple[str, str, str, str, str]] = []

    with jsonl_path.open(encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="Inserting"):
            record = json.loads(line)

            embedding_str = "[" + ",".join(str(x) for x in record["embedding"]) + "]"

            batch.append(
                (
                    "law",
                    record["metadata"].get("article_no", "unknown"),
                    record["content"],
                    embedding_str,
                    json.dumps(record["metadata"], ensure_ascii=False),
                )
            )

            if len(batch) >= BATCH_SIZE:
                async with pool.acquire() as conn:
                    await conn.executemany(
                        """
                        INSERT INTO embeddings
                            (source_type, source_id, content, embedding, metadata)
                        VALUES ($1, $2, $3, $4::vector, $5::jsonb)
                        """,
                        batch,
                    )
                inserted += len(batch)
                batch = []

    if batch:
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO embeddings
                    (source_type, source_id, content, embedding, metadata)
                VALUES ($1, $2, $3, $4::vector, $5::jsonb)
                """,
                batch,
            )
        inserted += len(batch)

    return inserted


async def build_hnsw_index() -> None:
    """Build the HNSW index for fast cosine similarity search."""
    pool = await init_postgres_pool()
    async with pool.acquire() as conn:
        await conn.execute("DROP INDEX IF EXISTS idx_embeddings_vector")

        print("Building HNSW index... (1-3 minutes)")
        await conn.execute(
            """
            CREATE INDEX idx_embeddings_vector ON embeddings
            USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
            """
        )
        print("HNSW index built")


async def main() -> None:
    """Run the full indexing pipeline."""
    await clear_embeddings_table()

    n = await insert_embeddings(Path("data/chunks/all_chunks_embedded.jsonl"))
    print(f"\nInserted {n} chunks into embeddings table")

    await build_hnsw_index()
    await close_postgres_pool()


if __name__ == "__main__":
    asyncio.run(main())
