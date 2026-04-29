"""Generate embeddings using Alibaba DashScope."""

import asyncio
import json
from pathlib import Path
from typing import Any

import dashscope
from tqdm import tqdm

from legal_agent.core.config import get_settings

# DashScope text-embedding-v3 单次最多 10 条(阿里云 2024 年后限制)
BATCH_SIZE = 10


def _setup_dashscope() -> None:
    settings = get_settings()
    dashscope.api_key = settings.dashscope_api_key


async def embed_batch(texts: list[str]) -> list[list[float]]:
    _setup_dashscope()
    settings = get_settings()
    loop = asyncio.get_event_loop()

    def _call() -> Any:
        return dashscope.TextEmbedding.call(
            model=settings.embedding_model,
            input=texts,
            dimension=settings.embedding_dim,
        )

    resp = await loop.run_in_executor(None, _call)

    if resp.status_code != 200:
        raise RuntimeError(f"DashScope error: {resp.code} - {resp.message}")

    embeddings = [item["embedding"] for item in resp.output["embeddings"]]
    return embeddings


async def embed_chunks_file(
    chunks_path: Path,
    output_path: Path,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[dict[str, Any]] = []
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    done_ids: set[str] = set()
    if output_path.exists():
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec["chunk_id"])
                except Exception:
                    continue

    todo = [c for c in chunks if c["chunk_id"] not in done_ids]
    print(f"Total chunks: {len(chunks)}")
    print(f"Already embedded: {len(done_ids)}")
    print(f"To embed: {len(todo)}")

    if not todo:
        print("\nNothing to do.")
        return 0

    failed_count = 0
    with output_path.open("a", encoding="utf-8") as out:
        for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="Embedding"):
            batch = todo[i : i + BATCH_SIZE]
            texts = [c["content"] for c in batch]

            try:
                vectors = await embed_batch(texts)
            except Exception as e:
                print(f"\n  Batch {i} failed: {e}")
                failed_count += len(batch)
                await asyncio.sleep(3)
                try:
                    vectors = await embed_batch(texts)
                except Exception as e2:
                    print(f"  Retry also failed: {e2}, skipping")
                    continue

            for chunk, vector in zip(batch, vectors, strict=True):
                record = {**chunk, "embedding": vector}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

    if failed_count:
        print(f"\n{failed_count} chunks failed, you can rerun to retry")
    return len(todo) - failed_count


async def main() -> None:
    n = await embed_chunks_file(
        Path("data/chunks/all_chunks.jsonl"),
        Path("data/chunks/all_chunks_embedded.jsonl"),
    )
    print(f"\nEmbedded {n} chunks this run")


if __name__ == "__main__":
    asyncio.run(main())
