"""Text chunking strategies for RAG."""

import json
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunk(TypedDict):
    """A single chunk of text ready for embedding."""

    chunk_id: str
    content: str
    metadata: dict[str, str | int | None]


# 配置:每块 ~400 字,重叠 50 字
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# 中文友好的分隔符(从粗到细)
SEPARATORS = [
    "\n\n",
    "\n",
    "。",
    ";",
    ",",
    " ",
    "",
]


def get_splitter() -> RecursiveCharacterTextSplitter:
    """Create the recursive splitter with Chinese-friendly settings."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )


def chunk_article(
    article_no: str,
    content: str,
    law_title: str,
    chapter: str | None,
) -> list[Chunk]:
    """Chunk a single article. Short articles stay as one chunk."""
    splitter = get_splitter()

    if len(content) <= CHUNK_SIZE:
        return [
            Chunk(
                chunk_id=str(uuid4()),
                content=content,
                metadata={
                    "law_title": law_title,
                    "chapter": chapter,
                    "article_no": article_no,
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
            )
        ]

    pieces = splitter.split_text(content)
    return [
        Chunk(
            chunk_id=str(uuid4()),
            content=piece,
            metadata={
                "law_title": law_title,
                "chapter": chapter,
                "article_no": article_no,
                "chunk_index": i,
                "total_chunks": len(pieces),
            },
        )
        for i, piece in enumerate(pieces)
    ]


def chunk_law_file(parsed_path: Path) -> list[Chunk]:
    """Chunk all articles in one parsed law file."""
    data = json.loads(parsed_path.read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for article in data["articles"]:
        chunks.extend(
            chunk_article(
                article_no=article["article_no"],
                content=article["content"],
                law_title=data["title"],
                chapter=article.get("chapter"),
            )
        )
    return chunks


def chunk_all(input_dir: Path, output_path: Path) -> int:
    """Chunk all parsed laws into a single JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    files = sorted(input_dir.glob("*.parsed.json"))
    print(f"Chunking {len(files)} parsed law files...\n")

    with output_path.open("w", encoding="utf-8") as f:
        for i, parsed_path in enumerate(files):
            chunks = chunk_law_file(parsed_path)
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            if i < 30:
                print(f"  {parsed_path.stem}: {len(chunks)} chunks")
            total += len(chunks)

    print(f"\nTotal: {total} chunks -> {output_path}")
    return total


if __name__ == "__main__":
    n = chunk_all(Path("data/parsed"), Path("data/chunks/all_chunks.jsonl"))
