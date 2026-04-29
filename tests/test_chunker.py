"""Test text chunker."""

from legal_agent.rag.chunker import chunk_article


def test_short_article_stays_whole() -> None:
    """Short articles should not be split."""
    chunks = chunk_article(
        article_no="第一条",
        content="这是一条短法律。",
        law_title="测试法",
        chapter=None,
    )
    assert len(chunks) == 1
    assert chunks[0]["content"] == "这是一条短法律。"


def test_long_article_gets_split() -> None:
    """Long articles should be split into chunks."""
    long_content = "这是一段很长的内容。" * 100
    chunks = chunk_article(
        article_no="第二条",
        content=long_content,
        law_title="测试法",
        chapter="第一章",
    )
    assert len(chunks) > 1
    for c in chunks:
        assert c["metadata"]["law_title"] == "测试法"
        assert c["metadata"]["article_no"] == "第二条"
        assert c["metadata"]["chapter"] == "第一章"


def test_metadata_contains_chunk_position() -> None:
    """Each chunk should know its position in the article."""
    long_content = "测试内容。" * 200
    chunks = chunk_article("第三条", long_content, "测试法", None)
    indices = [c["metadata"]["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))
