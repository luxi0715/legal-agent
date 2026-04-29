"""Test law parser."""

from legal_agent.rag.parser import (
    extract_title,
    split_into_articles,
    strip_frontmatter,
)


def test_strip_frontmatter() -> None:
    """YAML frontmatter should be stripped."""
    text = "---\ntitle: 测试法律\n---\n# 测试法律\n第一条 内容"
    result = strip_frontmatter(text)

    assert "title: 测试法律" not in result
    assert "# 测试法律" in result
    assert "第一条 内容" in result


def test_extract_title() -> None:
    """Should extract first markdown H1 title."""
    text = "# 中华人民共和国测试法\n\n第一条 内容"
    title = extract_title(text, fallback="fallback")

    assert title == "中华人民共和国测试法"


def test_extract_title_fallback() -> None:
    """Should use fallback when no H1 title exists."""
    text = "第一条 内容"
    title = extract_title(text, fallback="测试文件名")

    assert title == "测试文件名"


def test_split_into_articles_basic() -> None:
    """Should split a multi-article text correctly."""
    text = (
        "第一编 总则\n"
        "第一章 基本规定\n"
        "第一条 这是第一条的内容。\n"
        "第二条 这是第二条的内容,有多行。\n"
        "包含详细说明。\n"
        "第三条 简短的第三条。"
    )

    articles = split_into_articles(text)

    assert len(articles) == 3
    assert articles[0]["article_no"] == "第一条"
    assert "第一条的内容" in articles[0]["content"]
    assert articles[1]["article_no"] == "第二条"
    assert "包含详细说明" in articles[1]["content"]
    assert articles[2]["article_no"] == "第三条"


def test_split_into_articles_empty() -> None:
    """Text without articles should return empty list."""
    assert split_into_articles("没有条款的纯文本") == []
