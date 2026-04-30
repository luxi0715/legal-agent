"""单元测试:Lost in the Middle 重排逻辑.

纯算法,不连数据库,不调 API.
"""

from typing import cast

from legal_agent.rag.context_reorder import reorder_for_lost_in_the_middle
from legal_agent.rag.reranker import RerankedChunk


def make_chunk(content: str, score: float) -> RerankedChunk:
    """构造测试用 chunk."""
    return cast(
        RerankedChunk,
        {
            "content": content,
            "rerank_score": score,
            "original_score": None,
            "metadata": {},
            "sources": [],
        },
    )


def test_empty_input() -> None:
    """空列表应该返回空列表."""
    assert reorder_for_lost_in_the_middle([]) == []


def test_single_item() -> None:
    """1 条不重排."""
    chunks = [make_chunk("A", 0.9)]
    result = reorder_for_lost_in_the_middle(chunks)
    assert len(result) == 1
    assert result[0]["content"] == "A"


def test_two_items_unchanged() -> None:
    """2 条不重排."""
    chunks = [make_chunk("A", 0.9), make_chunk("B", 0.7)]
    result = reorder_for_lost_in_the_middle(chunks)
    assert [r["content"] for r in result] == ["A", "B"]


def test_five_items_reordered() -> None:
    """5 条应该重排成 [T1, T3, T5, T4, T2]."""
    chunks = [
        make_chunk("T1", 0.9),
        make_chunk("T2", 0.8),
        make_chunk("T3", 0.7),
        make_chunk("T4", 0.6),
        make_chunk("T5", 0.5),
    ]
    result = reorder_for_lost_in_the_middle(chunks)
    contents = [r["content"] for r in result]
    assert contents == ["T1", "T3", "T5", "T4", "T2"]


def test_seven_items_reordered() -> None:
    """7 条应该重排成 [T1, T3, T5, T7, T6, T4, T2]."""
    chunks = [make_chunk(f"T{i}", 1.0 - i * 0.1) for i in range(1, 8)]
    result = reorder_for_lost_in_the_middle(chunks)
    contents = [r["content"] for r in result]
    assert contents == ["T1", "T3", "T5", "T7", "T6", "T4", "T2"]


def test_does_not_mutate_input() -> None:
    """函数不应修改输入列表."""
    chunks = [make_chunk(f"T{i}", 1.0 - i * 0.1) for i in range(1, 6)]
    original_order = [c["content"] for c in chunks]
    _ = reorder_for_lost_in_the_middle(chunks)
    assert [c["content"] for c in chunks] == original_order


def test_strongest_at_start() -> None:
    """最强项必须在开头."""
    chunks = [make_chunk(f"T{i}", 1.0 - i * 0.1) for i in range(1, 6)]
    result = reorder_for_lost_in_the_middle(chunks)
    assert result[0]["rerank_score"] == max(c["rerank_score"] for c in chunks)


def test_second_strongest_at_end() -> None:
    """次强项应该在结尾(Top2 应该是 5 条情况下的最后一条)."""
    chunks = [make_chunk(f"T{i}", 1.0 - i * 0.1) for i in range(1, 6)]
    result = reorder_for_lost_in_the_middle(chunks)
    # 输入是 [T1=0.9, T2=0.8, T3=0.7, T4=0.6, T5=0.5]
    # 输出应该是 [T1, T3, T5, T4, T2],最后一个是 T2 = 0.8
    assert result[-1]["content"] == "T2"
