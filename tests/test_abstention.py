"""单元测试:Abstention Strategy 三信号决策逻辑."""

from typing import cast

from legal_agent.rag.abstention import decide_abstention
from legal_agent.rag.reranker import RerankedChunk


def make_chunk(
    rerank_score: float,
    rrf_score: float = 0.025,
    law_title: str = "中华人民共和国民法典",
) -> RerankedChunk:
    """构造测试 chunk."""
    return cast(
        RerankedChunk,
        {
            "content": "测试内容",
            "rerank_score": rerank_score,
            "original_score": rrf_score,
            "metadata": {"law_title": law_title},
            "sources": ["vector"],
        },
    )


def test_empty_input_abstains() -> None:
    """空召回必须拒答."""
    decision = decide_abstention([])
    assert decision.should_abstain
    assert "为空" in decision.reason


def test_strong_relevance_passes() -> None:
    """高分多领域 → 通过."""
    chunks = [
        make_chunk(0.20, 0.025, "中华人民共和国劳动法"),
        make_chunk(0.18, 0.022, "中华人民共和国劳动合同法"),
        make_chunk(0.17, 0.020, "中华人民共和国刑法"),
    ]
    decision = decide_abstention(chunks)
    assert not decision.should_abstain
    assert "通过" in decision.reason


def test_absolute_threshold_abstains() -> None:
    """Top1 < 0.05 必须拒答(信号 1)."""
    chunks = [make_chunk(0.03)]
    decision = decide_abstention(chunks)
    assert decision.should_abstain
    assert "绝对阈值" in decision.reason


def test_weak_rerank_and_weak_recall_abstains() -> None:
    """rerank 弱 + 召回弱 → 联合拒答(信号 2)."""
    chunks = [make_chunk(rerank_score=0.10, rrf_score=0.015)]
    decision = decide_abstention(chunks)
    assert decision.should_abstain
    assert "偏弱" in decision.reason


def test_weak_rerank_but_strong_recall_passes() -> None:
    """rerank 弱但召回强 → 通过(避免误拒)."""
    chunks = [
        make_chunk(rerank_score=0.10, rrf_score=0.030, law_title="中华人民共和国民法典"),
        make_chunk(rerank_score=0.09, rrf_score=0.025, law_title="中华人民共和国合同法"),
    ]
    decision = decide_abstention(chunks)
    assert not decision.should_abstain


def test_single_peripheral_domain_abstains() -> None:
    """Top5 全是气象法 → 联合拒答(信号 3)."""
    chunks = [
        make_chunk(0.14, 0.013, "中华人民共和国气象法"),
        make_chunk(0.14, 0.012, "中华人民共和国气象法"),
        make_chunk(0.14, 0.012, "中华人民共和国气象法"),
    ]
    decision = decide_abstention(chunks)
    assert decision.should_abstain
    assert "边缘领域" in decision.reason


def test_multi_domain_passes_even_if_low() -> None:
    """多领域命中即使分数低也通过(可能是合法的边缘问题)."""
    chunks = [
        make_chunk(0.13, 0.025, "中华人民共和国民法典"),
        make_chunk(0.13, 0.025, "中华人民共和国合同法"),
        make_chunk(0.12, 0.022, "中华人民共和国劳动法"),
    ]
    decision = decide_abstention(chunks)
    # rerank 0.13 < 0.15(weak),但 RRF 0.025 > 0.020(strong)
    # 不是单一边缘领域 → 通过
    assert not decision.should_abstain


def test_user_message_not_empty_when_abstaining() -> None:
    """拒答时 user_message 必须非空."""
    decision = decide_abstention([make_chunk(0.01)])
    assert decision.should_abstain
    assert len(decision.user_message) > 0
    assert "律师" in decision.user_message  # 法律场景的关键引导


def test_user_message_empty_when_passing() -> None:
    """通过时 user_message 应为空."""
    chunks = [
        make_chunk(0.20, 0.025, "中华人民共和国劳动法"),
        make_chunk(0.18, 0.022, "中华人民共和国民法典"),
    ]
    decision = decide_abstention(chunks)
    assert not decision.should_abstain
    assert decision.user_message == ""
