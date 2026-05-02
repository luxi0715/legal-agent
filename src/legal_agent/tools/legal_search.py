"""Tool: legal_search — wraps M5 two-stage retrieval.

输入:自然语言 query
输出:格式化的检索结果字符串(LLM 友好)

如果 abstention 触发,返回拒答提示供 LLM 处理(不抛异常).
"""

from legal_agent.rag.abstention import decide_abstention
from legal_agent.rag.context_reorder import reorder_for_lost_in_the_middle
from legal_agent.rag.two_stage_retriever import two_stage_retrieve


async def legal_search(query: str) -> str:
    """语义检索法律条文,返回格式化文本.

    Args:
        query: 用户的法律问题.

    Returns:
        多条法条拼接的字符串,带 [1] [2] 引用编号.
        如果拒答,返回一段说明文字让 LLM 据此回应.
    """
    reranked = await two_stage_retrieve(
        query=query,
        recall_top_k=50,
        final_top_k=5,
    )
    decision = decide_abstention(reranked)

    if decision.should_abstain:
        return (
            f"[未找到充分相关的法条] {decision.reason}\n"
            f"建议告知用户:这个问题可能不在法律咨询范畴,或建议咨询专业律师."
        )

    reordered = reorder_for_lost_in_the_middle(reranked)
    lines = []
    for i, chunk in enumerate(reordered, 1):
        meta = chunk["metadata"]
        law = meta.get("law_title", "")
        article = meta.get("article_no", "")
        content = chunk["content"]
        lines.append(f"[{i}] {law} {article}\n{content}")
    return "\n\n".join(lines)


__all__ = ["legal_search"]
