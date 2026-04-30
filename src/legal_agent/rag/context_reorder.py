"""Lost in the Middle 修复:重新排列 Context 顺序.

LLM 在长 context 中对中间位置的注意力较低
(Liu et al., 2023, "Lost in the Middle: How Language Models Use Long Contexts").

本模块把最相关的放首/尾,中等相关的塞中间,缓解该效应.
"""

from legal_agent.rag.reranker import RerankedChunk


def reorder_for_lost_in_the_middle(
    ranked_chunks: list[RerankedChunk],
) -> list[RerankedChunk]:
    """重新排列已精排的结果,缓解 Lost in the Middle.

    策略:奇数位(0,2,4...)放头部,偶数位(1,3,5...)放尾部.
    例如:
        输入 5 条 [T1, T2, T3, T4, T5](已按 rerank_score 降序)
        输出 5 条 [T1, T3, T5, T4, T2]
                  ↑                ↑
                  最强放开头        次强放结尾

    输入 7 条 [T1...T7] → 输出 [T1, T3, T5, T7, T6, T4, T2]

    Args:
        ranked_chunks: 已经按 rerank_score 降序排好的列表.

    Returns:
        重排后的列表,长度不变,内容不变,只动顺序.
        如果输入 ≤2 条,原样返回(无意义).
    """
    n = len(ranked_chunks)
    if n <= 2:
        return list(ranked_chunks)

    head: list[RerankedChunk] = []
    tail: list[RerankedChunk] = []
    for i, chunk in enumerate(ranked_chunks):
        if i % 2 == 0:
            head.append(chunk)
        else:
            tail.append(chunk)

    # tail 反转:让"次强"贴近结尾,"较弱的偶位"被推到中间
    return head + tail[::-1]


__all__ = ["reorder_for_lost_in_the_middle"]
