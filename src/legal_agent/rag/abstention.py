"""Abstention Strategy: 多信号组合的置信度拒答.

设计背景:
  DashScope gte-rerank-v2 在中文法律语料上分数压缩严重
  (相关 query Top1 通常 0.15~0.25,无关也能到 0.14).
  单一阈值法不可行,本模块用三信号组合判断.

三信号:
  1. 绝对阈值     Top1 rerank_score < 极低值 → 直接拒答
  2. 召回辅助     Hybrid RRF Top1 也很低     → 跨阶段确认
  3. 领域单一性   Top5 法律名收敛到 1~2 部     → 多半无关 query 沾边
"""

from dataclasses import dataclass

from legal_agent.rag.reranker import RerankedChunk

# ──────────────────────────────────────────
# 经验阈值 — M5.7 评测后可调
# ──────────────────────────────────────────
ABSOLUTE_REJECT_THRESHOLD = 0.05  # Top1 低于此值直接拒答
WEAK_RERANK_THRESHOLD = 0.15  # Top1 低于此值进入"待定"
WEAK_RECALL_THRESHOLD = 0.020  # Hybrid RRF Top1 低于此值算召回弱
SINGLE_DOMAIN_LAW_COUNT = 2  # Top5 涉及 ≤2 部法律算"领域单一"


# 法律领域分类(用于"无关-沾边"识别).
# Top5 全部来自这些"边缘领域"且数量少 → 很可能是无关 query 沾边.
PERIPHERAL_LAW_PREFIXES = (
    "中华人民共和国气象法",
    "中华人民共和国邮政法",
    "中华人民共和国测绘法",
    "中华人民共和国地震法",
    "中华人民共和国统计法",
)


@dataclass
class AbstentionDecision:
    """拒答决策结果."""

    should_abstain: bool
    top_rerank: float  # Top1 rerank_score(用于解释)
    top_recall: float  # Top1 RRF score(用于解释)
    reason: str  # 决策理由(给开发者看)
    user_message: str  # 拒答话术(给用户看)


def _is_single_peripheral_domain(reranked: list[RerankedChunk]) -> bool:
    """检测 Top5 是否集中在 1~2 部边缘法律.

    例如 "今天天气" 命中 5 条全是《气象法》→ 几乎肯定是无关 query.
    """
    if not reranked:
        return False

    law_titles = []
    for chunk in reranked:
        title = chunk["metadata"].get("law_title", "")
        if title:
            law_titles.append(title)

    if not law_titles:
        return False

    unique_laws = set(law_titles)
    if len(unique_laws) > SINGLE_DOMAIN_LAW_COUNT:
        return False

    # 所有命中的法律都来自边缘领域?
    return all(
        any(str(law).startswith(prefix) for prefix in PERIPHERAL_LAW_PREFIXES)
        for law in unique_laws
    )


def _build_user_message(reason_short: str) -> str:
    """生成给用户的拒答话术(对法律咨询场景偏保守)."""
    return (
        f"抱歉,我没有把握准确回答这个问题({reason_short})。"
        f"建议您咨询专业律师获取准确意见,或换种问法再试一次。"
    )


def decide_abstention(
    reranked: list[RerankedChunk],
) -> AbstentionDecision:
    """根据多信号组合决定是否拒答.

    Args:
        reranked: 精排后的结果(已按 rerank_score 降序).

    Returns:
        AbstentionDecision 决策对象.
    """
    # 边界:精排为空(召回阶段就空了)
    if not reranked:
        return AbstentionDecision(
            should_abstain=True,
            top_rerank=0.0,
            top_recall=0.0,
            reason="召回结果为空",
            user_message=_build_user_message("法律资料库未找到相关条款"),
        )

    top_rerank = reranked[0]["rerank_score"]
    top_recall = reranked[0]["original_score"] or 0.0

    # ────────────────────────────
    # 信号 1:绝对阈值(高置信拒答)
    # ────────────────────────────
    if top_rerank < ABSOLUTE_REJECT_THRESHOLD:
        return AbstentionDecision(
            should_abstain=True,
            top_rerank=top_rerank,
            top_recall=top_recall,
            reason=f"Top1 相关度 {top_rerank:.4f} < {ABSOLUTE_REJECT_THRESHOLD}(绝对阈值)",
            user_message=_build_user_message(f"最相关条款的相关度仅 {top_rerank:.0%}"),
        )

    # ────────────────────────────
    # 信号 2 + 信号 3:联合判断(中置信拒答)
    # ────────────────────────────
    weak_rerank = top_rerank < WEAK_RERANK_THRESHOLD
    weak_recall = top_recall < WEAK_RECALL_THRESHOLD
    single_domain = _is_single_peripheral_domain(reranked)

    # 领域单一收敛到边缘领域 → 优先判断(更 specific 的信号)
    if single_domain:
        law_set = {c["metadata"].get("law_title", "") for c in reranked}
        return AbstentionDecision(
            should_abstain=True,
            top_rerank=top_rerank,
            top_recall=top_recall,
            reason=f"Top5 全部来自边缘领域: {law_set}",
            user_message=_build_user_message("您的问题可能不在法律咨询范畴"),
        )

    # rerank 弱 + 召回也弱 → 联合拒答(更通用的兜底信号)
    if weak_rerank and weak_recall:
        return AbstentionDecision(
            should_abstain=True,
            top_rerank=top_rerank,
            top_recall=top_recall,
            reason=(f"Top1 相关度 {top_rerank:.4f} 偏弱 + 召回 RRF {top_recall:.4f} 也偏弱"),
            user_message=_build_user_message("库内未找到足够相关的内容"),
        )

    # ────────────────────────────
    # 通过:送给 LLM
    # ────────────────────────────
    return AbstentionDecision(
        should_abstain=False,
        top_rerank=top_rerank,
        top_recall=top_recall,
        reason=f"Top1 相关度 {top_rerank:.4f} 通过多信号校验",
        user_message="",
    )


__all__ = ["AbstentionDecision", "decide_abstention"]
