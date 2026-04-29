"""Reciprocal Rank Fusion for combining multiple retrievers."""

from typing import Any


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    k: int = 60,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked result lists using RRF.

    Reference: Cormack et al., 2009, "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods"

    Args:
        rankings: list of result lists, each from a different retriever.
                  Each result must have a 'content' field as identity.
        k: smoothing constant. 60 is the paper's default.
            Larger k = less aggressive ranking.
        top_k: how many fused results to return.

    Returns:
        Fused list sorted by RRF score descending.
        Each item contains:
        - content / metadata / original 'score' from one retriever
        - 'rrf_score': fused score
        - 'sources': list of retriever names that found it
    """
    # content -> {item, rrf_score, sources}
    fused: dict[str, dict[str, Any]] = {}

    for ranking_idx, ranking in enumerate(rankings):
        source_name = f"retriever_{ranking_idx}"
        for rank, item in enumerate(ranking):
            content = item["content"]
            rrf_increment = 1.0 / (k + rank + 1)

            if content not in fused:
                fused[content] = {
                    **item,
                    "rrf_score": 0.0,
                    "sources": [],
                }
            fused[content]["rrf_score"] += rrf_increment
            fused[content]["sources"].append(source_name)

    # 按 rrf_score 降序排序
    sorted_results = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
    return sorted_results[:top_k]
