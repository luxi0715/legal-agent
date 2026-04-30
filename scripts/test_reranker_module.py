"""验证 src/legal_agent/rag/reranker.py 工作正常.

跟 M5.1 的 smoke_test 区别:
- smoke_test 测 DashScope API 本身能不能调通
- 这个脚本测我们封装好的 reranker 模块能不能用
"""

import asyncio

from legal_agent.rag.reranker import rerank


async def main() -> None:
    query = "老板拖欠工资怎么办"

    # 模拟 hybrid_retrieve 返回的候选格式(注意字段名)
    candidates = [
        {
            "content": "用人单位拖欠劳动者工资的,劳动者可以向劳动行政部门投诉,也可以申请仲裁。",
            "rrf_score": 0.0325,
            "metadata": {"law_title": "劳动法", "article_no": "第91条"},
            "sources": ["vector", "bm25"],
        },
        {
            "content": "《劳动合同法》规定,用人单位应当按时足额支付劳动报酬。",
            "rrf_score": 0.0301,
            "metadata": {"law_title": "劳动合同法", "article_no": "第30条"},
            "sources": ["bm25"],
        },
        {
            "content": "工资集体协商制度是企业与职工就工资分配问题进行平等协商的制度。",
            "rrf_score": 0.0188,
            "metadata": {"law_title": "工资集体协商办法"},
            "sources": ["vector"],
        },
        {
            "content": "民法典第六百二十一条规定了买卖合同标的物的检验期限。",
            "rrf_score": 0.0156,
            "metadata": {"law_title": "民法典", "article_no": "第621条"},
            "sources": ["bm25"],
        },
    ]

    print("=" * 60)
    print(f"Query: {query}")
    print(f"输入候选: {len(candidates)} 条(模拟 hybrid 召回结果)")
    print("=" * 60)

    results = await rerank(query=query, candidates=candidates, top_n=4)

    print(f"\n✅ 精排成功,返回 {len(results)} 条")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        law = meta.get("law_title", "?")
        article = meta.get("article_no", "")
        print(
            f"  [{i}] rerank={r['rerank_score']:.4f}  "
            f"召回={r['original_score']:.4f}  "
            f"来源={r['sources']}"
        )
        print(f"      法条: {law} {article}")
        print(f"      内容: {r['content'][:40]}...")
    print("-" * 60)

    # 验证关键属性
    print("\n🔍 自动验证:")
    assert len(results) == 4, "返回数量错误"
    assert results[0]["rerank_score"] >= results[-1]["rerank_score"], "排序错误"
    assert results[0]["metadata"]["law_title"] == "劳动法", "Top1 应该是劳动法"
    print("  ✅ 返回数量正确")
    print("  ✅ 按 rerank_score 降序")
    print("  ✅ Top1 命中预期(劳动法 拖欠工资)")
    print("\n🎉 模块测试全部通过")


if __name__ == "__main__":
    asyncio.run(main())
