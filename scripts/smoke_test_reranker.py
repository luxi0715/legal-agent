"""DashScope gte-rerank-v2 冒烟测试.

验证目标:
1. API key 配置正确
2. gte-rerank 服务已开通
3. 请求/响应格式符合预期
4. 测出真实延迟基准
"""

import os
import time

import dashscope
from dotenv import load_dotenv

# 从 .env 读 API key(复用你 M3 的配置)
load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

if not dashscope.api_key:
    raise RuntimeError("❌ DASHSCOPE_API_KEY 没读到,检查 .env 文件")


def smoke_test() -> None:
    """一个非常简单的中文场景:看 Reranker 能不能区分相关度."""
    query = "老板拖欠工资怎么办"

    # 4 段候选文档,人为构造从最相关到最不相关
    documents = [
        # 应该排第 1:直接回答工资拖欠
        "用人单位拖欠劳动者工资的,劳动者可以向劳动行政部门投诉,也可以申请仲裁。",
        # 应该排第 2:相关法条但偏边缘
        "《劳动合同法》规定,用人单位应当按时足额支付劳动报酬。",
        # 应该排第 3:有"工资"关键词但内容偏题
        "工资集体协商制度是企业与职工就工资分配问题进行平等协商的制度。",
        # 应该排第 4:完全无关
        "民法典第六百二十一条规定了买卖合同标的物的检验期限。",
    ]

    print("=" * 60)
    print(f"Query: {query}")
    print(f"候选文档数: {len(documents)}")
    print("=" * 60)

    start = time.perf_counter()

    # 调 API
    response = dashscope.TextReRank.call(
        model="gte-rerank-v2",
        query=query,
        documents=documents,
        top_n=4,  # 返回前 4 条(全要)
        return_documents=True,  # 让结果带上原文,方便看
    )

    elapsed_ms = (time.perf_counter() - start) * 1000

    # 检查响应
    if response.status_code != 200:
        print(f"❌ API 调用失败: {response.code} - {response.message}")
        return

    print(f"\n✅ 调用成功,耗时 {elapsed_ms:.0f} ms")
    print("\n响应原始结构 (output 字段):")
    print(f"  type: {type(response.output)}")
    print(f"  keys: {list(response.output.keys()) if hasattr(response.output, 'keys') else 'N/A'}")

    # DashScope 的返回结构: response.output.results 是 list
    results = response.output["results"]

    print(f"\n排序结果 (共 {len(results)} 条):")
    print("-" * 60)
    for i, item in enumerate(results, 1):
        # 每条结果包含: index(原索引)、relevance_score、document.text
        idx = item["index"]
        score = item["relevance_score"]
        text = item["document"]["text"][:40]
        print(f"  [{i}] score={score:.4f}  原索引={idx}")
        print(f"      内容: {text}...")
    print("-" * 60)

    # 用量统计(很便宜,但记一下)
    if hasattr(response, "usage") and response.usage:
        print(f"\n💰 用量: {response.usage}")


if __name__ == "__main__":
    smoke_test()
