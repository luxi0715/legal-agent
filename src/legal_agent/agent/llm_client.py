"""LLM client wrapper around the OpenAI-compatible SDK."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from legal_agent.core.config import get_settings
from legal_agent.rag.abstention import decide_abstention
from legal_agent.rag.context_reorder import reorder_for_lost_in_the_middle
from legal_agent.rag.reranker import RerankedChunk
from legal_agent.rag.two_stage_retriever import two_stage_retrieve

DEFAULT_SYSTEM_PROMPT = (
    "你是一位专业的法律顾问助手。"
    "请用清晰、准确、友好的方式回答用户的法律问题。"
    "如果问题超出法律范畴,请礼貌地说明你的专长是法律咨询。"
)


def get_llm_client() -> AsyncOpenAI:
    """Create an AsyncOpenAI client pointing at DeepSeek."""
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )


async def generate_reply(
    user_message: str,
    system_prompt: str | None = None,
) -> str:
    """Send a message to DeepSeek and return the full reply."""
    client = get_llm_client()
    settings = get_settings()

    response = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
    )

    return response.choices[0].message.content or ""


async def stream_reply(
    user_message: str,
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    """Send a message to DeepSeek and yield reply chunks as they arrive."""
    client = get_llm_client()
    settings = get_settings()

    stream = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def generate_reply_with_rag(
    user_message: str,
    retrieved_chunks: list[dict[str, Any]],
    system_prompt: str | None = None,
) -> str:
    """Generate a reply using retrieved context (RAG mode, M3-M4 style).

    保留以兼容旧调用. M5 推荐使用 generate_reply_with_two_stage.
    """
    client = get_llm_client()
    settings = get_settings()

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        law = meta.get("law_title", "")
        article = meta.get("article_no", "")
        context_parts.append(f"[{i}] {law} {article}\n{chunk['content']}")
    context = "\n\n".join(context_parts)

    rag_system = system_prompt or DEFAULT_SYSTEM_PROMPT
    rag_system += (
        "\n\n以下是相关的法律条文,请基于这些条文回答用户问题。"
        "如果条文中没有明确答案,请说明并谨慎建议。"
        "回答时请用 [1]、[2] 等标注引用来源。"
    )

    response = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": rag_system},
            {
                "role": "user",
                "content": f"参考资料:\n{context}\n\n问题:{user_message}",
            },
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content or ""


# ──────────────────────────────────────────────────────
# M5: Two-Stage RAG with Abstention
# ──────────────────────────────────────────────────────


@dataclass
class TwoStageRagReply:
    """两阶段 RAG 的回复结果(信息丰富,便于监控/评测)."""

    reply: str
    abstained: bool
    abstain_reason: str = ""
    top_rerank: float = 0.0
    top_recall: float = 0.0
    used_chunks: list[RerankedChunk] = field(default_factory=list)


async def generate_reply_with_two_stage(
    user_message: str,
    system_prompt: str | None = None,
    final_top_k: int = 5,
    recall_top_k: int = 50,
) -> TwoStageRagReply:
    """Generate reply via Two-Stage RAG (M5).

    流程:
      1. two_stage_retrieve  Hybrid 召回 + Reranker 精排
      2. decide_abstention   置信度拒答检查 → 命中则直接返回
      3. reorder_for_lost_in_the_middle  调整 context 顺序
      4. 调 LLM 生成回复
    """
    reranked = await two_stage_retrieve(
        query=user_message,
        recall_top_k=recall_top_k,
        final_top_k=final_top_k,
    )

    decision = decide_abstention(reranked)
    if decision.should_abstain:
        return TwoStageRagReply(
            reply=decision.user_message,
            abstained=True,
            abstain_reason=decision.reason,
            top_rerank=decision.top_rerank,
            top_recall=decision.top_recall,
            used_chunks=[],
        )

    reordered = reorder_for_lost_in_the_middle(reranked)

    context_parts = []
    for i, chunk in enumerate(reordered, 1):
        meta = chunk["metadata"]
        law = meta.get("law_title", "")
        article = meta.get("article_no", "")
        context_parts.append(f"[{i}] {law} {article}\n{chunk['content']}")
    context = "\n\n".join(context_parts)

    rag_system = system_prompt or DEFAULT_SYSTEM_PROMPT
    rag_system += (
        "\n\n以下是相关的法律条文,请基于这些条文回答用户问题。"
        "如果条文中没有明确答案,请说明并谨慎建议。"
        "回答时请用 [1]、[2] 等标注引用来源。"
    )

    client = get_llm_client()
    settings = get_settings()
    response = await client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[
            {"role": "system", "content": rag_system},
            {
                "role": "user",
                "content": f"参考资料:\n{context}\n\n问题:{user_message}",
            },
        ],
        temperature=0.3,
    )
    reply_text = response.choices[0].message.content or ""

    return TwoStageRagReply(
        reply=reply_text,
        abstained=False,
        abstain_reason=decision.reason,
        top_rerank=decision.top_rerank,
        top_recall=decision.top_recall,
        used_chunks=reordered,
    )
