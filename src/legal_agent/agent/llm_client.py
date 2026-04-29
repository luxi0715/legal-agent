"""LLM client wrapper around the OpenAI-compatible SDK."""

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from legal_agent.core.config import get_settings

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
    """Generate a reply using retrieved context (RAG mode)."""
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
