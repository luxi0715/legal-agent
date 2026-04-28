"""LLM client wrapper around the OpenAI-compatible SDK."""

from collections.abc import AsyncIterator

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
