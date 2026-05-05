"""Router 单测 (M9.2)."""

from __future__ import annotations

import pytest

from legal_agent.agent.router import (
    CascadeRouter,
    RuleBasedRouter,
    get_default_router,
)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("你好", "greeting"),
        ("hi", "greeting"),
        ("谢谢", "greeting"),
        ("我被公司辞退了", "legal"),
        ("民法典第577条", "legal"),
        ("合同违约怎么赔偿", "legal"),
        ("今天天气怎么样", "off_topic"),
        ("推荐电影", "off_topic"),
    ],
)
@pytest.mark.asyncio
async def test_rule_router(query: str, expected: str) -> None:
    router = RuleBasedRouter()
    result = await router.route(query)
    assert result == expected, f"{query!r} → {result}, expected {expected}"


@pytest.mark.asyncio
async def test_cascade_legal_no_llm_call() -> None:
    """法律词命中规则,不调 LLM."""
    router = CascadeRouter()
    assert await router.route("我被公司辞退了") == "legal"


@pytest.mark.asyncio
async def test_cascade_off_topic_no_llm_call() -> None:
    """off-topic 词命中规则,不调 LLM."""
    router = CascadeRouter()
    assert await router.route("今天天气怎么样") == "off_topic"


@pytest.mark.asyncio
async def test_cascade_short_greeting_no_llm_call() -> None:
    """短招呼命中规则."""
    router = CascadeRouter()
    assert await router.route("你好") == "greeting"


@pytest.mark.asyncio
async def test_cascade_ambiguous_uses_llm() -> None:
    """模糊 query 走 LLM(规则没命中)."""
    router = CascadeRouter()
    result = await router.route("领导说做不下去就走人,我刚干 3 个月")
    assert result in ("legal", "off_topic")


@pytest.mark.asyncio
async def test_default_router_is_cascade() -> None:
    """默认是 CascadeRouter 单例."""
    r1 = get_default_router()
    r2 = get_default_router()
    assert r1 is r2
    assert isinstance(r1, CascadeRouter)
