"""单元测试 — Entity Extractor (M8.5)."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from legal_agent.memory.entity_extractor import (
    ALLOWED_KEYS,
    extract_and_persist,
    extract_entities,
)
from legal_agent.memory.hard_memory import (
    delete_all_user_facts,
    get_user_facts,
)


@pytest_asyncio.fixture
async def fresh_user_id() -> AsyncIterator[uuid.UUID]:
    """每个测试用全新 user_id."""
    uid = uuid.uuid4()
    try:
        yield uid
    finally:
        await delete_all_user_facts(uid)


@pytest.mark.asyncio
async def test_extract_empty_messages() -> None:
    """空消息 → 空结果(不调 LLM)."""
    facts = await extract_entities([])
    assert facts == []


@pytest.mark.asyncio
async def test_extract_clear_facts() -> None:
    """⭐ 明确陈述的 fact 应被抽取."""
    msgs = [
        {"role": "user", "content": "我是北京的程序员,30 岁"},
    ]
    facts = await extract_entities(msgs)

    keys = {f["key"] for f in facts}
    # 至少抽到 location 或 occupation
    assert keys & {"location", "occupation"}


@pytest.mark.asyncio
async def test_all_keys_in_whitelist() -> None:
    """⭐ LLM 抽出的 key 必须全在白名单."""
    msgs = [
        {"role": "user", "content": "我在上海做老师,已婚有娃,关心孩子抚养权"},
    ]
    facts = await extract_entities(msgs)

    for f in facts:
        assert f["key"] in ALLOWED_KEYS, f"非法 key: {f['key']}(应在 {ALLOWED_KEYS})"


@pytest.mark.asyncio
async def test_no_facts_from_others() -> None:
    """\"我朋友\" 提到的他人信息 不应 被当作用户的 fact."""
    msgs = [
        {"role": "user", "content": "我朋友小明在深圳做销售"},
    ]
    facts = await extract_entities(msgs)
    # 不强制 == [],但断言里头不该有 location=深圳
    locations = [f["value"] for f in facts if f["key"] == "location"]
    assert "深圳" not in locations


@pytest.mark.asyncio
async def test_extract_and_persist_writes_to_db(
    fresh_user_id: uuid.UUID,
) -> None:
    """⭐ 抽取 + 写入完整链路."""
    msgs = [
        {"role": "user", "content": "我在北京当程序员,正在咨询劳动纠纷"},
    ]
    persisted = await extract_and_persist(fresh_user_id, msgs)

    # 至少抽到 1 条
    assert len(persisted) >= 1

    # 数据库里能读出来
    db_facts = await get_user_facts(fresh_user_id)
    assert len(db_facts) >= 1


@pytest.mark.asyncio
async def test_extract_and_persist_threshold(
    fresh_user_id: uuid.UUID,
) -> None:
    """confidence 阈值生效:阈值 > 1.0 时一切都过滤."""
    msgs = [
        {"role": "user", "content": "我是北京的程序员"},
    ]
    persisted = await extract_and_persist(
        fresh_user_id,
        msgs,
        confidence_threshold=1.01,  # 不可能达到
    )
    assert persisted == []
    assert await get_user_facts(fresh_user_id) == {}


@pytest.mark.asyncio
async def test_extract_and_persist_empty_messages(
    fresh_user_id: uuid.UUID,
) -> None:
    """空消息 → 不写库."""
    persisted = await extract_and_persist(fresh_user_id, [])
    assert persisted == []
    assert await get_user_facts(fresh_user_id) == {}
