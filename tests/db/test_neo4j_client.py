"""Neo4j connectivity tests (M11.1)."""

from __future__ import annotations

import pytest

from legal_agent.db.neo4j_client import get_neo4j


@pytest.mark.asyncio
async def test_neo4j_connection() -> None:
    """⭐ 基础连通性 — 跑一条 RETURN 1."""
    driver = get_neo4j()
    async with driver.session() as session:
        result = await session.run("RETURN 1 AS value")
        record = await result.single()
        assert record is not None
        assert record["value"] == 1


@pytest.mark.asyncio
async def test_create_and_delete_node() -> None:
    """⭐ 写入 + 读取 + 删除节点链路."""
    driver = get_neo4j()
    async with driver.session() as session:
        # 创建临时测试节点
        await session.run(
            "CREATE (n:TestNode {name: $name, value: $value})",
            name="m11_test",
            value=42,
        )

        # 读出来
        result = await session.run(
            "MATCH (n:TestNode {name: $name}) RETURN n.value AS value",
            name="m11_test",
        )
        record = await result.single()
        assert record is not None
        assert record["value"] == 42

        # 清理
        await session.run(
            "MATCH (n:TestNode {name: $name}) DELETE n",
            name="m11_test",
        )

        # 确认清理成功
        result = await session.run(
            "MATCH (n:TestNode {name: $name}) RETURN count(n) AS cnt",
            name="m11_test",
        )
        record = await result.single()
        assert record is not None
        assert record["cnt"] == 0


@pytest.mark.asyncio
async def test_neo4j_constraint_exists() -> None:
    """⭐ M11.1 初始化的 article_id 约束应存在.

    注意:这个测试假设你已经手动跑过 docker/neo4j-init/01_indexes.cypher
    (Neo4j Community 不像 PG 那样自动跑 init 文件).
    M11.2 会在数据写入前自动 ensure constraints.
    """
    driver = get_neo4j()
    async with driver.session() as session:
        result = await session.run("SHOW CONSTRAINTS")
        records = await result.data()

    constraint_names = {r["name"] for r in records if "name" in r}
    # 不强制断言,因为 M11.1 没自动跑 cypher
    # M11.2 会保证这个约束存在
    if "article_id_unique" not in constraint_names:
        pytest.skip("article_id_unique 约束未创建 — M11.2 会自动建,M11.1 暂不强制")
