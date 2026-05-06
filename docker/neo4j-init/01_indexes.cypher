// M11 — 法条引用关系图初始化(M11.2 写入数据前跑)

// 唯一约束:每个 Article 节点 article_id 必须唯一
// 副作用:同时建索引,加速 MATCH (a:Article {article_id: ...}) 查询
CREATE CONSTRAINT article_id_unique IF NOT EXISTS
FOR (a:Article)
REQUIRE a.article_id IS UNIQUE;

// 普通索引:按 law_name 查询(M14+ 完整版用)
CREATE INDEX article_law_name IF NOT EXISTS
FOR (a:Article)
ON (a.law_name);
