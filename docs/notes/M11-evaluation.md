# M11 · 知识图谱评测报告(MVP)

> 从 M5 RAG 单条检索到 M11 知识图谱关系查询,
> 让 Agent 能回答"哪些条文引用了第X条""第X条的关联条款是什么"等结构化关系查询.

---

## 1. 实验设置

| 维度 | 配置 |
|---|---|
| 图数据库 | Neo4j 5.14 Community (Docker 容器) |
| Python 驱动 | neo4j async driver |
| 数据范围 | 5 部核心法律,1559 个 Article 节点 |
| 关系类型(MVP) | REFERENCES(引用)— 单一边 |
| 引用抽取 | 正则模式("依照/参照/依据/适用 + 第X条") |
| 跨法律引用 | MVP 不抓,只抓"本法"内部引用 |

---

## 2. 数据规模

### 2.1 节点分布

| 法律 | 条文数 |
|---|---|
| 中华人民共和国民法典 | 1259 |
| 中华人民共和国劳动法 | 107 |
| 中华人民共和国劳动合同法 | 98 |
| 中华人民共和国消费者权益保护法 | 65 |
| 中华人民共和国反不正当竞争法 | 33 |
| **合计** | **1562 → 去重 1559** |

### 2.2 引用关系分布

- **REFERENCES 边总数**:108 条
- **有引用的法条**:77 条(占 4.9%)
- **平均每条引用数**:1.40
- **引用最多的法条**:5 条(出度 Top1)

### 2.3 入度 Top5(被引用最多的法条)

| 法条 | 被引用次数 | 角色 |
|---|---|---|
| 民法典:510 | 26 | 合同补充协议规则(明星条款) |
| 劳动合同法:40 | 5 | 用人单位提前30日通知解除 |
| 民法典:1098 | 3 | 收养条件 |
| 劳动合同法:39 | 3 | 单位解除合同情形 |
| 劳动合同法:26 | 3 | 劳动合同效力规则 |

### 2.4 工程发现

- 民法典 510 是**网络中心节点**,被合同编各分编(602/619/637/782/831/858/875/902/976...)广泛引用,作为"约定不明时的兜底规则"
- 民法典 577 是**孤立节点**,本身是定义型条款,无引用关系
- 这印证一个判断:**孤立节点 ≠ 不重要**,反而可能是基础定义型条款

---

## 3. M11 KG 工具能力

### 3.1 工具签名

```python
async def get_related_articles(
    law_name: str,         # "民法典"
    article_no: str,       # "第五百七十七条" 或 "577"
    direction: str = "both",  # outgoing / incoming / both
    limit: int = 10,
) -> str
```

### 3.2 输入容错

- 法律名:接受"民法典"(简称)或"中华人民共和国民法典"(全称)
- 条款号:接受中文数字"第五百一十条"或阿拉伯"510",自动标准化
- 方向:默认 both,Agent 可按需指定 incoming/outgoing

### 3.3 错误处理

| 输入 | 输出 |
|---|---|
| 不存在的法律 | "[参数错误] 无法解析..." |
| 法律内不存在的条款号 | "[未找到] 知识图谱中没有..." |
| 孤立节点 | "[提示] 该法条在知识图谱中是 孤立节点..." |

---

## 4. Agent 工具选择评测

测试 4 类 query,观察 ReAct Agent 自主选择:

| Query 类型 | LLM 选择 | 期望 | 结果 |
|---|---|---|---|
| "民法典第510条都有谁引用?" | get_related_articles(direction=incoming) | KG | ✅ |
| "民法典第510条说啥?有哪些条文跟它有关?" | get_law_article + get_related_articles **并行** | 两个工具都用 | ✅ |
| "民法典第577条规定了什么?" | get_law_article | 原文 | ✅ |
| "老板拖欠工资怎么办?" | legal_search(自动 query expansion) | RAG | ✅ |

**关键证据 — Case 2 并行调用**:

LLM 在 ReAct 同一轮 thinker 节点输出 2 个 tool_calls:
[1] get_law_article(law_title="民法典", article_no="第五百一十条")
[1] get_related_articles(law_name="民法典", article_no="第五百一十条", direction="both")

这证明:
- LLM 能识别"原文 + 关联"是 2 个不同需求
- DeepSeek 支持 OpenAI 协议的 parallel tool calls
- M7 LangGraph actor_node 正确处理 list 遍历执行

---

## 5. 性能对比

| 工具 | 平均延迟 | 用途 |
|---|---|---|
| get_law_article (M6 SQL) | 50-200ms | 原文精确查 |
| get_related_articles (M11 KG) | **<50ms** | 关系查询 |
| legal_search (M5 RAG) | 1-3s | 语义检索 |
| LLM 一轮 thinker | 2-5s | 决策 |

**KG 是项目最快的工具** — Neo4j 单跳查询毫秒级,
比 RAG 快 50-100x.

---

## 6. M11 vs M5 vs M6 工具栈对比
M5 RAG     语义相似条文          1-3s
↓
M6 SQL     精确查"民法典 577"   < 200ms
↓
M11 KG     "577 引用了哪些"     < 50ms
↓
三栈协同 — Agent 自主路由

M5 解决"找相似",M6 解决"找精确",M11 解决"找关系".
三者互补,通过 ReAct Tool Use 自主选择.

---

## 7. MVP 局限 + 升级路线

### 当前 MVP 局限

1. **只有 1 种边类型**:REFERENCES
   - 缺:AMENDED_FROM / INTERPRETED_BY / SUPERIOR_LAW / CONTAINS_CONCEPT
2. **只有 5 部法律**:1559 节点
   - 缺:刑法、刑事诉讼法、民事诉讼法、公司法等
3. **跨法律引用未抓**
   - "依照刑法第X条"这类引用全部丢弃
4. **正则覆盖不全**
   - "前条"、"本条第二款"、"下列各项"等指代式引用未抓

### 升级路径(M14+)

- 扩量到全量 25k+ 条
- 加 4-5 类边
- LLM 抽取替代正则(覆盖隐式关系)
- 多跳查询能力("577 间接关联到哪些条")
- 加 Concept 节点("善意取得"等法律概念反查条文)

**关键设计:MVP 数据模型 = 完整版子集,升级时无需迁移.**

---

## 8. 工程踩坑记录

### 8.1 Cypher vs SQL 字符串函数
SQL:    substr(s, start, length)    start 1-indexed
Cypher: substring(s, start, length) start 0-indexed

M11.2 抽取脚本用 SQL 顺手,M11.3 KG 工具沿用 substr → CypherSyntaxError.
切换语言时字符串函数需注意.

### 8.2 中文数字 → 阿拉伯数字

法条 article_id 用阿拉伯数字便于排序,但 LLM 输入可能是中文.
工具内部做 _normalize_article_id 统一标准化.

### 8.3 法律名简称 vs 全称

PG 存"中华人民共和国民法典",但 article_id 用"民法典"避免节点 ID 过长.
工具入口做 LAW_NAME_MAP 双向兼容.

---

## 9. 简历金句

> "实现基于 Neo4j 的法条引用知识图谱(MVP),建立 1559 个 Article 节点和 108 条
> REFERENCES 边,覆盖民法典、劳动法、劳动合同法等 5 部核心法律.
> 通过正则模式抽取'依照/参照/依据 + 第X条'引用关系,中文数字自动标准化."

> "新增 ReAct Agent 第 3 个工具 get_related_articles(M11 KG)与 M5 RAG、
> M6 SQL 工具栈形成互补.LLM 自主决策三种检索路径,实测在'查原文 + 找关联'
> 复合 query 上,LLM 同轮并行调用 2 个工具,体现 LangGraph parallel tool calls
> 工程实践."

> "Neo4j 单跳查询 < 50ms,比 RAG 快 100x,从根本上解决 M5 RAG 在'结构化关系
> 查询'场景下的能力短板.如'民法典第510条都有哪些条文引用了它'这类
> question,M5 完全无法处理,M11 一查图就知道."
