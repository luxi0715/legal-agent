# M7 · LangGraph ReAct Agent 评测报告

> 从 M6 命令式 Agent Loop 到 M7 LangGraph 声明式图编排,
> 一次以 **重构语义等价 + 扩展性铺路** 为目标的架构升级.

---

## 1. 实验设置

| 维度 | 配置 |
|---|---|
| LLM | DeepSeek-Chat (OpenAI 协议 tool_use) |
| 框架 | LangGraph StateGraph |
| 节点 | thinker (LLM 决策) + actor (工具执行) |
| 状态管理 | TypedDict + add_messages reducer |
| 路由 | conditional_edge 根据 tool_calls 决定循环 / 终止 |
| 最大迭代 | 5 轮(防死循环) |
| 工具集 | 沿用 M6:legal_search + get_law_article |
| 评测对象 | M6 `/chat/agent` vs M7 `/chat/react` 横向对比 |

---

## 2. 评测方法学说明

M7 是 **架构性重构**,核心增量为控制流升级,而非召回质量提升.
因此本评测采用 **行为对比 + 维度埋点**,而非 R@K / Precision@K 等数值评测.

评测维度三性:

| 维度 | 含义 |
|---|---|
| 等价性 | M6 已有的能力 M7 是否保留(无退化) |
| 可观测性 | M7 是否暴露完整决策轨迹 |
| 扩展性 | M7 是否为后续模块提供解耦扩展点 |

---

## 3. 评测 Query 集

| # | 类型 | Query |
|---|---|---|
| 1 | 精确条款 | 民法典第五百七十七条规定了什么? |
| 2 | 语义检索 | 老板拖欠工资怎么办? |
| 3 | 跨工具混合 | 民法典577条说什么?这种情况下我该怎么维权? |
| 4 | 跨工具复杂 ⭐ | 我和公司签的合同被违约了,具体看哪条法律?该怎么办? |
| 5 | 多条同时问 | 民法典第五百七十七条和第五百七十八条分别说什么? |
| 6 | 闲聊负例 | 你好,介绍一下你自己 |

---

## 4. 实测结果

### 4.1 工具调用模式对比

| # | M6 耗时 | M7 耗时 | M7 迭代 | M7 工具调用 | 决策模式 |
|---|---|---|---|---|---|
| 1 | 28206ms | 9460ms | 2 | 1 次 | 单工具 |
| 2 | 13114ms | 15088ms | 2 | 1 次 | 单工具 |
| 3 | 13147ms | 16021ms | 2 | **2 次同轮并行** | 任务拆分 + 并行 |
| 4 | 19953ms | 22211ms | **3** | **4 次(链式+并行)** ⭐ | 多步推理 |
| 5 | 9619ms | 11539ms | 2 | 2 次同轮并行 | 多目标并行 |
| 6 | 5188ms | 6496ms | 1 | 0 次 | 不调工具 |

⚠️ **延迟说明**:M6 端点不暴露内部 trace,仅能对比总耗时.单次测试存在
DeepSeek API 抖动,延迟差异不应单点解读(如 Q1 M6 异常慢).

### 4.2 Q4 关键发现 ⭐⭐⭐

Q4 暴露了 M7 vs M6 的核心架构差异:

**M7 决策轨迹**:
iter 1: legal_search("合同违约 法律责任 民法典 违约责任")
→ 探索性语义检索,定位相关法律领域
iter 2: get_law_article("民法典", "第五百七十七条")
get_law_article("民法典", "第五百八十四条")
get_law_article("民法典", "第五百八十五条")
→ 基于 iter 1 发现,同轮并行精确取 3 条原文

**这是教科书级 ReAct 多步推理**:
- ① 先广撒网(legal_search 探索)
- ② 后聚焦细节(基于探索结果决定查哪几条)
- ③ 在聚焦阶段内部又做并行(3 个 get_law_article 同轮)

**M6 命令式循环无法实现这种模式**:
- M6 每轮决策独立,缺乏\"先看 iter 1 结果再决定 iter 2 怎么调\"的能力
- 只能 \"调一次工具 → 直接答\" 或 \"无脑并行\"

### 4.3 三性验证总结

| 维度 | 评测结果 |
|---|---|
| 等价性 | ✅ Q1/Q2/Q5/Q6 行为合理,无退化 |
| 可观测性 | ✅ tool_calls_log 完整暴露 iteration / name / arguments |
| 扩展性 | ✅ Q4 实测多步推理能力,超出 M6 表达力 |

---

## 5. 关键工程发现

### 发现 1:LangChain Message 与 OpenAI 协议的转换层

`add_messages` reducer 自动把 dict 转 LangChain Message 对象
(.type 属性而非 role),与 DeepSeek/OpenAI 原生 SDK 不兼容.

**解决**:实现 `_to_openai_messages` 双向转换,处理三种 tool_calls 格式
- OpenAI 原生 dict
- LangChain 转换后 dict
- OpenAI SDK 对象

→ 这是 LangChain 生态 + 第三方 LLM 的经典踩坑.

### 发现 2:节点切分粒度

最初考虑 thinker / actor / observer 三节点,实测发现 observer 无独立职责
(add_messages reducer 已自动处理 \"观察\"语义).

**结论**:节点切分原则 — 每个节点 = 一个独立的决策点.

### 发现 3:图编译惰性化

`_compiled_graph` 模块级单例 + `get_react_graph()` 惰性初始化:
- 单测时 import 不触发图编译
- 同进程内复用同一编译图
- 单例模式 + 惰性,与数据库连接池设计一致

### 发现 4:Q4 多步推理是 LangGraph 的 \"杀手 case\"

在 M6 的 while-loop 写法中,每轮 LLM 决策独立,无法表达
\"先 search 探索 → 后 article 精确\" 的链式策略.

LangGraph 的图结构允许 thinker 在每次进入时看到 完整 messages 历史,
包括上一轮 actor 的工具结果,自然支持 \"基于上一轮结果决定下一轮\".

→ 这是 ReAct 模式的核心价值,也是 M7 重构的真实回报.

---

## 6. 已知局限 + 后续优化

### 6.1 当前局限

1. **未实现 Self-correction 反思循环**
   - 技术可行,但当前阶段价值不高(\"能力堆栈\"判断):
     - 仅 2 个工具,反思能换的路径有限
     - 无对话上下文,反思缺乏改写依据
   - 留待 M8 记忆 + M11 知识图谱集成后回归

2. **延迟测量噪声**
   - M6 端点不暴露 trace,无法对比内部决策步数
   - 单次测试受 DeepSeek API 抖动影响,延迟数据不应单点解读
   - 真正延迟评测需要 M8 后多次重复 + 统计显著性检验

3. **评测 query 集偏小**
   - 6 个 query 仅做 vibe check + 维度埋点
   - 量化评测留到 M8 后做 LLM-as-Judge

### 6.2 优化路线图

- **M7.x(可选)**:Self-correction 反思边
- **M8** ⭐:三层记忆架构,Buffer + Summary + Hard Memory,与 ReAct 透明集成
- **M9**:模型路由 + Checkpoint 持久化(LangGraph 原生支持)
- **M11**:知识图谱工具加入后,Self-correction 才有真实换路价值

---

## 7. M7 完成里程碑

✓ LangGraph StateGraph 重构 M6 命令式 Agent Loop
✓ thinker / actor 节点解耦,ReAct 模式实现
✓ TypedDict 状态 + add_messages reducer 自动累加
✓ LangChain ↔ OpenAI 双向转换层(三格式兼容)
✓ FastAPI `/chat/react` 端点,完整 trace 暴露
✓ 横向 A/B 评测 6 类场景,三性(等价/可观测/扩展)验证通过
✓ Q4 实测多步推理能力,超出 M6 表达力

---

## 8. 简历金句

> "基于 LangGraph StateGraph 重构 M6 命令式 Agent Loop,采用
> thinker / actor 节点解耦的 ReAct 模式,支持任务拆分与并行工具调用
> (同轮多 tool 调度).重构核心价值在于:① 控制流声明化(图结构 vs
> while 嵌套);② 状态显式管理(StateGraph reducer);③ 为 M7.x
> self-correction、M8 memory 集成提供透明扩展点."

> "在跨工具复杂场景(Q4)实测 M7 自主完成 \"探索 → 聚焦\" 多步推理:
> iter 1 通过 legal_search 探索性定位相关法律领域,iter 2 基于探索结果
> 同轮并行精确取 3 条法条原文.这种 cross-iteration 链式 + 并行混合
> 决策模式,是 M6 命令式循环无法表达的 ReAct 核心价值."

> "处理 LangChain Message 对象与 OpenAI 协议 dict 的格式不兼容问题,
> 实现 `_to_openai_messages` 双向转换层,支持三种 tool_calls 格式
> (OpenAI 原生 / LangChain 转换 / SDK 对象),体现对 LangChain 生态
> 与第三方 LLM 集成的工程经验."

> "评测方法学:M7 重构对象是控制流而非召回质量,采用 行为对比 + 维度
> 埋点(等价性 / 可观测性 / 扩展性)而非 R@K / Precision@K 数值评测.
> 通过 6 类场景对比,验证 M7 在 Q4 多步推理 case 中表现出真实超越
> M6 的能力."
