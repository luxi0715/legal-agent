# M10 · Persona 防漂移系统评测报告

> 从 M8 单一人设到 M10 多模式 Persona,
> 实现 Agent Persona + User Persona + Guard 四层防护,
> 让法律咨询场景能根据任务类型切换专业风格,同时防止 LLM 漂移和 Prompt 注入.

---

## 1. 设计决策(M10.0)

| # | 维度 | 选择 | 理由 |
|---|---|---|---|
| 1 | Persona 存储 | yaml 文件 | 产品配置,不是用户数据;git 管理便于追溯 |
| 2 | User 应用方式 | system prompt 注入 | LLM 应用最直接的方式,影响生成行为 |
| 3 | Guard 实现 | 关键词黑名单 | 简单可解释,延迟为 0;LLM-as-Judge 留给 M11+ |
| 4 | 应用粒度 | session 级 5 套 | 法律场景任务模式差异大,需要切换 |
| 5 | 与 user_facts 集成 | 中间层翻译 | KV → 自然语言,LLM 更易理解 |

---

## 2. 架构图┌───────────────────────────────────────────────────────────┐
│  用户消息 + persona_mode (default/strict/friendly/...)    │
│         ↓                                                 │
│  ┌─────────────────────┐                                  │
│  │ Persona Loader      │  M10.1 — 5 套 yaml 配置加载      │
│  │  AgentPersona       │  description + style + 禁词      │
│  └─────────┬───────────┘                                  │
│            ↓                                              │
│  ┌─────────────────────┐                                  │
│  │ User Persona Builder│  M10.2 — user_facts → 自然语言   │
│  │  KV → 画像文本      │  含职业 / 家庭关怀维度           │
│  └─────────┬───────────┘                                  │
│            ↓                                              │
│  ┌─────────────────────┐                                  │
│  │ Memory Manager      │  M10.3 — 注入到 system prompt   │
│  │  inject_memory      │  Persona desc + User 画像 + 摘要│
│  │  (向后兼容 M8)      │                                  │
│  └─────────┬───────────┘                                  │
│            ↓                                              │
│  ┌─────────────────────┐                                  │
│  │ M7 ReAct Agent      │  原推理引擎(零修改)            │
│  └─────────┬───────────┘                                  │
│            ↓                                              │
│  ┌─────────────────────┐                                  │
│  │ Persona Guard       │  M10.4 — 漂移检测                │
│  │  全局 + 专属禁词    │  high / low / none 三级严重度    │
│  │  logger.warning     │                                  │
│  └─────────────────────┘                                  │
└───────────────────────────────────────────────────────────┘

---

## 3. 评测方法学

M10 是 **行为多样性升级**(从单一回答风格到 5 套任务模式).
评测采用 **同 query 跨 persona 对比 + Guard 触发率**.

| 维度 | 含义 |
|---|---|
| 风格差异 | 同 query 跑 5 个 persona,回复风格是否明显不同 |
| Persona 一致性 | 每个 persona 的回复是否符合人设 |
| Guard 检出 | 漂移触发率(none/low/high) |
| 集成无侵入 | M8 调用方零回归 |

---

## 4. 实测结果

### 4.1 评测 1:同 query 跨 5 个 persona ⭐ 核心证据

**Query:**
> 我刚被公司辞退,没有签书面合同,该怎么办?

| Persona | 耗时 | 风格关键特征 | 回复开头 |
|---|---|---|---|
| **default** | 53.2s | Markdown emoji + 大白话 | "好的,法律条文已经查全了" |
| **strict** | 28.5s | "法律分析意见" + 法条编号 | "根据《中华人民共和国劳动合同法》第十条第一款,**建立劳动关系,应当订立书面劳动合同**" |
| **friendly** | 20.6s | 🌸 + "别担心" + "你" | "好的,现在我来帮你梳理一下情况.别担心,虽然没签合同,但法律对你是**有保护的**" |
| **enterprise** | 20.0s | "行动建议" + 商务化 | "## 法律分析与行动建议... 涉及两个核心法律问题" |
| **litigation** | 27.2s | "可主张" + 具体金额举例 | "## 一、你的核心权利(法律上你可以主张什么)" |

**风格差异肉眼可见**:
- default 用 emoji 标号(1️⃣ 2️⃣)
- strict 用引用块强调法条文本
- friendly 用 "你" 而不是 "您",加 🌸 表情
- enterprise 强调 "关键点" + "行动建议"
- litigation 给具体金额("5 个月双倍工资差额")

**Guard 全部 drift=False, severity=none** — 5 个 persona 的回复都没自爆禁词.

### 4.2 评测 2:User Persona 跨轮注入

**Turn 1:** "我是上海的中学老师,30 岁"
- Entity 抽取异步触发(M9.1)
- 等待 5 秒让 facts 落库

**Turn 2:** "我能问个法律问题吗?涉及孩子抚养权那种"
- M8 user_facts 应已写入(location=上海, occupation=中学老师, age_range=30)
- M10 User Persona 翻译成自然语言画像注入 system prompt
- LLM 回复:列出 5 个抚养权常见问题(标准回答)

**观察**:
- ✅ 流程跑通,session 隔离正确
- ⚠️ LLM 没在回复中显式提及 "教师"/"上海" — 选择了泛化回答
- 这是 LLM 决策自由度,M10 不强制人设彩蛋

**结论**:User Persona 注入了 system prompt,但 LLM 是否显式引用取决于场景必要性.
若需强制提及画像,可在 prompt 加 "如适用,引用用户画像信息" 指令(M11 优化空间).

### 4.3 Guard 漂移检测覆盖

单元测试 9 个 case 全通过(`tests/persona/test_guard.py`):

| 触发场景 | Guard 检测 | 严重度 |
|---|---|---|
| 干净回复 | ✅ 不触发 | none |
| LLM 自称律师 | ✅ 触发 | high |
| 过度承诺("保证胜诉") | ✅ 触发 | high |
| Prompt 注入("忽略之前的指令") | ✅ 触发 | high |
| 教唆违法("如何规避法律") | ✅ 触发 | high |
| 非 default persona 仍能检测全局禁词 | ✅ 触发 | high |

---

## 5. 关键工程发现

### 发现 1:Persona 切换的 LLM 输出差异显著

5 个 persona 用 同样 LLM、同样工具、同样 query,只换 system prompt,
回复风格差异度 极高:emoji 用法、人称、结构、专业度全部不同.

→ 验证了 "system prompt 是 LLM 应用最强工程杠杆" 的工业共识.
→ 没有微调,没有 RAG 改造,纯 prompt engineering 实现产品差异化.

### 发现 2:M10 集成 M8 / M9 完全零回归

`inject_memory_into_messages` 加可选参数 `persona_mode`,
`persona_mode=None` 时 100% 走 M8 老路,
`persona_mode` 指定时启用 M10.

→ 体现 "扩展开放,修改封闭" 的 OCP 原则.
→ M8 的 4 个集成测试 + M9.1 的异步测试 全部 不需修改.

### 发现 3:User Persona 中间层的设计权衡

原始 user_facts(KV)→ 自然语言画像 的翻译用了模板拼接而非 LLM 翻译:

| 方案 | 延迟 | 一致性 | 成本 | M10 选择 |
|---|---|---|---|---|
| 模板拼接 | < 1ms | 100% | 0 | ✅ |
| LLM 翻译 | 200-500ms | ~95% | $$ | ❌(M11+ 再考虑) |

→ user_facts key 数量有限(6 个),模板完全够用.
→ LLM 翻译会引入随机性,Persona 可解释性反而下降.

### 发现 4:Guard 选 "仅记录" 而非 "重写回复"

M10.4 选择 logger.warning 而非主动重写漂移回复,理由:

1. **观察期优先**:M10 是 v0,需先收集真实漂移数据再设计治理策略
2. **误判成本**:重写过激会反伤正常回复("我建议" 包含 "我" 误判)
3. **可解释性**:logger 让运维清楚"哪些场景容易漂移"

→ M11+ 收集 1 周日志后,再决定是否上 LLM-as-Judge 主动重写.

### 发现 5:5 套 Persona 的取舍

法律场景的 5 套是 任务模式 不是 娱乐人格(消费 C 端的人格切换):
- default:通用咨询
- strict:商务合同 / 合规审查
- friendly:个人维权(用户焦虑)
- enterprise:企业法务
- litigation:诉讼方向

→ 每套都对应真实业务场景,不是为多样性而多样性.
→ API 调用方根据场景预选,不让用户挑(α 决策被否定的原因).

---

## 6. 已知局限 + 后续优化

### 6.1 当前局限

1. **User Persona 注入但 LLM 不一定显式引用**
   - Turn 2 LLM 选择了泛化回答而非 "作为老师..." 开头
   - 解决:M11 在 prompt 加 "必要时引用用户画像" 指令

2. **Guard 只检测关键词,不识别语义漂移**
   - LLM 可能用同义词绕过("我虽然不是律师但...")
   - 解决:M11+ 加 LLM-as-Judge 语义检测

3. **Persona 切换需要调用方决策**
   - 没有自动选 persona 的能力("用户问劳动纠纷 → friendly")
   - 解决:M11 / M12 加 Persona Auto-Router

4. **5 套 Persona 是产品决策,不一定符合所有部署**
   - 某些客户可能只需要 3 套或者要加新模式
   - 解决:yaml 改一下就能加,但需要重启服务

### 6.2 优化路线图

- **M11 知识图谱** ⭐:法条关系推理 + Persona 加 "如适用引用" 指令
- **M12 Proactive 主动推送**:基于 user_facts 主动提醒
- **M13 上线 + 监控**:收集 Guard 漂移率,完成闭环

---

## 7. M10 完成里程碑

✓ 5 套 Agent Persona(yaml 配置,5 种法律咨询任务模式)
✓ User Persona 中间层(KV → 自然语言画像)
✓ inject_memory_into_messages 加 persona_mode 参数(M8 零回归)
✓ Persona Guard 关键词检测(全局禁词 + 专属禁词,3 级严重度)
✓ react_agent_with_memory 集成 Guard(自动 logger 漂移)
✓ FastAPI `/chat/persona` 端点(支持 5 模式切换)
✓ 30+ 单元测试(loader / user_persona / injection / guard)
✓ 端到端评测(5 persona × 1 query + 跨轮画像注入)

---

## 8. 简历金句

> "设计并实现 5 套 Persona 防漂移系统(default / strict / friendly / enterprise / litigation),
> 通过 yaml + 中间层 + Guard 四层防护:Agent 角色锁定、User 画像注入、漂移检测、
> Prompt 注入识别.集成 M8 三层记忆零回归,M9 异步抽取流水线无影响."

> "评测验证 5 套 Persona 在同一 query 下的回复风格差异:
> default 用 emoji + 大白话,strict 用法条编号 + 引用块,friendly 用 \"你\" + 🌸 共情,
> enterprise 用 \"关键点\" + 商务化,litigation 用具体金额举例.
> 验证 system prompt 是 LLM 应用最强工程杠杆."

> "Persona Guard 关键词黑名单(全局 10 词 + 专属配置),
> 检测 LLM 自称律师 / 过度承诺 / 教唆违法 / Prompt 注入,
> 3 级严重度(high/low/none).为运维提供漂移频率观察窗口,
> 留待 M11+ 升 LLM-as-Judge 语义检测."

> "M10 与 M8 / M9 通过可选参数 persona_mode 解耦集成:
> persona_mode=None 走 M8 老路,M8 的 4 个集成测试 + M9 异步测试零回归.
> 体现\"扩展开放,修改封闭\"的 OCP 原则."
