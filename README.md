# Legal Agent · 企业级智能法律顾问

基于 LangGraph + FastAPI 构建的端到端 AI 法律咨询 Agent。

## ✨ 项目特性

> 项目正在分模块迭代中,当前进度: **M0 · 项目脚手架**

- [ ] M1 · 最小聊天服务(SSE 流式输出)
- [ ] M2 · 数据存储底座(PostgreSQL + Redis + pgvector)
- [ ] M3 · 文档摄入与分块
- [ ] M4 · 双路检索(BM25 + 向量)
- [ ] M5 · Hybrid Search + Reranker
- [ ] M6 · Tool Use 基础
- [ ] M7 · LangGraph 编排(ReAct + Plan-Execute)
- [ ] M8 · 三层记忆架构
- [ ] M9 · 模型路由与三级 Fallback
- [ ] M10 · Persona 防漂移
- [ ] M11 · Neo4j 知识图谱
- [ ] M12 · Proactive 主动推送
- [ ] M13 · 可观测性与上线

## 🛠 技术栈

- **语言**: Python 3.11+
- **包管理**: uv
- **代码质量**: ruff + mypy + pytest
- (后续模块的技术栈会逐步加入)

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/<你的用户名>/legal-agent.git
cd legal-agent

# 安装依赖
uv sync

# 运行测试
uv run pytest
```

## 📁 项目结构

\`\`\`
src/legal_agent/
├── api/      # FastAPI 路由(M1)
├── agent/    # LangGraph 编排(M7)
├── rag/      # 检索引擎(M3-M5)
├── memory/   # 三层记忆(M8)
├── tools/    # 工具集(M6)
├── core/     # 配置、日志、通用
└── db/       # 数据库连接(M2)
\`\`\`

## 📝 学习笔记

每个模块完成后会在 `docs/notes/` 下记录学习笔记和踩坑日志。

## 📜 License

MIT