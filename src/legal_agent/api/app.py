"""FastAPI application entry point."""

import asyncio
import sys

# M9.3: psycopg async 在 Windows 不兼容默认的 ProactorEventLoop,
# 切到 SelectorEventLoop 才能用 LangGraph PG checkpointer.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from legal_agent.agent.agent_loop import run_agent_loop
from legal_agent.agent.checkpointer import close_checkpointer, init_checkpointer
from legal_agent.agent.llm_client import (
    generate_reply,
    generate_reply_with_two_stage,
    stream_reply,
)
from legal_agent.agent.react_agent import run_react_agent
from legal_agent.agent.react_agent_with_memory import run_react_agent_with_memory
from legal_agent.api.schemas import ChatRequest, ChatResponse
from legal_agent.core.config import get_settings
from legal_agent.core.version import get_version
from legal_agent.db.messages import get_or_create_session, save_message
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.db.redis_client import close_redis, init_redis
from legal_agent.rag.abstention import decide_abstention
from legal_agent.rag.context_reorder import reorder_for_lost_in_the_middle
from legal_agent.rag.hybrid_retriever import hybrid_retrieve
from legal_agent.rag.retriever import retrieve
from legal_agent.rag.two_stage_retriever import two_stage_retrieve


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up resources tied to the app lifecycle."""
    await init_postgres_pool()
    await init_redis()
    await init_checkpointer()  # M9.3: ReAct 状态持久化
    yield
    await close_checkpointer()
    await close_postgres_pool()
    await close_redis()


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(
        title="Legal Agent API",
        description="企业级智能法律顾问 Agent",
        version=get_version(),
        lifespan=lifespan,
    )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "legal-agent", "version": get_version()}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        """Plain chat (no RAG)."""
        session_id = await get_or_create_session(req.session_id)
        await save_message(session_id, "user", req.message)
        reply = await generate_reply(
            user_message=req.message,
            system_prompt=req.system_prompt,
        )
        await save_message(session_id, "assistant", reply)
        settings = get_settings()
        return ChatResponse(
            reply=reply,
            model=settings.deepseek_model,
            session_id=session_id,
        )

    @app.post("/chat/stream")
    async def chat_stream(req: ChatRequest) -> EventSourceResponse:
        """Streaming chat (no RAG)."""
        session_id = await get_or_create_session(req.session_id)
        await save_message(session_id, "user", req.message)

        async def event_generator() -> AsyncIterator[dict[str, str]]:
            full_reply: list[str] = []
            yield {"event": "session", "data": str(session_id)}
            async for chunk in stream_reply(
                user_message=req.message,
                system_prompt=req.system_prompt,
            ):
                full_reply.append(chunk)
                yield {"data": chunk}
            await save_message(session_id, "assistant", "".join(full_reply))
            yield {"event": "done", "data": ""}

        return EventSourceResponse(event_generator())

    @app.post("/search")
    async def search(req: ChatRequest) -> dict[str, list[dict[str, Any]]]:
        """Vector-only search (M3 baseline)."""
        results = await retrieve(req.message, top_k=5, min_score=0.3)
        return {"results": [dict(r) for r in results]}

    @app.post("/search/hybrid")
    async def search_hybrid(req: ChatRequest) -> dict[str, list[dict[str, Any]]]:
        """Hybrid search: vector + BM25 fused with RRF (M4 baseline)."""
        results = await hybrid_retrieve(req.message, top_k=5, candidates_per_method=50)
        return {"results": [dict(r) for r in results]}

    @app.post("/search/rerank")
    async def search_rerank(req: ChatRequest) -> dict[str, Any]:
        """Two-Stage retrieval: Hybrid recall + Cross-encoder rerank (M5)."""
        reranked = await two_stage_retrieve(
            query=req.message,
            recall_top_k=50,
            final_top_k=5,
        )
        decision = decide_abstention(reranked)

        if decision.should_abstain:
            return {
                "results": [],
                "abstained": True,
                "reason": decision.reason,
                "user_message": decision.user_message,
                "top_rerank": decision.top_rerank,
                "top_recall": decision.top_recall,
            }

        reordered = reorder_for_lost_in_the_middle(reranked)
        return {
            "results": [dict(r) for r in reordered],
            "abstained": False,
            "top_rerank": decision.top_rerank,
            "top_recall": decision.top_recall,
        }

    @app.post("/chat/rag", response_model=ChatResponse)
    async def chat_rag(req: ChatRequest) -> ChatResponse:
        """RAG-enhanced chat with Two-Stage retrieval + Abstention (M5)."""
        session_id = await get_or_create_session(req.session_id)
        await save_message(session_id, "user", req.message)

        result = await generate_reply_with_two_stage(
            user_message=req.message,
            system_prompt=req.system_prompt,
        )
        await save_message(session_id, "assistant", result.reply)

        settings = get_settings()
        return ChatResponse(
            reply=result.reply,
            model=settings.deepseek_model,
            session_id=session_id,
            abstained=result.abstained,
            top_rerank=result.top_rerank,
        )

    @app.post("/chat/agent", response_model=ChatResponse)
    async def chat_agent(req: ChatRequest) -> ChatResponse:
        """Agent chat: LLM autonomously chooses tools (M6 ⭐).

        Uses while-loop based agent_loop, single-tool-per-iteration pattern.
        """
        session_id = await get_or_create_session(req.session_id)
        await save_message(session_id, "user", req.message)

        trace = await run_agent_loop(
            user_message=req.message,
            system_prompt=req.system_prompt,
        )
        await save_message(session_id, "assistant", trace.final_reply)

        settings = get_settings()
        return ChatResponse(
            reply=trace.final_reply,
            model=settings.deepseek_model,
            session_id=session_id,
        )

    @app.post("/chat/react")
    async def chat_react(req: ChatRequest) -> dict[str, Any]:
        """ReAct Agent chat: LangGraph-based reasoning loop (M7 ⭐).

        vs M6 /chat/agent:
        - M6: imperative while-loop, control flow embedded in code.
        - M7: declarative StateGraph, thinker / actor nodes,
          supports parallel tool calls and easier extension.
        """
        session_id = await get_or_create_session(req.session_id)
        await save_message(session_id, "user", req.message)

        result = await run_react_agent(
            user_message=req.message,
            system_prompt=req.system_prompt,
        )
        await save_message(session_id, "assistant", result["final_reply"])

        settings = get_settings()
        return {
            "reply": result["final_reply"],
            "model": settings.deepseek_model,
            "session_id": str(session_id),
            "iterations": result["iterations"],
            "tool_calls_count": len(result["tool_calls_log"]),
            "tool_calls": result["tool_calls_log"],
        }

    @app.post("/chat/memory")
    async def chat_memory(req: ChatRequest) -> dict[str, Any]:
        """ReAct Agent + 三层记忆 (M8 ⭐⭐⭐).

        vs M7 /chat/react:
        - M7: 单轮无上下文,无法解析\"那\"\"他\"\"上次说的\"等指代
        - M8: Buffer(滑窗) + Summary(长程) + Hard Memory(用户档案)
              自动注入,跨轮记忆持续性

        M8 决策(详见 docs/notes/M8-evaluation.md):
        - user_id = session_id(伪用户标识,M9 接入正式用户系统时无需迁移)
        - Buffer 7 轮(14 条消息)
        - Buffer 满触发 Summary 压缩
        - 同步 Entity 抽取(每轮 +1 LLM 调用,~1-3 秒延迟)
        """
        session_id = await get_or_create_session(req.session_id)
        # M8.0 决策 1:user_id 用 UUID,初期值 = session_id
        user_id = session_id

        await save_message(session_id, "user", req.message)

        result = await run_react_agent_with_memory(
            user_message=req.message,
            user_id=user_id,
            session_id=session_id,
            system_prompt=req.system_prompt,
        )
        await save_message(session_id, "assistant", result["final_reply"])

        settings = get_settings()
        return {
            "reply": result["final_reply"],
            "model": settings.deepseek_model,
            "session_id": str(session_id),
            "iterations": result["iterations"],
            "tool_calls_count": len(result["tool_calls_log"]),
            "tool_calls": result["tool_calls_log"],
            "memory_meta": result["memory_meta"],
        }

    @app.post("/chat/persona")
    async def chat_persona(req: ChatRequest) -> dict[str, Any]:
        """ReAct + Memory + Persona + Guard (M10 ⭐⭐⭐).

        vs M8 /chat/memory:
        - M8: 单一 system_prompt + 用户档案 KV
        - M10: 5 套 Persona(default/strict/friendly/enterprise/litigation)
              + 自然语言用户画像 + 漂移检测

        Persona 模式选择:
        - default     :通用法律顾问助手
        - strict      :严谨法律分析师(商务合同 / 合规)
        - friendly    :亲切助手(个人维权 / 焦虑用户)
        - enterprise  :企业法务顾问
        - litigation  :诉讼方向(已发生纠纷)

        通过 query 参数 persona_mode 切换,默认 default.
        """
        session_id = await get_or_create_session(req.session_id)
        user_id = session_id

        await save_message(session_id, "user", req.message)

        # M10 — persona_mode 从 request 读取,默认 default
        persona_mode = getattr(req, "persona_mode", None) or "default"

        result = await run_react_agent_with_memory(
            user_message=req.message,
            user_id=user_id,
            session_id=session_id,
            system_prompt=req.system_prompt,
            persona_mode=persona_mode,
        )
        await save_message(session_id, "assistant", result["final_reply"])

        settings = get_settings()
        return {
            "reply": result["final_reply"],
            "model": settings.deepseek_model,
            "session_id": str(session_id),
            "persona_mode": result["persona_mode"],
            "iterations": result["iterations"],
            "tool_calls_count": len(result["tool_calls_log"]),
            "tool_calls": result["tool_calls_log"],
            "memory_meta": result["memory_meta"],
            "guard": result["guard"],
        }

    return app


app = create_app()
