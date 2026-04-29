"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from legal_agent.agent.llm_client import (
    generate_reply,
    generate_reply_with_rag,
    stream_reply,
)
from legal_agent.api.schemas import ChatRequest, ChatResponse
from legal_agent.core.config import get_settings
from legal_agent.core.version import get_version
from legal_agent.db.messages import get_or_create_session, save_message
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.db.redis_client import close_redis, init_redis
from legal_agent.rag.hybrid_retriever import hybrid_retrieve
from legal_agent.rag.retriever import retrieve


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and clean up resources tied to the app lifecycle."""
    await init_postgres_pool()
    await init_redis()
    yield
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
        """Hybrid search: vector + BM25 fused with RRF."""
        results = await hybrid_retrieve(req.message, top_k=5, candidates_per_method=50)
        return {"results": [dict(r) for r in results]}

    @app.post("/chat/rag", response_model=ChatResponse)
    async def chat_rag(req: ChatRequest) -> ChatResponse:
        """RAG-enhanced chat (now using hybrid retrieval)."""
        session_id = await get_or_create_session(req.session_id)
        await save_message(session_id, "user", req.message)

        chunks = await hybrid_retrieve(req.message, top_k=5, candidates_per_method=50)
        reply = await generate_reply_with_rag(
            user_message=req.message,
            retrieved_chunks=[dict(c) for c in chunks],
            system_prompt=req.system_prompt,
        )
        await save_message(session_id, "assistant", reply)

        settings = get_settings()
        return ChatResponse(
            reply=reply,
            model=settings.deepseek_model,
            session_id=session_id,
        )

    return app


app = create_app()
