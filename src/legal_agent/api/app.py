"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse

from legal_agent.agent.llm_client import generate_reply, stream_reply
from legal_agent.api.schemas import ChatRequest, ChatResponse
from legal_agent.core.config import get_settings
from legal_agent.core.version import get_version
from legal_agent.db.messages import (
    get_or_create_session,
    save_message,
)
from legal_agent.db.postgres import close_postgres_pool, init_postgres_pool
from legal_agent.db.redis_client import close_redis, init_redis


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
        """Send a message and get a non-streaming reply."""
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
        """Stream the reply as Server-Sent Events."""
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

    return app


app = create_app()
