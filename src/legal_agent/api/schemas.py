"""API request/response schemas using Pydantic."""

from uuid import UUID

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """User chat input."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User's chat message",
        examples=["什么是合同违约?"],
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt to override the default",
        examples=["你是一位专业的法律顾问"],
    )
    session_id: UUID | None = Field(
        default=None,
        description="Existing session UUID. Omit to start a new session.",
    )
    # ⭐ M10: Persona 模式选择(/chat/persona 端点用)
    persona_mode: str | None = Field(
        default=None,
        description="Persona mode: default/strict/friendly/enterprise/litigation",
        examples=["default", "strict"],
    )


class ChatResponse(BaseModel):
    """AI chat response."""

    reply: str = Field(..., description="The AI's reply text")
    model: str = Field(..., description="The model that produced this reply")
    session_id: UUID = Field(..., description="The session this turn belongs to")

    # ⭐ M5: Two-Stage RAG 可观测字段(可选,旧客户端兼容)
    abstained: bool | None = Field(
        default=None,
        description="Whether the system abstained from answering due to low confidence",
    )
    top_rerank: float | None = Field(
        default=None,
        description="Top-1 reranker score (M5: useful for monitoring/eval)",
    )
