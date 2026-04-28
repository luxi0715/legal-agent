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


class ChatResponse(BaseModel):
    """AI chat response."""

    reply: str = Field(..., description="The AI's reply text")
    model: str = Field(..., description="The model that produced this reply")
    session_id: UUID = Field(..., description="The session this turn belongs to")
