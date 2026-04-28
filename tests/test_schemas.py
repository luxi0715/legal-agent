"""Test API schemas."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from legal_agent.api.schemas import ChatRequest, ChatResponse


def test_chat_request_minimal() -> None:
    """A minimal valid request needs only a message."""
    req = ChatRequest(message="hello")
    assert req.message == "hello"
    assert req.system_prompt is None
    assert req.session_id is None


def test_chat_request_with_system_prompt() -> None:
    """System prompt is optional."""
    req = ChatRequest(message="hi", system_prompt="You are a lawyer")
    assert req.system_prompt == "You are a lawyer"


def test_chat_request_with_session_id() -> None:
    """Session ID is optional."""
    sid = uuid4()
    req = ChatRequest(message="hi", session_id=sid)
    assert req.session_id == sid


def test_chat_request_rejects_empty_message() -> None:
    """Empty message should fail validation."""
    with pytest.raises(ValidationError):
        ChatRequest(message="")


def test_chat_request_rejects_too_long_message() -> None:
    """Message longer than 4000 chars should fail."""
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 4001)


def test_chat_response_basic() -> None:
    """ChatResponse should hold reply, model, and session_id."""
    sid = uuid4()
    resp = ChatResponse(reply="你好", model="deepseek-chat", session_id=sid)
    assert resp.reply == "你好"
    assert resp.model == "deepseek-chat"
    assert resp.session_id == sid
