from typing import Literal
from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str
    system_prompt: str | None = Field(default=None, max_length=4000)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v: str) -> str:
        stripped = v.strip()
        if not (1 <= len(stripped) <= 128):
            raise ValueError("session_id must be between 1 and 128 characters after stripping whitespace")
        return stripped

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 1:
            raise ValueError("message must not be empty or whitespace-only")
        if len(stripped) > 4000:
            raise ValueError("message must be at most 4000 characters")
        return stripped

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: str | None) -> str | None:
        if v is None:
            return None
        stripped = v.strip()
        if not stripped:
            return None
        if len(stripped) > 4000:
            raise ValueError("system_prompt must be at most 4000 characters")
        return stripped


class ResponseMetadata(BaseModel):
    model: str
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    key_id: str | None = None
    fallback_attempts: int = 0
    latency_ms: int
    input_tokens: int
    output_tokens: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ResponseMetadata


class HealthResponse(BaseModel):
    status: str
    keys_configured: int
    cache_enabled: bool
