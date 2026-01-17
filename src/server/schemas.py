from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class FunctionDefinition(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    description: str = Field(..., max_length=4096)
    parameters: dict[str, Any]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function"] = Field(
        ..., description="Message role"
    )
    content: str = Field(..., max_length=100000)  # ~25k tokens max


class GenerationRequest(BaseModel):
    """
    Request schema for generation API with input validation (FM-13).
    """

    # Standard Chat Format
    messages: list[Message] = Field(..., min_length=1, max_length=100)
    functions: list[FunctionDefinition] | None = Field(default=None, max_length=50)

    # Generation Parameters with bounded ranges
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=4096, ge=1, le=32768)  # FM-13: Upper bound
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)

    # Model Selection (Optional, validation happens in manager)
    model_id: str | None = Field(default=None, max_length=256)
    adapter_id: str | None = Field(default=None, max_length=512)

    @field_validator("messages")
    @classmethod
    def validate_messages_not_empty(cls, v: list[Message]) -> list[Message]:
        """Ensure at least one message is provided."""
        if not v:
            raise ValueError("At least one message is required")
        return v


class GenerationResponse(BaseModel):
    role: str = "assistant"
    content: str
    finish_reason: str = "stop"
