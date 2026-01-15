from typing import Any

from pydantic import BaseModel, Field


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class Message(BaseModel):
    role: str
    content: str


class GenerationRequest(BaseModel):
    # Standard Chat Format
    messages: list[Message]
    functions: list[FunctionDefinition] | None = None

    # Generation Parameters
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=4096, ge=1)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)

    # Model Selection (Optional, validation happens in manager)
    model_id: str | None = None
    adapter_id: str | None = None


class GenerationResponse(BaseModel):
    role: str = "assistant"
    content: str
    finish_reason: str = "stop"
