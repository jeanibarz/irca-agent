import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.model_manager import ModelManager
from server.schemas import FunctionDefinition

logger = logging.getLogger(__name__)
router = APIRouter()


class SyntheticQueryRequest(BaseModel):
    tools: list[FunctionDefinition]
    type: Literal["feasible", "infeasible"]


class SyntheticQueryResponse(BaseModel):
    query: str


@router.post("/query", response_model=SyntheticQueryResponse)
async def generate_synthetic_query(req: SyntheticQueryRequest) -> SyntheticQueryResponse:
    """
    Generate a synthetic user query using the loaded model.
    """
    manager = ModelManager.get_instance()
    # Use the default model for synthetic generation
    tokenizer = manager.tokenizers.get("default")
    if not tokenizer:
        raise HTTPException(status_code=503, detail="No model loaded. Please load a model first.")

    # Prepare tool descriptions
    tool_descriptions = "\n".join([f"- {tool.name}: {tool.description}" for tool in req.tools])

    if req.type == "feasible":
        content = (
            "You are a creative user testing an AI assistant.\n"
            f"The assistant has access to the following tools:\n{tool_descriptions}\n\n"
            f"Write a short, natural, one-sentence query that would require the assistant to use one of these tools.\n"
            f"Do not mention the tool name explicitly if possible, just ask for the utility.\n"
            f"Output ONLY the query, no quotes or explanation."
        )
    else:
        content = (
            f"You are a creative user testing an AI assistant.\n"
            f"The assistant has access to the following tools:\n{tool_descriptions}\n\n"
            f"Write a short, natural, one-sentence query that requires a capability DEFENITELY NOT in this list.\n"
            f"For example, if the tools are only for weather, ask for the latest news or stock prices or to send an email.\n"
            f"Output ONLY the query, no quotes or explanation."
        )

    # Use chat template to ensure model understands the instruction format
    messages = [{"role": "user", "content": content}]

    try:
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"[INST] {content} [/INST]"
    except Exception:
        # Fallback if template fails
        prompt = f"[INST] {content} [/INST]"

    try:
        output = await manager.generate(
            prompt=prompt,
            alias="default",
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.95,
            stop_tokens=["\n", "[/INST]", "User:", "Assistant:"],
        )

        # Clean up output
        query = output.strip().strip('"')
        return SyntheticQueryResponse(query=query)

    except Exception as e:
        logger.error(f"Failed to generate synthetic query: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
