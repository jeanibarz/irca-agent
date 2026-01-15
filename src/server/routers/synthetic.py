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


@router.post("/synthetic/query", response_model=SyntheticQueryResponse)
async def generate_synthetic_query(req: SyntheticQueryRequest) -> SyntheticQueryResponse:
    """
    Generate a synthetic user query using the loaded model.
    """
    manager = ModelManager.get_instance()
    if not manager.model:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")

    # tool_names removed as unused
    tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in req.tools])

    if req.type == "feasible":
        prompt = (
            f"[INST] You are a creative user testing an AI assistant.\n"
            f"The assistant has access to the following tools:\n{tool_descriptions}\n\n"
            f"Write a short, natural, one-sentence query that would require the assistant to use one of these tools.\n"
            f"Do not mention the tool name explicitly if possible, just ask for the utility.\n"
            f"Output ONLY the query, no quotes or explanation. [/INST]"
        )
    else:
        prompt = (
            f"[INST] You are a creative user testing an AI assistant.\n"
            f"The assistant has access to the following tools:\n{tool_descriptions}\n\n"
            f"Write a short, natural, one-sentence query that requires a capability DEFENITELY NOT in this list.\n"
            f"For example, if the tools are only for weather, ask for the latest news or stock prices or to send an email.\n"
            f"Output ONLY the query, no quotes or explanation. [/INST]"
        )

    # We use a simple generation call
    # Note: We assume the model follows instruction well enough.

    try:
        output = await manager.generate(
            prompt=prompt,
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
