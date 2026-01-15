import json
import logging

from fastapi import APIRouter, HTTPException

from core.prompt_builder import build_full_prompt  # Reuse existing logic
from server.model_manager import ModelManager
from server.schemas import GenerationRequest, GenerationResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat/completions", response_model=GenerationResponse)
async def generate_chat_completion(request: GenerationRequest) -> GenerationResponse:
    """
    Generate completion for chat messages.
    Handles prompt construction locally for now (adapting chat to prompt).
    """
    manager = ModelManager.get_instance()

    # Simple adapter: Convert messages to prompt expected by build_full_prompt
    # Assuming last message is user query, and we extract system prompt

    system_instruction = "You are a helpful assistant."
    user_query = ""

    for msg in request.messages:
        if msg.role == "system":
            system_instruction = msg.content
        elif msg.role == "user":
            user_query = msg.content

    # Convert functions to expected dict format if present
    functions_json = "[]"
    if request.functions:
        functions_list = [f.dict() for f in request.functions]
        # remove None values to be clean
        functions_list = [{k: v for k, v in f.items() if v is not None} for f in functions_list]
        functions_json = json.dumps(functions_list, indent=4)

    sample = {
        "system_instructions": system_instruction,
        "example": "",
        "available_functions_json": functions_json,
        "user_query": user_query,
        "assistant_completion": "",
    }

    prompt = build_full_prompt(sample)

    try:
        # Implicitly use currently loaded model if IDs not provided
        # Or load default if nothing loaded.
        if not manager.model:
            # Fallback default (this should be configurable)
            # For now, simplistic error
            raise HTTPException(status_code=503, detail="Model not loaded. Use /v1/model/load endpoint.")

        content = manager.generate(
            prompt=prompt, max_new_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p
        )

        return GenerationResponse(content=content)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
