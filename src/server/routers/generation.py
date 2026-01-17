import json
import logging

from fastapi import APIRouter, HTTPException

from src.formatting.chat_template import format_for_inference, get_stop_sequences
from src.server.model_manager import ModelManager
from src.server.schemas import GenerationRequest, GenerationResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/completions", response_model=GenerationResponse)
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

    # Get tokenizer from loaded model to apply proper chat template
    tokenizer = manager.tokenizers.get("default")
    if not tokenizer:
        # Fallback if only one model loaded
        if len(manager.tokenizers) == 1:
            tokenizer = list(manager.tokenizers.values())[0]
        else:
            raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")

    # Format prompt using model's native chat template
    # This ensures inference uses the same format as training
    prompt = format_for_inference(
        tokenizer=tokenizer,
        system_instructions=system_instruction,
        functions_json=functions_json,
        user_query=user_query,
    )

    # Get model-specific stop sequences
    stop_sequences = get_stop_sequences(tokenizer)
    logger.debug(f"Using stop sequences: {stop_sequences}")

    try:
        content = await manager.generate(
            prompt=prompt,
            alias="default",
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_tokens=stop_sequences,
        )

        return GenerationResponse(content=content)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
