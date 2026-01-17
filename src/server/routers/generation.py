import json
import logging

from fastapi import APIRouter, HTTPException

from src.formatting.chat_template import format_for_inference, get_stop_sequences
from src.server.model_manager import ModelManager
from src.server.schemas import GenerationRequest, GenerationResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Default IRCA system instructions - matches the training data format
DEFAULT_IRCA_INSTRUCTIONS = """You are an AI assistant with specific functions at your disposal. Your task is to answer the user's question succinctly. Utilize markdown links to refer to detailed data in the context when relevant.
To answer the user's query, engage in the Iterative Resolution Cycle. This cycle involves repeated 'Thought' and 'Call Function' steps until enough information is gathered to formulate a 'Final Thought'.

### ITERATIVE RESOLUTION CYCLE
Thought: Reflect on the necessary steps and functions to answer the question. Clearly state your plan or indicate if you cannot proceed and why.
Action choice: `call function` or `final answer`. This will determine if you will call a function next or if you will break the iterative resolution cycle and return a final answer.
Call function: Execute a function by passing a dictionary with the function's name and parameters in a dictionary format. Include `<|wait|>` after the call to await for results. Repeat the 'Thought' and 'Call Function' steps as necessary until you have all the information required to answer or determine that you cannot answer.

Once you have enough information or need to abort:
Final answer: Conclude the Iterative Resolution Cycle by preparing a concise answer or admitting the inability to provide a satisfactory response due to specific reasons.

Finally, write the response in markdown format, using markdown links to point the user to relevant detailed data if needed.
Action choice: final answer

### FINAL ANSWER
[Your concise final answer here, possibly with links to detailed outputs or a statement of inability to provide an answer with an explanation.]

EXAMPLE:
### ITERATIVE RESOLUTION CYCLE
Thought: To answer the user's request, I need to know the weather at their location. The first step is to identify the user's location.
Action choice: call function
Call function: {"name": "get_user_location"}<|wait|>
Output[ryuzyRNy98ue2sQkfBgfJr]: {'lat': 46.899, 'long': 56.4546}
Thought: Now, I can use the retrieved location to get the weather at the user's location.
Action choice: call function
Call function: {"name": "get_weather", "parameters": {"lat": 46.899, "long": 56.4546}}<|wait|>
Output[pTnEwkeyTVpzTjVcseLdrS]: {'temperature': 23.0, 'unit': 'celsius', 'rain': True}
Thought: I have all necessary information to answer the user's request.
Action choice: final answer

### FINAL ANSWER
Based on the [weather data](Output[pTnEwkeyTVpzTjVcseLdrS]) at [your current location](Output[ryuzyRNy98ue2sQkfBgfJr]), it is recommended to take an umbrella due to rain."""


@router.post("/completions", response_model=GenerationResponse)
async def generate_chat_completion(request: GenerationRequest) -> GenerationResponse:
    """
    Generate completion for chat messages.
    Handles prompt construction locally for now (adapting chat to prompt).
    """
    manager = ModelManager.get_instance()

    # Simple adapter: Convert messages to prompt expected by build_full_prompt
    # Assuming last message is user query, and we extract system prompt

    system_instruction = None
    user_query = ""

    for msg in request.messages:
        if msg.role == "system":
            system_instruction = msg.content
        elif msg.role == "user":
            user_query = msg.content

    # Use IRCA instructions if no custom system prompt provided
    if not system_instruction or system_instruction == "You are a helpful assistant.":
        system_instruction = DEFAULT_IRCA_INSTRUCTIONS

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
