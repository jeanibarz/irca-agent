import ast
import json
import logging
import operator
import random
import re
import string

from fastapi import APIRouter, HTTPException

from src.formatting.chat_template import format_for_inference, get_stop_sequences
from src.server.model_manager import ModelManager
from src.server.schemas import GenerationRequest, GenerationResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Maximum iterations for the agentic loop to prevent infinite loops
MAX_AGENT_ITERATIONS = 5


def generate_output_id() -> str:
    """Generate a random output ID like 'Output[abc123xyz]'."""
    chars = string.ascii_letters + string.digits
    return "".join(random.choices(chars, k=12))


# FM-01: Safe math expression evaluator (replaces dangerous eval())
# Only supports basic arithmetic: +, -, *, /, parentheses, and numbers
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node for safe math expressions."""
    if isinstance(node, ast.Constant):  # Python 3.8+
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Num):  # Python 3.7 compatibility
        return float(node.n)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        return _SAFE_OPERATORS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp):
        if type(node.op) not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        operand = _safe_eval_node(node.operand)
        return _SAFE_OPERATORS[type(node.op)](operand)
    elif isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


def safe_math_eval(expression: str) -> float:
    """
    Safely evaluate a math expression string.

    Only supports: numbers, +, -, *, /, parentheses.
    Raises ValueError for invalid or unsafe expressions.

    Examples:
        safe_math_eval("2 + 3") -> 5.0
        safe_math_eval("10 / 2 * 3") -> 15.0
        safe_math_eval("(1 + 2) * 3") -> 9.0
    """
    if not expression or not expression.strip():
        return 0.0

    try:
        tree = ast.parse(expression, mode="eval")
        return _safe_eval_node(tree)
    except (SyntaxError, ValueError) as e:
        logger.warning(f"Invalid math expression '{expression}': {e}")
        raise ValueError(f"Invalid math expression: {expression}") from e


def generate_mock_output(function_name: str, parameters: dict) -> str:
    """Generate a plausible mock output for a function call."""
    mock_outputs = {
        "get_weather": lambda p: {
            "temperature": random.randint(15, 30),
            "unit": "celsius",
            "condition": random.choice(["sunny", "cloudy", "rainy"]),
        },
        "get_stock_price": lambda p: {
            "ticker": p.get("ticker", "AAPL"),
            "price": round(random.uniform(100, 500), 2),
            "currency": "USD",
        },
        # FM-01: Use safe_math_eval instead of eval() to prevent RCE
        "calculator": lambda p: {
            "result": safe_math_eval(p.get("expression", "0"))
            if p.get("expression")
            else 0
        },
        "get_user_location": lambda p: {
            "lat": round(random.uniform(40, 50), 3),
            "long": round(random.uniform(-5, 10), 3),
        },
    }

    generator = mock_outputs.get(function_name, lambda p: {"status": "success", "data": "Mock response"})
    try:
        return str(generator(parameters))
    except Exception as e:
        # FM-40: Log error instead of silently swallowing, return error indicator
        logger.warning(f"Mock output generation failed for {function_name}: {e}")
        return str({"status": "error", "error": f"Mock generation failed: {type(e).__name__}"})


def extract_function_call(text: str) -> tuple[str, dict] | None:
    """Extract function name and parameters from generated text."""
    # Look for JSON function call pattern
    patterns = [
        r"Call function:\s*(\{[^}]+\})",
        r"Function Call[:\s]*(\{[^}]+\})",
        r'"name":\s*"([^"]+)"[^}]*"parameters":\s*(\{[^}]*\})',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            try:
                if len(match.groups()) == 1:
                    call_json = json.loads(match.group(1))
                    return call_json.get("name", "unknown"), call_json.get("parameters", {})
                else:
                    return match.group(1), json.loads(match.group(2))
            except json.JSONDecodeError:
                continue
    return None


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
    # IMPORTANT: Use compact JSON (no indent) to match training data format
    functions_json = "[]"
    if request.functions:
        functions_list = [f.dict() for f in request.functions]
        # remove None values to be clean
        functions_list = [{k: v for k, v in f.items() if v is not None} for f in functions_list]
        functions_json = json.dumps(functions_list, separators=(",", ": "))

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
        # Agentic loop: continue generation until FINAL ANSWER or max iterations
        full_response = ""
        current_prompt = prompt
        iteration = 0

        while iteration < MAX_AGENT_ITERATIONS:
            iteration += 1
            logger.info(f"Agent iteration {iteration}/{MAX_AGENT_ITERATIONS}")

            # Generate next chunk
            chunk = await manager.generate(
                prompt=current_prompt,
                alias="default",
                max_new_tokens=request.max_tokens or 4096,
                temperature=request.temperature or 0.7,
                top_p=request.top_p or 1.0,
                stop_tokens=stop_sequences,
            )

            full_response += chunk
            logger.info(f"Generated chunk ({len(chunk)} chars): {chunk[:100]}...")

            # Check if we've reached the final answer
            if "### FINAL ANSWER" in full_response or "FINAL ANSWER" in chunk:
                logger.info("Reached FINAL ANSWER, stopping agent loop")
                break

            # Check if there's a function call to handle
            func_call = extract_function_call(chunk)
            if func_call:
                func_name, func_params = func_call
                logger.info(f"Detected function call: {func_name}({func_params})")

                # Generate mock output
                output_id = generate_output_id()
                mock_result = generate_mock_output(func_name, func_params)
                output_line = f"\nOutput[{output_id}]: {mock_result}\n"

                # Add to response and continue prompt
                full_response += output_line
                current_prompt = current_prompt + chunk + output_line

                logger.info(f"Added mock output: {output_line.strip()}")
            else:
                # No function call and no final answer - might be stuck
                logger.warning("No function call detected and no FINAL ANSWER, stopping")
                break

        return GenerationResponse(content=full_response)

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
