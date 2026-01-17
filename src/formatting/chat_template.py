"""
Chat Template Formatting for Model-Specific Training and Inference.

This module provides utilities to convert between model-agnostic message format
and model-specific chat templates. This ensures:
1. Training data includes proper EOS tokens for each model
2. Inference uses the same format as training
3. Models learn when to stop generating

Usage:
    # Convert IRCA format to messages
    messages = irca_to_messages(sample)

    # Apply model's chat template
    text = apply_chat_template(tokenizer, messages)

    # For inference, get just the prompt (no assistant completion)
    prompt = apply_chat_template(tokenizer, messages, add_generation_prompt=True)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


def irca_to_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    """
    Convert IRCA dataset format to model-agnostic messages format.

    The IRCA format has these fields:
    - system_instructions: System prompt
    - example: Example interaction (optional)
    - available_functions_json: JSON list of available functions
    - user_query: The user's question
    - assistant_completion: The agent's full response (thoughts + actions + answer)

    Args:
        sample: Dictionary with IRCA format fields.

    Returns:
        List of message dictionaries with 'role' and 'content' keys.
    """
    messages = []

    # Build system message with instructions and function definitions
    system_parts = []

    system_instructions = sample.get("system_instructions", "").strip()
    if system_instructions:
        system_parts.append(system_instructions)

    example = sample.get("example", "").strip()
    if example:
        system_parts.append(f"EXAMPLE:\n{example}")

    functions_json = sample.get("available_functions_json", "")
    if functions_json:
        if isinstance(functions_json, str):
            functions_str = functions_json
        else:
            functions_str = json.dumps(functions_json, indent=2)
        system_parts.append(
            f"The functions available to you are described below.\n\n### FUNCTIONS AVAILABLE\n{functions_str}"
        )

    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    # User message
    user_query = sample.get("user_query", "").strip()
    if user_query:
        messages.append({"role": "user", "content": user_query})

    # Assistant completion (the training target)
    assistant_completion = sample.get("assistant_completion", "").strip()
    if assistant_completion:
        messages.append({"role": "assistant", "content": assistant_completion})

    return messages


def parse_irca_text_to_messages(text: str) -> list[dict[str, str]]:
    """
    Parse a full IRCA-formatted text string into messages.

    This handles the legacy format with markers like:
    - ### INSTRUCTIONS
    - ### FUNCTIONS AVAILABLE
    - ### USER QUERY
    - ### ITERATIVE RESOLUTION CYCLE

    Args:
        text: Full IRCA-formatted prompt text.

    Returns:
        List of message dictionaries.
    """
    from src.core.prompt_builder import parse_corrected_agent_trace

    # Parse the text into components
    parsed = parse_corrected_agent_trace(text)

    # Convert to messages format
    return irca_to_messages(parsed)


def apply_chat_template(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = False,
    tokenize: bool = False,
) -> str | list[int]:
    """
    Apply the model's native chat template to messages.

    This ensures the output includes proper special tokens (EOS, etc.)
    that the model was pre-trained with.

    IMPORTANT: Some models (e.g., Mistral) drop the system message when the
    last message is from the assistant (training scenario). This function
    detects and fixes this by prepending the system content to the first
    user message when necessary.

    Args:
        tokenizer: The model's tokenizer (has apply_chat_template method).
        messages: List of message dictionaries with 'role' and 'content'.
        add_generation_prompt: If True, add the prompt for assistant to continue.
                              Use True for inference, False for training.
        tokenize: If True, return token IDs instead of string.

    Returns:
        Formatted text string or token IDs.

    Example:
        # For training (full conversation with EOS):
        text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)

        # For inference (prompt only, ready for generation):
        prompt = apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        logger.warning(
            f"Tokenizer {type(tokenizer).__name__} doesn't have apply_chat_template. "
            "Falling back to simple concatenation."
        )
        return _fallback_format(messages, add_generation_prompt)

    try:
        # Check if there's a system message that might get dropped
        fixed_messages = _fix_system_message_for_training(tokenizer, messages)

        return tokenizer.apply_chat_template(
            fixed_messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )
    except Exception as e:
        logger.warning(f"apply_chat_template failed: {e}. Using fallback.")
        return _fallback_format(messages, add_generation_prompt)


def _fix_system_message_for_training(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    Fix system message handling for models that drop it during training.

    Some models (Mistral family) only include the system message when the last
    message is from the user (inference). For training (where assistant is last),
    the system message is silently dropped.

    This function detects this issue and prepends the system content to the
    first user message to ensure it's included in training.

    Args:
        tokenizer: The model's tokenizer.
        messages: Original list of messages.

    Returns:
        Fixed messages with system content preserved.
    """
    # Quick checks - no fix needed if:
    # 1. No messages
    # 2. No system message
    # 3. Last message is from user (system will be included)
    if not messages:
        return messages

    has_system = messages[0]["role"] == "system"
    if not has_system:
        return messages

    # Last message is user - system should be included (inference case)
    if messages[-1]["role"] == "user":
        return messages

    # Training case: last message is assistant
    # Test if system message gets dropped by the template
    system_content = messages[0]["content"]

    # Use a unique marker to detect if system is included
    test_marker = "<<SYSTEM_TEST_MARKER_12345>>"
    test_messages = [
        {"role": "system", "content": test_marker},
        {"role": "user", "content": "test"},
        {"role": "assistant", "content": "test"},
    ]

    try:
        test_output = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        system_included = test_marker in test_output
    except Exception:
        # If test fails, assume system is included
        return messages

    if system_included:
        # Template handles system correctly
        return messages

    # System message is dropped - fix by prepending to first user message
    logger.debug(
        "Detected chat template that drops system message for training. "
        "Prepending system content to first user message."
    )

    fixed_messages = []
    system_prepended = False

    for msg in messages:
        if msg["role"] == "system":
            # Skip system message, we'll prepend it to user message
            continue
        elif msg["role"] == "user" and not system_prepended:
            # Prepend system content to first user message
            fixed_messages.append(
                {
                    "role": "user",
                    "content": f"{system_content}\n\n{msg['content']}",
                }
            )
            system_prepended = True
        else:
            fixed_messages.append(msg)

    return fixed_messages


def _fallback_format(messages: list[dict[str, str]], add_generation_prompt: bool) -> str:
    """
    Simple fallback format when chat template is not available.

    Uses a generic format that works with most instruction-tuned models.
    """
    parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            parts.append(f"### SYSTEM\n{content}")
        elif role == "user":
            parts.append(f"### USER\n{content}")
        elif role == "assistant":
            parts.append(f"### ASSISTANT\n{content}")

    text = "\n\n".join(parts)

    if add_generation_prompt:
        text += "\n\n### ASSISTANT\n"

    return text


def format_for_training(
    tokenizer: PreTrainedTokenizer,
    sample: dict[str, Any],
) -> str:
    """
    Format a sample for training with the model's native chat template.

    This is the main function to use when preparing training data.
    It ensures the EOS token is included at the end.

    Args:
        tokenizer: The model's tokenizer.
        sample: IRCA format sample dictionary.

    Returns:
        Formatted text ready for training.
    """
    messages = irca_to_messages(sample)
    text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)

    # Ensure EOS token is present (some templates don't add it automatically)
    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text = text + tokenizer.eos_token

    return text


def format_for_inference(
    tokenizer: PreTrainedTokenizer,
    system_instructions: str,
    functions_json: str | list[dict],
    user_query: str,
) -> str:
    """
    Format a prompt for inference (generation).

    Creates the prompt up to where the assistant should start responding.

    Args:
        tokenizer: The model's tokenizer.
        system_instructions: System prompt / instructions.
        functions_json: Available functions as JSON string or list.
        user_query: The user's question.

    Returns:
        Formatted prompt ready for generation.
    """
    sample = {
        "system_instructions": system_instructions,
        "available_functions_json": functions_json,
        "user_query": user_query,
        "assistant_completion": "",  # Empty - we want the model to generate this
    }

    messages = irca_to_messages(sample)

    # Remove empty assistant message if present
    if messages and messages[-1]["role"] == "assistant" and not messages[-1]["content"]:
        messages = messages[:-1]

    # Add generation prompt (signals model to start generating)
    return apply_chat_template(tokenizer, messages, add_generation_prompt=True)


def get_model_eos_token(tokenizer: PreTrainedTokenizer) -> str:
    """
    Get the EOS token for a model.

    Different models use different EOS tokens:
    - Qwen: <|im_end|>
    - Mistral: </s>
    - Llama3: <|eot_id|>

    Args:
        tokenizer: The model's tokenizer.

    Returns:
        The EOS token string.
    """
    return tokenizer.eos_token or ""


def get_stop_sequences(tokenizer: PreTrainedTokenizer) -> list[str]:
    """
    Get appropriate stop sequences for a model.

    Returns model-specific stop sequences that should terminate generation.

    Args:
        tokenizer: The model's tokenizer.

    Returns:
        List of stop sequence strings.
    """
    stop_sequences = []

    # Always include EOS token
    if tokenizer.eos_token:
        stop_sequences.append(tokenizer.eos_token)

    # Check for common special tokens
    special_tokens = getattr(tokenizer, "special_tokens_map", {})

    # Qwen-style tokens
    if "<|im_end|>" in str(special_tokens) or "im_end" in str(tokenizer.eos_token):
        stop_sequences.append("<|im_end|>")

    # Llama3-style tokens
    if "<|eot_id|>" in str(special_tokens):
        stop_sequences.append("<|eot_id|>")

    # Generic markers that should stop generation
    stop_sequences.extend(
        [
            "### USER",  # Don't generate fake user messages
            "### SYSTEM",  # Don't regenerate system prompt
            "<|user|>",  # ChatML user marker
            "<|system|>",  # ChatML system marker
        ]
    )

    return list(set(stop_sequences))  # Remove duplicates
