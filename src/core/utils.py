"""
Utility Functions

Common utilities used across the codebase.
"""

from __future__ import annotations

import json
import random
from typing import Any


def extract_and_remove(
    start_marker: str | None,
    end_marker: str | None,
    full_prompt: str,
    include_start_marker: bool = False,
    include_end_marker: bool = False,
) -> tuple[str, str]:
    """
    Extract a section from a prompt and return the extracted section with the remaining text.

    Args:
        start_marker: Starting marker for extraction (None for beginning)
        end_marker: Ending marker for extraction (None for end of string)
        full_prompt: The full prompt to extract from
        include_start_marker: Whether to include the start marker in extraction
        include_end_marker: Whether to include the end marker in extraction

    Returns:
        Tuple of (extracted_section, remaining_prompt)
    """
    # Find start position
    if start_marker is None:
        start_idx = 0
    else:
        start_idx = full_prompt.find(start_marker)
        if start_idx == -1:
            return "", full_prompt

    # Determine extraction start
    if include_start_marker or start_marker is None:
        start_extract_idx = start_idx
    else:
        start_extract_idx = start_idx + len(start_marker)

    # Find end position
    if end_marker is None:
        end_idx = len(full_prompt)
    else:
        end_idx = full_prompt.find(end_marker, start_idx)
        if end_idx == -1:
            extracted_section = full_prompt[start_extract_idx:].strip()
            remaining = full_prompt[:start_idx].strip()
            return extracted_section, remaining

    # Determine extraction end
    if include_end_marker and end_marker is not None:
        end_extract_idx = end_idx + len(end_marker)
    else:
        end_extract_idx = end_idx

    # Extract and reconstruct
    extracted_section = full_prompt[start_extract_idx:end_extract_idx].strip()
    remaining = full_prompt[:start_idx].strip() + full_prompt[end_extract_idx:].strip()

    return extracted_section, remaining


def print_trainable_parameters(model: Any) -> None:
    """
    Print the number of trainable parameters in a model.

    Args:
        model: PyTorch model with parameters
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable_params / all_params if all_params > 0 else 0

    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_params:,} || "
        f"trainable%: {percentage:.2f}"
    )


def format_generate_user_query(
    available_functions: str,
    user_query: str | None = None,
) -> str:
    """
    Format a prompt for generating user queries.

    Used to generate synthetic user queries based on available functions.

    Args:
        available_functions: JSON string of available functions
        user_query: Optional existing query (for training)

    Returns:
        Formatted prompt string
    """
    text = (
        "The following functions are available to an AI ASSISTANT:\n"
        + available_functions
        + "\n\n"
        + "Please generate a potential user query for the AI ASSISTANT. "
        + "Be very creative ! The query should be natural, and look like spontaneous. "
        + "Feel free to imagine any kind of interesting scenarios you can think of. "
        + "You can also add some typos some times or make some small grammar errors.\n"
        + "Sure ! Here's a potential query that a user could make to the AI ASSISTANT "
        + "given the available functions described above: "
    )

    if user_query:
        # For training, include the expected query
        text += user_query + "</s>"

    return text


def shuffle_json_functions(available_functions: str) -> str:
    """
    Shuffle the order of functions in a JSON string.

    Used for data augmentation to reduce positional bias.

    Args:
        available_functions: JSON string of function list

    Returns:
        JSON string with shuffled functions
    """
    functions = json.loads(available_functions)
    random.shuffle(functions)
    return json.dumps(functions)


def get_trainable_param_count(model: Any) -> tuple[int, int]:
    """
    Get trainable and total parameter counts.

    Args:
        model: PyTorch model

    Returns:
        Tuple of (trainable_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
