"""
Prompt Builder

Utilities for constructing and manipulating prompts.
Includes functions for:
- Building full prompts from components
- Parsing prompts to extract components
- Data augmentation through formatting randomization
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from src.core.utils import extract_and_remove

logger = logging.getLogger(__name__)


def build_full_prompt(sample: dict[str, Any]) -> str:
    """
    Construct a full prompt string from components.

    Args:
        sample: Dictionary with prompt components:
            - system_instructions: Instructions for the agent
            - example: Example interaction
            - available_functions_json: JSON list of functions
            - user_query: The user's question
            - assistant_completion: Agent's response

    Returns:
        Complete formatted prompt string
    """
    system_instructions = sample.get("system_instructions", "")
    example = sample.get("example", "")
    available_functions_json = sample.get("available_functions_json", [])
    user_query = sample.get("user_query", "")
    assistant_completion = sample.get("assistant_completion", "")

    full_prompt = f"""
### INSTRUCTIONS
{system_instructions}

EXAMPLE:
{example}

The functions available to you are described below.

### FUNCTIONS AVAILABLE
{available_functions_json}

### USER QUERY
{user_query}

### ITERATIVE RESOLUTION CYCLE
{assistant_completion}
"""
    return full_prompt


def parse_corrected_agent_trace(full_prompt: str) -> dict[str, str]:
    """
    Parse a full prompt string to extract its components.

    Uses markers to separate the prompt into:
    - system_instructions
    - example
    - available_functions_json
    - user_query
    - assistant_completion

    Args:
        full_prompt: Complete prompt string to parse

    Returns:
        Dictionary with parsed components
    """
    # Extract each section using markers
    system_instructions, full_prompt = extract_and_remove(
        start_marker="### INSTRUCTIONS",
        end_marker="EXAMPLE:",
        full_prompt=full_prompt,
        include_start_marker=False,
        include_end_marker=False,
    )

    example, full_prompt = extract_and_remove(
        start_marker="EXAMPLE:",
        end_marker="The functions available to you are described below.",
        full_prompt=full_prompt,
        include_start_marker=False,
        include_end_marker=False,
    )

    available_functions_json, full_prompt = extract_and_remove(
        start_marker="### FUNCTIONS AVAILABLE",
        end_marker="### USER QUERY",
        full_prompt=full_prompt,
        include_start_marker=False,
        include_end_marker=False,
    )

    user_query, full_prompt = extract_and_remove(
        start_marker="### USER QUERY",
        end_marker="### ITERATIVE RESOLUTION CYCLE",
        full_prompt=full_prompt,
        include_start_marker=False,
        include_end_marker=False,
    )

    assistant_completion, _ = extract_and_remove(
        start_marker="### ITERATIVE RESOLUTION CYCLE",
        end_marker=None,
        full_prompt=full_prompt,
        include_start_marker=False,
        include_end_marker=False,
    )

    return {
        "system_instructions": system_instructions,
        "example": example,
        "available_functions_json": available_functions_json,
        "user_query": user_query,
        "assistant_completion": assistant_completion,
    }


def randomize_newline_characters(text: str) -> str:
    """
    Randomize newline characters for data augmentation.

    Replaces all newlines with either \n or \r\n randomly
    to improve model robustness to formatting variations.

    Args:
        text: Input text

    Returns:
        Text with randomized newlines
    """
    newline_choice = random.choice(["\n", "\r\n"])
    return text.replace("\n", newline_choice)


def randomize_system_instructions_formatting(system_instructions: str) -> str:
    """
    Randomize system instruction formatting for data augmentation.

    Randomly alters formatting by:
    1. Optionally replacing markers with alternatives
    2. Adding random newline variations

    Args:
        system_instructions: Original instructions

    Returns:
        Instructions with randomized formatting
    """
    # Randomly choose a formatting style
    format_style = random.choice([1, 2])

    if format_style == 2:
        replacements = {
            "### INSTRUCTIONS": "",
            "### FUNCTIONS AVAILABLE": "<|FUNCTIONS AVAILABLE|>",
            "### USER QUERY": "<|USER QUERY|>",
        }
        for old, new in replacements.items():
            newline_count = random.choice(["", "\n", "\n\n"])
            system_instructions = system_instructions.replace(old, newline_count + new)

    # Randomize newline characters
    system_instructions = randomize_newline_characters(system_instructions)

    return system_instructions


class InstructionFormatter:
    """
    Formats instructions for training with optional augmentation.

    Tracks iteration count for logging purposes.
    """

    def __init__(self, random_augmentation: bool = True):
        """
        Initialize the formatter.

        Args:
            random_augmentation: Whether to apply random formatting variations
        """
        self.random_augmentation = random_augmentation
        self.iteration_count = 0

    def format(self, sample: dict[str, Any]) -> str:
        """
        Format a sample for training.

        Args:
            sample: Sample containing 'corrected_agent_trace'

        Returns:
            Formatted instruction string
        """
        self.iteration_count += 1
        logger.debug(f"Formatting sample {self.iteration_count}")

        try:
            full_prompt = sample["corrected_agent_trace"][0]["value"]
            full_prompt = full_prompt.replace("\r\n", "\n")
        except (TypeError, KeyError, IndexError) as e:
            logger.error(f"Error accessing 'corrected_agent_trace' in sample: {e}")
            logger.error(f"Sample type: {type(sample)}")
            logger.error(f"Sample: {sample}")
            raise e

        parsed_data = parse_corrected_agent_trace(full_prompt)

        if self.random_augmentation:
            parsed_data["system_instructions"] = randomize_system_instructions_formatting(
                parsed_data["system_instructions"]
            )

        available_functions_json = parsed_data.get("available_functions_json", "")
        if available_functions_json:
            available_functions = json.loads(available_functions_json)
            indent = None

            if self.random_augmentation:
                random.shuffle(available_functions)
                indent = random.choice([None, 3, 4])

            parsed_data["available_functions_json"] = json.dumps(available_functions, indent=indent)

        return build_full_prompt(parsed_data)


# Default formatter instance for backwards compatibility
_default_formatter = InstructionFormatter(random_augmentation=True)


def format_instruction(sample: dict[str, Any], random_augmentation: bool = True) -> str:
    """
    Format a sample instruction with optional augmentation.

    This is the main function used during training to format samples.

    Args:
        sample: Sample containing 'corrected_agent_trace'
        random_augmentation: Whether to apply random formatting

    Returns:
        Formatted instruction string
    """
    global _default_formatter

    # Create new formatter if augmentation setting differs
    if _default_formatter.random_augmentation != random_augmentation:
        _default_formatter = InstructionFormatter(random_augmentation=random_augmentation)

    return _default_formatter.format(sample)
