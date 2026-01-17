"""
Step Generators

Functions for generating individual steps in an agent trace.
Each generator handles one type of step (thought, action choice, function call, etc.)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import shortuuid
from guidance import gen, select

from src.core.domain import StepType, Trace, create_step

from .constants import (
    ACTION_CALL_FUNCTION,
    ACTION_CHOICE_PROMPT,
    ACTION_FINAL_ANSWER,
    CALL_FUNCTION_PROMPT,
    FINAL_ANSWER_PROMPT,
    FINAL_ANSWER_STOP,
    FUNCTION_OUTPUT_PROMPT,
    STOP_SEQUENCES,
    THOUGHT_PROMPT,
)

if TYPE_CHECKING:
    from guidance import Program

logger = logging.getLogger(__name__)


def generate_thought(
    lm: Program,
    trace: Trace,
    prefix: str = "",
    suffix: str = "",
    temperature: float = 0.25,
    max_tokens: int = 500,
) -> Program:
    """
    Generate a thought step.

    Args:
        lm: The language model program
        trace: The trace to append to
        prefix: Text to prepend
        suffix: Text to append
        temperature: Generation temperature
        max_tokens: Maximum tokens to generate

    Returns:
        Updated language model program
    """
    try:
        lm += (
            prefix
            + THOUGHT_PROMPT
            + gen(
                max_tokens=max_tokens,
                name="thought",
                stop=STOP_SEQUENCES,
                temperature=temperature,
            )
            + suffix
        )
        step = create_step(
            step_type=StepType.THOUGHT,
            thought=lm["thought"],
            diff=prefix + THOUGHT_PROMPT + lm["thought"] + suffix,
        )
        trace.append(step)
        return lm
    except Exception as e:
        logger.error(f"Error generating thought: {e}")
        return lm


def generate_thought_missing_function(
    lm: Program,
    trace: Trace,
    prefix: str = "",
    suffix: str = "",
) -> Program:
    """
    Generate a forced thought when the required function is missing.

    Used for data augmentation - teaches the model to acknowledge
    when it cannot complete a task.
    """
    forced_thought = (
        "I can't find any function that could be helpful to answer user query. "
        "I need to abort the Iterative Resolution Cycle and return a final answer."
    )
    lm += prefix + forced_thought + suffix
    step = create_step(
        step_type=StepType.THOUGHT,
        thought=forced_thought,
        diff=prefix + THOUGHT_PROMPT + forced_thought + suffix,
    )
    trace.append(step)
    return lm


def generate_action_choice(
    lm: Program,
    trace: Trace,
    prefix: str = "",
    suffix: str = "",
) -> Program:
    """
    Generate an action choice step.

    The model chooses between calling a function or providing a final answer.
    """
    lm += (
        prefix
        + ACTION_CHOICE_PROMPT
        + select(
            options=[ACTION_CALL_FUNCTION, ACTION_FINAL_ANSWER],
            name="action_choice",
        )
        + suffix
    )
    step = create_step(
        step_type=StepType.ACTION_CHOICE,
        action_choice=lm["action_choice"],
        diff=prefix + ACTION_CHOICE_PROMPT + lm["action_choice"] + suffix,
    )
    trace.append(step)
    return lm


def generate_function_call(
    lm: Program,
    trace: Trace,
    prefix: str = "",
    suffix: str = "<|wait|>",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> Program:
    """
    Generate a function call step.

    The model generates a function name and parameters.
    """
    lm += (
        prefix
        + CALL_FUNCTION_PROMPT
        + gen("fct_name", stop='"')
        + '"}, "parameters": '
        + gen(
            max_tokens=max_tokens,
            name="fct_parameters",
            stop=STOP_SEQUENCES,
            temperature=temperature,
        )
        + suffix
    )
    step = create_step(
        step_type=StepType.FUNCTION_CALL,
        fct_name=lm["fct_name"],
        fct_parameters=lm["fct_parameters"],
        diff=prefix + CALL_FUNCTION_PROMPT + lm["fct_name"] + '"}, "parameters": ' + lm["fct_parameters"] + suffix,
    )
    trace.append(step)
    return lm


def generate_function_output(
    lm: Program,
    trace: Trace,
    prefix: str = "",
    suffix: str = "",
    temperature: float = 1.0,
    max_tokens: int = 500,
) -> Program:
    """
    Generate a synthetic function output step.

    The model generates a plausible output for the function call.
    """
    output_uuid = shortuuid.uuid()
    lm += (
        prefix
        + FUNCTION_OUTPUT_PROMPT.format(shortuuid=output_uuid)
        + gen(
            max_tokens=max_tokens,
            name="function_output",
            stop="\n",
            temperature=temperature,
        )
        + suffix
    )
    step = create_step(
        step_type=StepType.FUNCTION_OUTPUT,
        shortuuid=output_uuid,
        function_output=lm["function_output"],
        diff=prefix + FUNCTION_OUTPUT_PROMPT.format(shortuuid=output_uuid) + lm["function_output"] + suffix,
    )
    trace.append(step)
    return lm


def generate_final_answer(
    lm: Program,
    trace: Trace,
    prefix: str = FINAL_ANSWER_PROMPT,
    suffix: str = "<|wait|>",
    temperature: float = 0.5,
    max_tokens: int = 500,
) -> Program:
    """
    Generate the final answer step.

    This concludes the agent's trace with a response to the user.
    """
    lm += (
        prefix
        + gen(
            max_tokens=max_tokens,
            name="final_answer",
            temperature=temperature,
            stop=FINAL_ANSWER_STOP,
        )
        + suffix
    )
    step = create_step(
        step_type=StepType.FINAL_ANSWER,
        final_answer=lm["final_answer"].rstrip("\n"),
        diff=prefix + lm["final_answer"].rstrip("\n") + suffix,
    )
    trace.append(step)
    return lm
