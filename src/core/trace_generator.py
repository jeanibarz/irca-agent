"""
Trace Generator (Backwards Compatibility)

This module is kept for backwards compatibility.
New code should import from `core.generation` instead.

Example:
    # Old way (still works)
    from src.core.trace_generator import GuidedTraceGenerator

    # New way (preferred)
    from src.core.generation import TraceGenerator
"""

# Re-export everything from the new location
from src.core.generation import (
    ACTION_CALL_FUNCTION,
    ACTION_CHOICE_PROMPT,
    ACTION_FINAL_ANSWER,
    CALL_FUNCTION_PROMPT,
    FINAL_ANSWER_PROMPT,
    FUNCTION_OUTPUT_PROMPT,
    THOUGHT_PROMPT,
    GuidedTraceGenerator,
    TraceGenerator,
    generate_action_choice,
    generate_final_answer,
    generate_function_call,
    generate_function_output,
    generate_thought,
    generate_thought_missing_function,
    trace_to_argilla_record,
)

__all__ = [
    "TraceGenerator",
    "GuidedTraceGenerator",
    "generate_thought",
    "generate_thought_missing_function",
    "generate_action_choice",
    "generate_function_call",
    "generate_function_output",
    "generate_final_answer",
    "trace_to_argilla_record",
    "THOUGHT_PROMPT",
    "ACTION_CHOICE_PROMPT",
    "CALL_FUNCTION_PROMPT",
    "FUNCTION_OUTPUT_PROMPT",
    "FINAL_ANSWER_PROMPT",
    "ACTION_CALL_FUNCTION",
    "ACTION_FINAL_ANSWER",
]
