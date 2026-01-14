"""
Generation Module

This module provides trace generation capabilities:
- TraceGenerator: Main class for generating agent traces
- Step generators: Functions for generating individual steps
- Argilla integration: Converting traces to Argilla records
"""

from .argilla import trace_to_argilla_record
from .constants import (
    ACTION_CALL_FUNCTION,
    ACTION_CHOICE_PROMPT,
    ACTION_FINAL_ANSWER,
    CALL_FUNCTION_PROMPT,
    FINAL_ANSWER_PROMPT,
    FUNCTION_OUTPUT_PROMPT,
    THOUGHT_PROMPT,
)
from .step_generators import (
    generate_action_choice,
    generate_final_answer,
    generate_function_call,
    generate_function_output,
    generate_thought,
    generate_thought_missing_function,
)
from .trace_generator import GuidedTraceGenerator, TraceGenerator

__all__ = [
    # Main generator
    "TraceGenerator",
    "GuidedTraceGenerator",  # Backwards compatibility
    # Step generators
    "generate_thought",
    "generate_thought_missing_function",
    "generate_action_choice",
    "generate_function_call",
    "generate_function_output",
    "generate_final_answer",
    # Argilla
    "trace_to_argilla_record",
    # Constants
    "THOUGHT_PROMPT",
    "ACTION_CHOICE_PROMPT",
    "CALL_FUNCTION_PROMPT",
    "FUNCTION_OUTPUT_PROMPT",
    "FINAL_ANSWER_PROMPT",
    "ACTION_CALL_FUNCTION",
    "ACTION_FINAL_ANSWER",
]
