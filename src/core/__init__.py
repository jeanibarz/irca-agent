"""
IRCA Agent Core Module

This module contains the core functionality for generating agent traces:

- `domain`: Step and Trace models
- `generation`: Trace generation with constrained LLM
- `prompt_builder`: Prompt construction and parsing
- `utils`: Utility functions
"""

# Re-export commonly used items
from core.domain import (
    ActionChoiceStep,
    BaseStep,
    FinalAnswerStep,
    FunctionCallStep,
    FunctionOutputStep,
    InitialPromptStep,
    Step,
    StepType,
    ThoughtStep,
    Trace,
    create_step,
)
from core.generation import GuidedTraceGenerator, TraceGenerator

# Backwards compatibility - import from step_factory still works
from core.step_factory import STEP_MODEL_MAPPING, create_step_model

__all__ = [
    # Domain models
    "StepType",
    "BaseStep",
    "Step",
    "ThoughtStep",
    "ActionChoiceStep",
    "FunctionCallStep",
    "FunctionOutputStep",
    "FinalAnswerStep",
    "InitialPromptStep",
    "Trace",
    "create_step",
    # Generation
    "TraceGenerator",
    "GuidedTraceGenerator",
    # Backwards compatibility
    "create_step_model",
    "STEP_MODEL_MAPPING",
]
