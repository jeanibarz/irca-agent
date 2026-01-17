"""
IRCA Agent Core Module

This module contains the core functionality for generating agent traces:

- `domain`: Step and Trace models
- `generation`: Trace generation with constrained LLM
- `prompt_builder`: Prompt construction and parsing
- `utils`: Utility functions
"""

# Re-export commonly used items from domain (always available)
from .domain import (
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

# Backwards compatibility - import from step_factory still works
from .step_factory import STEP_MODEL_MAPPING, create_step_model

# Generation modules require additional dependencies (guidance, shortuuid, etc.)
# Import them lazily to allow basic usage without all deps
try:
    from .generation import GuidedTraceGenerator, TraceGenerator

    _GENERATION_AVAILABLE = True
except ImportError:
    TraceGenerator = None
    GuidedTraceGenerator = None
    _GENERATION_AVAILABLE = False

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
    # Backwards compatibility
    "create_step_model",
    "STEP_MODEL_MAPPING",
]

# Only include generation if available
if _GENERATION_AVAILABLE:
    __all__.extend(["TraceGenerator", "GuidedTraceGenerator"])
