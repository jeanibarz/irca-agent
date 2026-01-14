"""
Step Factory (Backwards Compatibility)

This module is kept for backwards compatibility.
New code should import from `core.domain` instead.

Example:
    # Old way (still works)
    from core.step_factory import create_step_model, StepType

    # New way (preferred)
    from core.domain import create_step, StepType
"""

# Re-export everything from the new location
from core.domain.steps import (
    STEP_MODEL_MAPPING,
    ActionChoiceStep,
    BaseStep,
    FinalAnswerStep,
    FunctionCallStep,
    FunctionOutputStep,
    InitialPromptStep,
    Step,
    StepType,
    ThoughtStep,
    create_step,
    create_step_model,
)

__all__ = [
    "StepType",
    "BaseStep",
    "ThoughtStep",
    "ActionChoiceStep",
    "FunctionCallStep",
    "FunctionOutputStep",
    "FinalAnswerStep",
    "InitialPromptStep",
    "Step",
    "STEP_MODEL_MAPPING",
    "create_step_model",
    "create_step",
]
