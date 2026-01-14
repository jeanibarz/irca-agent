"""
Domain Models for IRCA Agent

This module contains the core domain models:
- Step: Individual steps in an agent trace
- Trace: Complete agent interaction record
"""

from .steps import (
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
from .trace import Trace

__all__ = [
    # Step types
    "StepType",
    "BaseStep",
    "ThoughtStep",
    "ActionChoiceStep",
    "FunctionCallStep",
    "FunctionOutputStep",
    "FinalAnswerStep",
    "InitialPromptStep",
    "Step",
    # Factory
    "create_step",
    "create_step_model",
    # Trace
    "Trace",
]
