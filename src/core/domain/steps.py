"""
Step Models for Agent Traces

Defines the step types that make up an agent trace:
- ThoughtStep: Agent's reasoning
- ActionChoiceStep: Decision to call function or provide final answer
- FunctionCallStep: Function invocation
- FunctionOutputStep: Result from function
- FinalAnswerStep: Agent's final response
- InitialPromptStep: Starting prompt
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StepType(str, Enum):
    """Types of steps in an agent trace."""

    THOUGHT = "thought"
    ACTION_CHOICE = "action_choice"
    FUNCTION_CALL = "function_call"
    FUNCTION_OUTPUT = "function_output"
    FINAL_ANSWER = "final_answer"
    INITIAL_PROMPT = "initial_prompt"


class BaseStep(BaseModel):
    """Base class for all step types."""

    type: StepType
    diff: str = Field(description="The raw text difference for this step")


class ThoughtStep(BaseStep):
    """Agent's reasoning step."""

    type: StepType = StepType.THOUGHT
    thought: str = Field(description="The agent's thought content")


class ActionChoiceStep(BaseStep):
    """Decision step: call function or provide final answer."""

    type: StepType = StepType.ACTION_CHOICE
    action_choice: str = Field(description="The chosen action: 'call function' or 'final answer'")


class FunctionCallStep(BaseStep):
    """Function invocation step."""

    type: StepType = StepType.FUNCTION_CALL
    fct_name: str = Field(description="Name of the function being called")
    fct_parameters: str = Field(description="JSON string of function parameters")


class FunctionOutputStep(BaseStep):
    """Result from a function call."""

    type: StepType = StepType.FUNCTION_OUTPUT
    shortuuid: str = Field(description="Unique identifier for this output")
    function_output: str = Field(description="The function's output")


class FinalAnswerStep(BaseStep):
    """Agent's final response to the user."""

    type: StepType = StepType.FINAL_ANSWER
    final_answer: str = Field(description="The final answer content")


class InitialPromptStep(BaseStep):
    """The starting prompt for the agent."""

    type: StepType = StepType.INITIAL_PROMPT


# Type alias for any step
Step = ThoughtStep | ActionChoiceStep | FunctionCallStep | FunctionOutputStep | FinalAnswerStep | InitialPromptStep

# Mapping of step types to their models
STEP_MODEL_MAPPING: dict[StepType, type[BaseStep]] = {
    StepType.THOUGHT: ThoughtStep,
    StepType.ACTION_CHOICE: ActionChoiceStep,
    StepType.FUNCTION_CALL: FunctionCallStep,
    StepType.FUNCTION_OUTPUT: FunctionOutputStep,
    StepType.FINAL_ANSWER: FinalAnswerStep,
    StepType.INITIAL_PROMPT: InitialPromptStep,
}


def create_step(step_type: StepType, **kwargs: Any) -> BaseStep:
    """
    Factory function to create a step model.

    Args:
        step_type: The type of step to create
        **kwargs: Additional fields for the step

    Returns:
        An instance of the appropriate step model

    Raises:
        ValueError: If an unknown step type is provided
    """
    model = STEP_MODEL_MAPPING.get(step_type)
    if not model:
        raise ValueError(f"Unknown step type: {step_type}")
    return model(**kwargs)


# Backwards compatibility alias
create_step_model = create_step
