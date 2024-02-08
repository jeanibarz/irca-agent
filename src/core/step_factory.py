from enum import Enum
from typing import Type, Dict
from pydantic import BaseModel


# Define an enum for step types
class StepType(Enum):
    THOUGHT = "thought"
    ACTION_CHOICE = "action_choice"
    FUNCTION_CALL = "function_call"
    FUNCTION_OUTPUT = "function_output"
    FINAL_ANSWER = "final_answer"
    INITIAL_PROMPT = "initial_prompt"


# Define each step type as a Pydantic model
class ThoughtStep(BaseModel):
    type: StepType = StepType.THOUGHT
    thought: str
    diff: str


class ActionChoiceStep(BaseModel):
    type: StepType = StepType.ACTION_CHOICE
    action_choice: str
    diff: str


class FunctionCallStep(BaseModel):
    type: StepType = StepType.FUNCTION_CALL
    fct_name: str
    fct_parameters: str
    diff: str


class FunctionOutputStep(BaseModel):
    type: StepType = StepType.FUNCTION_OUTPUT
    shortuuid: str
    function_output: str
    diff: str


class FinalAnswerStep(BaseModel):
    type: StepType = StepType.FINAL_ANSWER
    final_answer: str
    diff: str


class InitialPromptStep(BaseModel):
    type: StepType = StepType.INITIAL_PROMPT
    diff: str


# Mapping of step types to their corresponding Pydantic models
STEP_MODEL_MAPPING: Dict[StepType, Type[BaseModel]] = {
    StepType.THOUGHT: ThoughtStep,
    StepType.ACTION_CHOICE: ActionChoiceStep,
    StepType.FUNCTION_CALL: FunctionCallStep,
    StepType.FUNCTION_OUTPUT: FunctionOutputStep,
    StepType.FINAL_ANSWER: FinalAnswerStep,
    StepType.INITIAL_PROMPT: InitialPromptStep,
}


def create_step_model(step_type: StepType, **kwargs) -> BaseModel:
    """
    Factory function to create a step model based on the given step type.

    Args:
        step_type (StepType): The type of step to create.
        **kwargs: Additional keyword arguments to pass to the model constructor.

    Returns:
        BaseModel: An instance of the corresponding step model.

    Raises:
        ValueError: If an unknown step type is provided.
    """
    model = STEP_MODEL_MAPPING.get(step_type)
    if not model:
        raise ValueError(f"Unknown step type: {step_type}")
    return model(**kwargs)
