"""
Tests for Step Models

Tests the step types and factory function in core.domain.steps
"""

import pytest

from core.domain import (
    ActionChoiceStep,
    FinalAnswerStep,
    FunctionCallStep,
    FunctionOutputStep,
    InitialPromptStep,
    StepType,
    ThoughtStep,
    Trace,
    create_step,
)


class TestStepType:
    """Tests for StepType enum."""

    def test_step_type_values(self):
        """Verify all step types have correct string values."""
        assert StepType.THOUGHT.value == "thought"
        assert StepType.ACTION_CHOICE.value == "action_choice"
        assert StepType.FUNCTION_CALL.value == "function_call"
        assert StepType.FUNCTION_OUTPUT.value == "function_output"
        assert StepType.FINAL_ANSWER.value == "final_answer"
        assert StepType.INITIAL_PROMPT.value == "initial_prompt"

    def test_step_type_is_string_enum(self):
        """StepType should be usable as a string."""
        assert str(StepType.THOUGHT) == "StepType.THOUGHT"
        assert StepType.THOUGHT == "thought"


class TestThoughtStep:
    """Tests for ThoughtStep model."""

    def test_create_thought_step(self):
        """Can create a valid ThoughtStep."""
        step = ThoughtStep(
            thought="I need to help the user",
            diff="Thought: I need to help the user",
        )
        assert step.type == StepType.THOUGHT
        assert step.thought == "I need to help the user"
        assert step.diff == "Thought: I need to help the user"

    def test_thought_step_serialization(self):
        """ThoughtStep can be serialized to dict."""
        step = ThoughtStep(thought="Test thought", diff="Thought: Test thought")
        data = step.model_dump()
        assert data["type"] == "thought"
        assert data["thought"] == "Test thought"


class TestActionChoiceStep:
    """Tests for ActionChoiceStep model."""

    def test_create_action_choice_call_function(self):
        """Can create action choice for calling a function."""
        step = ActionChoiceStep(
            action_choice="call function",
            diff="Action choice: call function",
        )
        assert step.type == StepType.ACTION_CHOICE
        assert step.action_choice == "call function"

    def test_create_action_choice_final_answer(self):
        """Can create action choice for final answer."""
        step = ActionChoiceStep(
            action_choice="final answer",
            diff="Action choice: final answer",
        )
        assert step.action_choice == "final answer"


class TestFunctionCallStep:
    """Tests for FunctionCallStep model."""

    def test_create_function_call_step(self):
        """Can create a valid FunctionCallStep."""
        step = FunctionCallStep(
            fct_name="get_weather",
            fct_parameters='{"location": "Paris"}',
            diff='Call function: {"name": "get_weather"}, "parameters": {"location": "Paris"}',
        )
        assert step.type == StepType.FUNCTION_CALL
        assert step.fct_name == "get_weather"
        assert step.fct_parameters == '{"location": "Paris"}'


class TestFunctionOutputStep:
    """Tests for FunctionOutputStep model."""

    def test_create_function_output_step(self):
        """Can create a valid FunctionOutputStep."""
        step = FunctionOutputStep(
            shortuuid="abc123",
            function_output='{"temperature": 20}',
            diff='Output[abc123]: {"temperature": 20}',
        )
        assert step.type == StepType.FUNCTION_OUTPUT
        assert step.shortuuid == "abc123"
        assert step.function_output == '{"temperature": 20}'


class TestFinalAnswerStep:
    """Tests for FinalAnswerStep model."""

    def test_create_final_answer_step(self):
        """Can create a valid FinalAnswerStep."""
        step = FinalAnswerStep(
            final_answer="The weather in Paris is 20 degrees.",
            diff="\n\n### FINAL ANSWER\nThe weather in Paris is 20 degrees.",
        )
        assert step.type == StepType.FINAL_ANSWER
        assert step.final_answer == "The weather in Paris is 20 degrees."


class TestInitialPromptStep:
    """Tests for InitialPromptStep model."""

    def test_create_initial_prompt_step(self):
        """Can create a valid InitialPromptStep."""
        step = InitialPromptStep(diff="### INSTRUCTIONS\nYou are an AI assistant.")
        assert step.type == StepType.INITIAL_PROMPT


class TestCreateStep:
    """Tests for create_step factory function."""

    def test_create_thought_step(self):
        """Factory creates correct ThoughtStep."""
        step = create_step(
            step_type=StepType.THOUGHT,
            thought="Test thought",
            diff="Thought: Test thought",
        )
        assert isinstance(step, ThoughtStep)
        assert step.thought == "Test thought"

    def test_create_action_choice_step(self):
        """Factory creates correct ActionChoiceStep."""
        step = create_step(
            step_type=StepType.ACTION_CHOICE,
            action_choice="call function",
            diff="Action choice: call function",
        )
        assert isinstance(step, ActionChoiceStep)

    def test_create_function_call_step(self):
        """Factory creates correct FunctionCallStep."""
        step = create_step(
            step_type=StepType.FUNCTION_CALL,
            fct_name="test_func",
            fct_parameters="{}",
            diff="Call function: ...",
        )
        assert isinstance(step, FunctionCallStep)

    def test_create_unknown_step_type_raises(self):
        """Factory raises ValueError for unknown step type."""
        with pytest.raises(ValueError, match="Unknown step type"):
            create_step(step_type="invalid_type", diff="test")


class TestTrace:
    """Tests for Trace model."""

    def test_create_empty_trace(self):
        """Can create an empty trace."""
        trace = Trace()
        assert len(trace) == 0
        assert trace.last_step is None

    def test_append_step_to_trace(self):
        """Can append steps to a trace."""
        trace = Trace()
        step = ThoughtStep(thought="Test", diff="Thought: Test")
        trace.append(step)
        assert len(trace) == 1
        assert trace.last_step == step

    def test_trace_to_string(self):
        """Trace converts to string by concatenating diffs."""
        trace = Trace()
        trace.append(ThoughtStep(thought="Test 1", diff="Thought: Test 1"))
        trace.append(ActionChoiceStep(action_choice="final answer", diff="\nAction choice: final answer"))
        result = trace.to_string()
        assert result == "Thought: Test 1\nAction choice: final answer"

    def test_trace_indexing(self):
        """Can access trace steps by index."""
        trace = Trace()
        step1 = ThoughtStep(thought="First", diff="1")
        step2 = ActionChoiceStep(action_choice="call function", diff="2")
        trace.append(step1)
        trace.append(step2)
        assert trace[0] == step1
        assert trace[1] == step2

    def test_trace_iteration(self):
        """Can iterate over trace steps."""
        trace = Trace()
        steps = [
            ThoughtStep(thought="1", diff="1"),
            ActionChoiceStep(action_choice="call function", diff="2"),
        ]
        for step in steps:
            trace.append(step)

        collected = list(trace)
        assert collected == steps
