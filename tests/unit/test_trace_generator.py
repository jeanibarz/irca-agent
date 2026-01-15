"""
Tests for Trace Generator

Tests the core orchestration logic of the TraceGenerator.
Covers Requirements:
- FR-GEN-01 (Trace Generation)
- FR-GEN-03 (Function Removal)
"""

import json
import logging
from unittest.mock import patch

import pytest

from core.domain import (
    ActionChoiceStep,
    FinalAnswerStep,
    FunctionCallStep,
    FunctionOutputStep,
    StepType,
    ThoughtStep,
    Trace,
)
from core.generation.trace_generator import TraceGenerator
from tests.mocks import MockGuidanceModel


@pytest.fixture
def mock_model():
    return MockGuidanceModel()


@pytest.fixture
def trace_generator(mock_model):
    # Initialize with the mock model
    tg = TraceGenerator(model=mock_model)
    return tg


class TestTraceGeneratorOrchestration:
    """
    Tests the main generation loop logic.
    Req: FR-GEN-01
    """

    @patch("core.generation.trace_generator.generate_thought")
    @patch("core.generation.trace_generator.generate_action_choice")
    @patch("core.generation.trace_generator.generate_function_call")
    @patch("core.generation.trace_generator.generate_function_output")
    @patch("core.generation.trace_generator.generate_final_answer")
    def test_generate_single_trace_success(
        self, mock_final_answer, mock_fct_output, mock_fct_call, mock_action, mock_thought, trace_generator
    ):
        """
        Test a successful trace generation loop:
        Init -> Thought -> Action(Call) -> Function Call -> Output -> Thought -> Action(Answer) -> Final Answer
        """
        # Setup mocks to simulate the flow

        # 0. Initial Prompt is handled by _generate_initial_prompt (adds 1 step)

        # 1. Thought
        def side_effect_thought(lm, trace, **kwargs):
            trace.append(ThoughtStep(thought="Thinking...", diff="Thinking..."))
            return lm

        mock_thought.side_effect = side_effect_thought

        # 2. Action Choice
        def side_effect_action(lm, trace, **kwargs):
            # If we haven't called a function yet, call one.
            has_called = any(s.type == StepType.FUNCTION_CALL for s in trace.steps)
            if not has_called:
                choice = "call function"  # Matches ACTION_CALL_FUNCTION
            else:
                choice = "final answer"

            trace.append(ActionChoiceStep(action_choice=choice, diff=f"Action: {choice}"))
            return lm

        mock_action.side_effect = side_effect_action

        # 3. Function Call
        def side_effect_call(lm, trace, **kwargs):
            trace.append(FunctionCallStep(fct_name="test_tool", fct_parameters='{"arg": 1}', diff="Call: test_tool"))
            return lm

        mock_fct_call.side_effect = side_effect_call

        # 4. Function Output
        def side_effect_output(lm, trace, **kwargs):
            trace.append(
                FunctionOutputStep(shortuuid="123", function_output='{"result": "success"}', diff="Output: success")
            )
            return lm

        mock_fct_output.side_effect = side_effect_output

        # 5. Final Answer
        def side_effect_answer(lm, trace, **kwargs):
            trace.append(FinalAnswerStep(final_answer="Done", diff="Answer: Done"))
            return lm

        mock_final_answer.side_effect = side_effect_answer

        # Run generation
        trace = trace_generator.generate_single_trace(
            available_functions="[]",  # Empty schema OK for mock
            user_query="Do something",
            max_steps=5,
        )

        assert isinstance(trace, Trace)
        assert len(trace.steps) == 8
        assert trace.steps[2].action_choice == "call function"
        assert trace.steps[3].type == StepType.FUNCTION_CALL
        assert trace.steps[6].action_choice == "final answer"
        assert trace.steps[7].type == StepType.FINAL_ANSWER

    @patch("core.generation.trace_generator.generate_thought")
    @patch("core.generation.trace_generator.generate_action_choice")
    @patch("core.generation.trace_generator.generate_function_call")
    @patch("core.generation.trace_generator.generate_function_output")
    def test_max_steps_reached(self, mock_out, mock_call, mock_action, mock_thought, trace_generator):
        """Test that generation stops if max_steps is reached."""

        # Infinite loop behavior
        mock_thought.side_effect = lambda lm, trace, **kwargs: (
            trace.append(ThoughtStep(thought="...", diff="...")),
            lm,
        )[1]
        mock_action.side_effect = lambda lm, trace, **kwargs: (
            trace.append(ActionChoiceStep(action_choice="call function", diff="...")),
            lm,
        )[1]
        mock_call.side_effect = lambda lm, trace, **kwargs: (
            trace.append(FunctionCallStep(fct_name="x", fct_parameters="{}", diff="...")),
            lm,
        )[1]
        mock_out.side_effect = lambda lm, trace, **kwargs: (
            trace.append(FunctionOutputStep(shortuuid="1", function_output="{}", diff="...")),
            lm,
        )[1]

        with patch("core.generation.trace_generator.generate_final_answer") as mock_final:
            mock_final.side_effect = lambda lm, trace, **kwargs: (
                trace.append(FinalAnswerStep(final_answer="Done", diff="...")),
                lm,
            )[1]

            trace = trace_generator.generate_single_trace(available_functions="[]", user_query="Loop", max_steps=2)

        # Expected steps: 1(Init) + 2(Start) + 4x2(Loops) + 1(Final) = 12
        assert len(trace.steps) == 12


class TestFunctionRemovalAugmentation:
    """
    Tests logic for Function Removal Augmentation.
    Req: FR-GEN-03
    """

    def test_augment_functions_calls_missing_function_generator(self, trace_generator, caplog):
        """
        Verify that generate_traces_with_augmentation correctly identifies the called function
        and calls generate_trace_missing_function with that function removed.
        """
        caplog.set_level(logging.WARNING)

        all_functions = [
            {"name": "func_a", "description": "A"},
            {"name": "func_b", "description": "B"},
        ]
        all_funcs_json = json.dumps(all_functions)

        # Create a canned trace that calls "func_a"
        canned_trace = Trace()
        canned_trace.append(ThoughtStep(thought="...", diff="..."))
        canned_trace.append(ActionChoiceStep(action_choice="call function", diff="..."))
        # Index 2: FunctionCall
        canned_trace.append(FunctionCallStep(fct_name="func_a", fct_parameters="{}", diff="..."))
        canned_trace.append(FunctionOutputStep(shortuuid="1", function_output="{}", diff="..."))

        # Mock generate_single_trace to return our canned trace
        with patch.object(trace_generator, "generate_single_trace") as mock_gen_single:
            mock_gen_single.return_value = canned_trace

            # Use logger patch just to be safe if debugging needs (optional)
            with patch("core.generation.trace_generator.logger"):
                # Mock generate_trace_missing_function to verify it's called
                with patch.object(trace_generator, "generate_trace_missing_function") as mock_gen_missing:
                    # Return a trace with steps so it evaluates to True (len > 0)
                    valid_trace = Trace()
                    valid_trace.append(ThoughtStep(thought="dummy", diff="dummy"))
                    mock_gen_missing.return_value = valid_trace

                    result = trace_generator.generate_traces_with_augmentation(
                        available_functions=all_funcs_json, user_query="Query"
                    )

                    # Check results
                    assert len(result) == 2, f"Expected 2 traces, got {len(result)}"

                    # Verify generate_trace_missing_function was called with "func_a" removed
                    mock_gen_missing.assert_called_once()
                    call_args = mock_gen_missing.call_args

                    # extract passed functions
                    passed_funcs_json = call_args.kwargs.get("available_functions") or call_args[0]
                    passed_funcs = json.loads(passed_funcs_json)

                    # func_a should be gone, func_b remains
                    names = [f["name"] for f in passed_funcs]
                    assert "func_a" not in names
                    assert "func_b" in names
