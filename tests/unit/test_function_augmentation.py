import json
import unittest
from unittest.mock import MagicMock

from src.core.domain import StepType, Trace, create_step
from src.core.generation.trace_generator import TraceGenerator
from src.core.utils import shuffle_json_functions


class TestFunctionAugmentation(unittest.TestCase):
    def test_shuffle_json_functions(self):
        """Test that function shuffling preserves all functions but potentially changes order."""
        functions = [
            {"name": "func_a", "description": "do a"},
            {"name": "func_b", "description": "do b"},
            {"name": "func_c", "description": "do c"},
        ]
        functions_json = json.dumps(functions)

        # Run multiple times to ensure we don't accidentally fail if shuffle returns same order (probability exists)
        # But here we mainly validation integrity.
        shuffled_json = shuffle_json_functions(functions_json)
        shuffled_list = json.loads(shuffled_json)

        self.assertEqual(len(shuffled_list), 3)
        # Check all items are present
        names = {f["name"] for f in shuffled_list}
        expected_names = {"func_a", "func_b", "func_c"}
        self.assertEqual(names, expected_names)

    def test_generate_alternative_trace_removes_function(self):
        """Test that _generate_alternative_trace correctly filters the called function from available_functions."""

        # Setup TraceGenerator with a mock model
        generator = TraceGenerator(model=MagicMock())

        # Mock generate_trace_missing_function to just return the filtered functions it received
        # We store the args passed to it to verify them later
        generator.generate_trace_missing_function = MagicMock(return_value="mock_trace_result")

        # Create a dummy main trace
        # Steps: PROMPT -> THOUGHT -> ACTION -> CALL(func_b)
        main_trace = Trace()
        main_trace.steps = [
            create_step(StepType.INITIAL_PROMPT, "prompt"),  # 0
            create_step(StepType.THOUGHT, "thought 1"),  # 1
            create_step(StepType.ACTION_CHOICE, "action 1"),  # 2
            create_step(StepType.FUNCTION_CALL, "func_b"),  # 3 (<-- target)
        ]
        # Set fct_name specifically for the CALL step as the logic logic relies on it
        main_trace.steps[3].fct_name = "func_b"

        available_functions = json.dumps([{"name": "func_a"}, {"name": "func_b"}, {"name": "func_c"}])
        user_query = "help me"

        # Execute
        result = generator._generate_alternative_trace(
            main_trace=main_trace, function_call_index=3, available_functions=available_functions, user_query=user_query
        )

        # Verify
        self.assertEqual(result, "mock_trace_result")

        # Check calls
        generator.generate_trace_missing_function.assert_called_once()
        call_args = generator.generate_trace_missing_function.call_args
        passed_functions_json = call_args[1]["available_functions"]
        # Note: call_args is (args, kwargs). If passed as kwargs:
        if not passed_functions_json:
            passed_functions_json = call_args[0][0]  # assumes positional if not kwarg

        passed_functions = json.loads(passed_functions_json)
        passed_names = [f["name"] for f in passed_functions]

        self.assertNotIn("func_b", passed_names)
        self.assertIn("func_a", passed_names)
        self.assertIn("func_c", passed_names)

    def test_generate_alternative_trace_slicing(self):
        """Test that the trace passed to generate_trace_missing_function is correctly sliced."""
        generator = TraceGenerator(model=MagicMock())
        generator.generate_trace_missing_function = MagicMock()

        # Steps: PROMPT(0) -> THOUGHT(1) -> ACTION(2) -> CALL(3) -> OUTPUT(4) -> THOUGHT(5) -> ACTION(6) -> CALL(7)
        main_trace = Trace()
        steps = [
            create_step(StepType.INITIAL_PROMPT, "prompt"),
            create_step(StepType.THOUGHT, "thought 1"),
            create_step(StepType.ACTION_CHOICE, "action 1"),
            create_step(StepType.FUNCTION_CALL, "call 1"),  # index 3
            create_step(StepType.FUNCTION_OUTPUT, "out 1"),
            create_step(StepType.THOUGHT, "thought 2"),
            create_step(StepType.ACTION_CHOICE, "action 2"),
            create_step(StepType.FUNCTION_CALL, "call 2"),  # index 7
        ]
        steps[3].fct_name = "func_1"
        steps[7].fct_name = "func_2"
        main_trace.steps = steps

        available_functions = json.dumps([{"name": "func_1"}, {"name": "func_2"}])

        # Test slicing for the SECOND call (index 7)
        # Should preserve everything up to index 7-2 = 5 (exclusive), so 0, 1, 2, 3, 4.
        # i.e., It should keep the first full interaction loop, but remove the thought leading to the second call.

        generator._generate_alternative_trace(
            main_trace=main_trace, function_call_index=7, available_functions=available_functions, user_query="query"
        )

        # Verify passed base_trace
        call_args = generator.generate_trace_missing_function.call_args
        base_trace = call_args[1]["base_trace"]

        self.assertEqual(len(base_trace.steps), 5)
        self.assertEqual(base_trace.steps[-1].type, StepType.FUNCTION_OUTPUT)
        self.assertEqual(base_trace.steps[-1].content, "out 1")
