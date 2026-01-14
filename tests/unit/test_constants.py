"""
Tests for Generation Constants

Tests the constants defined in core.generation.constants
"""

from core.generation.constants import (
    ACTION_CALL_FUNCTION,
    ACTION_CHOICE_PROMPT,
    ACTION_FINAL_ANSWER,
    CALL_FUNCTION_PROMPT,
    FINAL_ANSWER_PROMPT,
    FINAL_ANSWER_STOP,
    FUNCTION_OUTPUT_PROMPT,
    STOP_SEQUENCES,
    THOUGHT_PROMPT,
)


class TestPromptConstants:
    """Tests for prompt constant values."""

    def test_thought_prompt_format(self):
        """THOUGHT_PROMPT has expected format."""
        assert THOUGHT_PROMPT == "Thought: "
        assert THOUGHT_PROMPT.endswith(" ")

    def test_action_choice_prompt_format(self):
        """ACTION_CHOICE_PROMPT has expected format."""
        assert ACTION_CHOICE_PROMPT == "Action choice: "

    def test_call_function_prompt_format(self):
        """CALL_FUNCTION_PROMPT starts JSON object."""
        assert CALL_FUNCTION_PROMPT.startswith('Call function: {"name": "')
        assert CALL_FUNCTION_PROMPT.endswith('"')

    def test_function_output_prompt_has_placeholder(self):
        """FUNCTION_OUTPUT_PROMPT contains shortuuid placeholder."""
        assert "{shortuuid}" in FUNCTION_OUTPUT_PROMPT
        # Can be formatted
        formatted = FUNCTION_OUTPUT_PROMPT.format(shortuuid="abc123")
        assert "abc123" in formatted

    def test_final_answer_prompt_format(self):
        """FINAL_ANSWER_PROMPT has expected format."""
        assert "### FINAL ANSWER" in FINAL_ANSWER_PROMPT
        assert FINAL_ANSWER_PROMPT.startswith("\n")


class TestActionConstants:
    """Tests for action choice constants."""

    def test_action_call_function_value(self):
        """ACTION_CALL_FUNCTION has expected value."""
        assert ACTION_CALL_FUNCTION == "call function"

    def test_action_final_answer_value(self):
        """ACTION_FINAL_ANSWER has expected value."""
        assert ACTION_FINAL_ANSWER == "final answer"

    def test_actions_are_distinct(self):
        """Action constants have different values."""
        assert ACTION_CALL_FUNCTION != ACTION_FINAL_ANSWER


class TestStopSequences:
    """Tests for stop sequence constants."""

    def test_stop_sequences_is_list(self):
        """STOP_SEQUENCES is a list."""
        assert isinstance(STOP_SEQUENCES, list)
        assert len(STOP_SEQUENCES) > 0

    def test_stop_sequences_contains_newline(self):
        """STOP_SEQUENCES includes newline."""
        assert "\n" in STOP_SEQUENCES

    def test_stop_sequences_contains_wait(self):
        """STOP_SEQUENCES includes wait token."""
        assert "<|wait|>" in STOP_SEQUENCES

    def test_final_answer_stop_is_list(self):
        """FINAL_ANSWER_STOP is a list."""
        assert isinstance(FINAL_ANSWER_STOP, list)
        assert len(FINAL_ANSWER_STOP) > 0

    def test_final_answer_stop_contains_headers(self):
        """FINAL_ANSWER_STOP includes section headers."""
        assert "### INSTRUCTIONS" in FINAL_ANSWER_STOP
        assert "### USER QUERY" in FINAL_ANSWER_STOP
