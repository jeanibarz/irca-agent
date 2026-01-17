"""
Unit tests for formatting augmentation steps.

Tests FR-DATA-25: Staged Augmentation.
"""

import random

import pytest

from src.formatting.config import StepConfig
from src.formatting.steps import (
    IndentFunctionsStep,
    NewlineVariationStep,
    ShuffleFunctionsStep,
    create_step,
    get_step_stage,
)


class TestShuffleFunctionsStep:
    """Tests for shuffle_functions step."""

    def test_shuffles_functions(self):
        """Functions are shuffled."""
        step = ShuffleFunctionsStep(
            StepConfig(type="shuffle_functions"),
            random.Random(42),
        )

        data = {
            "available_functions": [
                {"name": "func_a"},
                {"name": "func_b"},
                {"name": "func_c"},
            ]
        }

        result = step.apply(data)

        # Functions should be shuffled (not in original order)
        names = [f["name"] for f in result["available_functions"]]
        assert set(names) == {"func_a", "func_b", "func_c"}
        # With seed 42, should produce consistent result
        assert names != ["func_a", "func_b", "func_c"]

    def test_empty_functions(self):
        """Handles empty function list."""
        step = ShuffleFunctionsStep(
            StepConfig(type="shuffle_functions"),
            random.Random(42),
        )

        data = {"available_functions": []}
        result = step.apply(data)
        assert result["available_functions"] == []

    def test_preserves_other_data(self):
        """Other data is preserved."""
        step = ShuffleFunctionsStep(
            StepConfig(type="shuffle_functions"),
            random.Random(42),
        )

        data = {
            "available_functions": [{"name": "func"}],
            "user_query": "test query",
        }

        result = step.apply(data)
        assert result["user_query"] == "test query"

    def test_stage_is_formatting(self):
        """Step stage is formatting."""
        assert ShuffleFunctionsStep.stage == "formatting"

    def test_get_name(self):
        """get_name returns correct name."""
        step = ShuffleFunctionsStep(
            StepConfig(type="shuffle_functions"),
            random.Random(42),
        )
        data = {"available_functions": []}
        step.apply(data)
        assert step.get_name() == "shuffle_functions"


class TestIndentFunctionsStep:
    """Tests for indent_functions step."""

    def test_sets_indent_preference(self):
        """Sets JSON indent preference."""
        step = IndentFunctionsStep(
            StepConfig(type="indent_functions", params={"choices": [2, 4]}),
            random.Random(42),
        )

        data = {}
        result = step.apply(data)

        assert "_json_indent" in result
        assert result["_json_indent"] in [2, 4]

    def test_default_choices(self):
        """Uses default indent choices."""
        step = IndentFunctionsStep(
            StepConfig(type="indent_functions"),
            random.Random(42),
        )

        data = {}
        result = step.apply(data)

        assert result["_json_indent"] in [None, 2, 4]

    def test_stage_is_formatting(self):
        """Step stage is formatting."""
        assert IndentFunctionsStep.stage == "formatting"


class TestNewlineVariationStep:
    """Tests for newline_variation step."""

    def test_can_change_newlines(self):
        """Newlines can be changed."""
        step = NewlineVariationStep(
            StepConfig(type="newline_variation"),
            random.Random(42),
        )

        data = {"text": "line1\nline2\nline3"}
        result = step.apply(data)

        # Either \n or \r\n
        assert "\n" in result["text"] or "\r\n" in result["text"]

    def test_empty_text(self):
        """Handles empty text."""
        step = NewlineVariationStep(
            StepConfig(type="newline_variation"),
            random.Random(42),
        )

        data = {"text": ""}
        result = step.apply(data)
        assert result["text"] == ""

    def test_stage_is_presentation(self):
        """Step stage is presentation."""
        assert NewlineVariationStep.stage == "presentation"


class TestCreateStep:
    """Tests for create_step factory."""

    def test_create_shuffle_functions(self):
        """Creates shuffle_functions step."""
        config = StepConfig(type="shuffle_functions")
        step = create_step(config, random.Random(42))
        assert isinstance(step, ShuffleFunctionsStep)

    def test_create_indent_functions(self):
        """Creates indent_functions step."""
        config = StepConfig(type="indent_functions")
        step = create_step(config, random.Random(42))
        assert isinstance(step, IndentFunctionsStep)

    def test_create_newline_variation(self):
        """Creates newline_variation step."""
        config = StepConfig(type="newline_variation")
        step = create_step(config, random.Random(42))
        assert isinstance(step, NewlineVariationStep)

    def test_unknown_step_type(self):
        """Raises error for unknown step type."""
        config = StepConfig(type="unknown_step")
        with pytest.raises(ValueError, match="Unknown step type"):
            create_step(config, random.Random(42))


class TestGetStepStage:
    """Tests for get_step_stage."""

    def test_formatting_steps(self):
        """Formatting steps have correct stage."""
        assert get_step_stage("shuffle_functions") == "formatting"
        assert get_step_stage("indent_functions") == "formatting"

    def test_presentation_steps(self):
        """Presentation steps have correct stage."""
        assert get_step_stage("newline_variation") == "presentation"

    def test_unknown_step(self):
        """Raises error for unknown step."""
        with pytest.raises(ValueError):
            get_step_stage("unknown")
