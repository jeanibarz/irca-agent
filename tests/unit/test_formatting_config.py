"""
Unit tests for formatting configuration.

Tests FR-DATA-27: Format Config.
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.formatting.config import (
    FormatConfig,
    StepConfig,
    create_baseline_config,
    load_format_config,
)


class TestStepConfig:
    """Tests for StepConfig."""

    def test_step_config_basic(self):
        """StepConfig with required fields."""
        step = StepConfig(type="translate")
        assert step.type == "translate"
        assert step.probability == 1.0
        assert step.params is None

    def test_step_config_with_params(self):
        """StepConfig with params."""
        step = StepConfig(
            type="translate",
            probability=0.5,
            params={"languages": ["fr", "es"]},
        )
        assert step.type == "translate"
        assert step.probability == 0.5
        assert step.params["languages"] == ["fr", "es"]

    def test_step_config_probability_bounds(self):
        """StepConfig probability is bounded."""
        with pytest.raises(ValueError):
            StepConfig(type="test", probability=1.5)
        with pytest.raises(ValueError):
            StepConfig(type="test", probability=-0.1)


class TestFormatConfig:
    """Tests for FormatConfig."""

    def test_format_config_defaults(self):
        """FormatConfig has sensible defaults."""
        config = FormatConfig()
        assert config.seed == 42
        assert config.multiply == 1
        assert config.deduplicate is True
        assert config.source_column == "corrected_agent_trace"
        assert config.structural == []
        assert config.formatting == []
        assert config.presentation == []

    def test_format_config_with_steps(self):
        """FormatConfig with augmentation steps."""
        config = FormatConfig(
            seed=123,
            multiply=3,
            structural=[StepConfig(type="translate", probability=0.5)],
            formatting=[StepConfig(type="shuffle_functions", probability=0.8)],
            presentation=[StepConfig(type="newline_variation", probability=0.2)],
        )
        assert config.seed == 123
        assert config.multiply == 3
        assert len(config.structural) == 1
        assert len(config.formatting) == 1
        assert len(config.presentation) == 1

    def test_format_config_from_dict(self):
        """FormatConfig can be created from dict."""
        data = {
            "seed": 99,
            "multiply": 2,
            "structural": [{"type": "translate", "probability": 0.5}],
            "formatting": [{"type": "shuffle_functions"}],
        }
        config = FormatConfig(**data)
        assert config.seed == 99
        assert config.multiply == 2
        assert len(config.structural) == 1
        assert config.structural[0].type == "translate"

    def test_has_augmentations_false(self):
        """has_augmentations returns False when no steps."""
        config = FormatConfig()
        assert config.has_augmentations() is False

    def test_has_augmentations_true(self):
        """has_augmentations returns True when steps present."""
        config = FormatConfig(formatting=[StepConfig(type="shuffle_functions")])
        assert config.has_augmentations() is True

    def test_get_all_steps(self):
        """get_all_steps returns all steps with stages."""
        config = FormatConfig(
            structural=[StepConfig(type="translate")],
            formatting=[StepConfig(type="shuffle_functions")],
            presentation=[StepConfig(type="newline_variation")],
        )
        steps = config.get_all_steps()
        assert len(steps) == 3
        assert steps[0] == ("structural", config.structural[0])
        assert steps[1] == ("formatting", config.formatting[0])
        assert steps[2] == ("presentation", config.presentation[0])


class TestLoadFormatConfig:
    """Tests for load_format_config."""

    def test_load_valid_config(self):
        """Load valid JSON config file."""
        config_data = {
            "seed": 42,
            "multiply": 2,
            "formatting": [{"type": "shuffle_functions", "probability": 0.8}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            config = load_format_config(config_path)
            assert config.seed == 42
            assert config.multiply == 2
            assert len(config.formatting) == 1
        finally:
            Path(config_path).unlink()

    def test_load_nonexistent_file(self):
        """FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_format_config("/nonexistent/path/config.json")

    def test_load_invalid_json(self):
        """ValueError for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            config_path = f.name

        try:
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_format_config(config_path)
        finally:
            Path(config_path).unlink()


class TestCreateBaselineConfig:
    """Tests for create_baseline_config."""

    def test_baseline_config_defaults(self):
        """Baseline config has no augmentations."""
        config = create_baseline_config()
        assert config.multiply == 1
        assert config.structural == []
        assert config.formatting == []
        assert config.presentation == []
        assert config.has_augmentations() is False

    def test_baseline_config_custom_column(self):
        """Baseline config with custom source column."""
        config = create_baseline_config(source_column="agent_trace")
        assert config.source_column == "agent_trace"
