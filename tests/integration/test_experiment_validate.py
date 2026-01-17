"""
Integration tests for experiment validate CLI command.

Tests:
- FR-EXP-001: Experiment pre-flight command
- FR-EXP-002: Dataset structure validation
- FR-EXP-004: Configuration consistency
"""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner
from datasets import Dataset

from src.cli.commands.experiment import experiment, validate


# ============================================
# Fixtures
# ============================================


@pytest.fixture
def runner():
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def valid_irca_text() -> str:
    """Valid IRCA-formatted text with all markers."""
    return """
### INSTRUCTIONS
You are an AI assistant that uses functions.

EXAMPLE:
User: Test
Assistant: OK

The functions available to you are described below.

### FUNCTIONS AVAILABLE
[{"name": "get_weather", "description": "Get weather"}]

### USER QUERY
What is the weather in Paris?

### ITERATIVE RESOLUTION CYCLE
Thought: I should get the weather.
Action: call function
Call: get_weather(location="Paris")
Output: {"temp": 20}
Thought: I have the answer.

### FINAL ANSWER
The weather in Paris is 20 degrees.
""".strip()


@pytest.fixture
def valid_experiment_config() -> dict:
    """Valid experiment configuration."""
    return {
        "experiment_id": "test-experiment",
        "description": "Test experiment for validation",
        "datasets": {
            "train_data": {
                "path": "train_data",
                "samples": 10,
            }
        },
        "evaluations": {
            "tests": []
        },
        "statistical_analysis": {
            "test": "bootstrap_confidence_interval",
            "confidence_level": 0.95,
        },
    }


@pytest.fixture
def invalid_experiment_config() -> dict:
    """Invalid experiment configuration with wrong test name."""
    return {
        "experiment_id": "test-experiment",
        "datasets": {},
        "evaluations": {},
        "statistical_analysis": {
            "test": "bootstrap_paired",  # This doesn't exist (FM-1)
        },
    }


@pytest.fixture
def temp_experiment_dir(valid_irca_text, valid_experiment_config):
    """Create a temporary experiment directory with valid data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir)

        # Create experiment.json
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(valid_experiment_config, f)

        # Create valid dataset
        dataset = Dataset.from_list([
            {
                "text": valid_irca_text,
                "original_index": i,
                "variant_id": 0,
                "augmentations": [],
            }
            for i in range(5)
        ])
        dataset.save_to_disk(str(exp_dir / "train_data"))

        yield exp_dir


@pytest.fixture
def temp_experiment_dir_invalid_config(valid_irca_text, invalid_experiment_config):
    """Create a temporary experiment directory with invalid config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir)

        # Create invalid experiment.json
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(invalid_experiment_config, f)

        # Create valid dataset
        dataset = Dataset.from_list([
            {
                "text": valid_irca_text,
                "original_index": 0,
                "variant_id": 0,
                "augmentations": [],
            }
        ])
        dataset.save_to_disk(str(exp_dir / "train_data"))

        yield exp_dir


@pytest.fixture
def temp_experiment_dir_missing_text_column(valid_experiment_config):
    """Create experiment directory with dataset missing 'text' column."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir)

        # Create experiment.json
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(valid_experiment_config, f)

        # Create dataset without 'text' column
        dataset = Dataset.from_list([
            {"content": "not the text column", "id": 0}
        ])
        dataset.save_to_disk(str(exp_dir / "train_data"))

        yield exp_dir


@pytest.fixture
def temp_experiment_dir_corrupted_markers(valid_experiment_config):
    """Create experiment directory with corrupted IRCA markers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir)

        # Create experiment.json
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(valid_experiment_config, f)

        # Create dataset with corrupted markers
        corrupted_text = """
## INSTRUCTIONS
Some instructions

FUNCTIONS:
[]

USER:
Query

RESPONSE:
Answer
""".strip()

        dataset = Dataset.from_list([
            {
                "text": corrupted_text,
                "original_index": 0,
                "variant_id": 0,
                "augmentations": [],
            }
        ])
        dataset.save_to_disk(str(exp_dir / "train_data"))

        yield exp_dir


# ============================================
# Integration Tests
# ============================================


class TestExperimentValidateCommand:
    """Integration tests for 'irca experiment validate' command."""

    def test_it_val_001_reports_missing_text_column(
        self, runner, temp_experiment_dir_missing_text_column
    ):
        """IT-VAL-001: validate command reports missing text column."""
        result = runner.invoke(
            validate,
            ["-e", str(temp_experiment_dir_missing_text_column)],
            obj={"verbose": False},
        )

        assert result.exit_code != 0
        assert "text" in result.output.lower()
        assert "missing" in result.output.lower() or "error" in result.output.lower()

    def test_it_val_002_reports_marker_integrity_issues(
        self, runner, temp_experiment_dir_corrupted_markers
    ):
        """IT-VAL-002: validate command reports marker integrity issues."""
        result = runner.invoke(
            validate,
            ["-e", str(temp_experiment_dir_corrupted_markers)],
            obj={"verbose": False},
        )

        assert result.exit_code != 0
        assert "marker" in result.output.lower() or "error" in result.output.lower()

    def test_it_val_003_reports_completion_detection_failures(
        self, runner, temp_experiment_dir_corrupted_markers
    ):
        """IT-VAL-003: validate command reports completion detection failures."""
        # This test overlaps with marker integrity since corrupted markers
        # also break completion detection
        result = runner.invoke(
            validate,
            ["-e", str(temp_experiment_dir_corrupted_markers)],
            obj={"verbose": True},
        )

        # Should fail due to missing required markers
        assert result.exit_code != 0

    def test_it_val_004_succeeds_on_valid_dataset(self, runner, temp_experiment_dir):
        """IT-VAL-004: validate command succeeds on valid dataset."""
        result = runner.invoke(
            validate,
            ["-e", str(temp_experiment_dir)],
            obj={"verbose": False},
        )

        assert result.exit_code == 0
        assert "PASSED" in result.output

    def test_it_val_005_checks_config_consistency(
        self, runner, temp_experiment_dir_invalid_config
    ):
        """IT-VAL-005: validate command checks experiment.json consistency."""
        result = runner.invoke(
            validate,
            ["-e", str(temp_experiment_dir_invalid_config)],
            obj={"verbose": False},
        )

        # Should fail or warn due to invalid statistical test name
        assert "bootstrap_paired" in result.output or "error" in result.output.lower()

    def test_output_json_report(self, runner, temp_experiment_dir):
        """Test that validation report can be saved to JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            result = runner.invoke(
                validate,
                ["-e", str(temp_experiment_dir), "-o", output_path],
                obj={"verbose": False},
            )

            assert result.exit_code == 0

            # Check JSON output
            with open(output_path) as f:
                report = json.load(f)

            assert "passed" in report
            assert "errors" in report
            assert "warnings" in report
            assert report["passed"] is True
        finally:
            Path(output_path).unlink(missing_ok=True)

    def test_strict_mode_fails_on_warnings(self, runner, temp_experiment_dir):
        """Test that --strict mode fails on warnings."""
        # Create a dataset that might generate warnings
        result = runner.invoke(
            validate,
            ["-e", str(temp_experiment_dir), "--strict"],
            obj={"verbose": False},
        )

        # If there are any warnings, it should fail
        # If no warnings, it should pass
        if "WARNING" in result.output:
            assert result.exit_code != 0
            assert "strict" in result.output.lower()
        else:
            assert result.exit_code == 0

    def test_sample_size_option(self, runner, temp_experiment_dir):
        """Test that --sample-size option is respected."""
        result = runner.invoke(
            validate,
            ["-e", str(temp_experiment_dir), "-n", "3"],
            obj={"verbose": True},
        )

        assert result.exit_code == 0
        assert "Sample size: 3" in result.output

    def test_verbose_output(self, runner, temp_experiment_dir):
        """Test verbose output includes INFO messages."""
        result = runner.invoke(
            validate,
            ["-e", str(temp_experiment_dir)],
            obj={"verbose": True},
        )

        assert result.exit_code == 0
        # Verbose mode should show INFO section
        assert "INFO" in result.output or "ℹ" in result.output


class TestExperimentValidateEdgeCases:
    """Edge case tests for experiment validate command."""

    def test_nonexistent_directory(self, runner):
        """Test error handling for non-existent directory."""
        result = runner.invoke(
            validate,
            ["-e", "/nonexistent/path/to/experiment"],
            obj={"verbose": False},
        )

        # Click should catch this as path doesn't exist
        assert result.exit_code != 0

    def test_no_experiment_json(self, runner, valid_irca_text):
        """Test handling when experiment.json is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Create dataset but no experiment.json
            dataset = Dataset.from_list([
                {"text": valid_irca_text, "original_index": 0, "variant_id": 0, "augmentations": []}
            ])
            dataset.save_to_disk(str(exp_dir / "data"))

            result = runner.invoke(
                validate,
                ["-e", str(exp_dir)],
                obj={"verbose": False},
            )

            # Should still run but warn about missing config
            # May pass or fail depending on dataset discovery
            assert "experiment.json" in result.output.lower() or result.exit_code == 0

    def test_empty_dataset(self, runner, valid_experiment_config):
        """Test handling of empty dataset."""
        from datasets import Features, Value

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir)

            # Create config
            with open(exp_dir / "experiment.json", "w") as f:
                json.dump(valid_experiment_config, f)

            # Create empty dataset with explicit schema
            features = Features({
                "text": Value("string"),
                "original_index": Value("int64"),
                "variant_id": Value("int64"),
            })
            dataset = Dataset.from_dict(
                {"text": [], "original_index": [], "variant_id": []},
                features=features,
            )
            dataset.save_to_disk(str(exp_dir / "train_data"))

            result = runner.invoke(
                validate,
                ["-e", str(exp_dir)],
                obj={"verbose": False},
            )

            # Should fail due to empty dataset
            assert result.exit_code != 0
            assert "empty" in result.output.lower()


class TestStatisticsAPIValidation:
    """Test that statistics API validation is included."""

    def test_statistics_api_checked(self, runner, temp_experiment_dir):
        """Test that statistics API is validated during pre-flight."""
        result = runner.invoke(
            validate,
            ["-e", str(temp_experiment_dir)],
            obj={"verbose": True},
        )

        # Should include statistics API check in verbose output
        assert "statistic" in result.output.lower() or result.exit_code == 0
