"""
Integration tests for format CLI command.

Tests FR-DATA-24: Dataset Format Command.
Tests FR-DATA-26: Format Baseline.
Tests FR-DATA-28: Diversity Evaluation Workflow.
"""

import json
import os
import shutil
import tempfile
import unittest

from click.testing import CliRunner

import datasets
from src.cli.commands.dataset import dataset


class TestFormatCLIBaseline(unittest.TestCase):
    """Tests for format CLI in baseline mode."""

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.test_dir, "test_dataset")
        self.output_path = os.path.join(self.test_dir, "formatted_dataset")

        # Create sample dataset with corrected_agent_trace format
        sample_trace = """
### INSTRUCTIONS
You are an AI assistant.

EXAMPLE:
Sample example.

The functions available to you are described below.

### FUNCTIONS AVAILABLE
[{"name": "test_func", "description": "A test function"}]

### USER QUERY
What is the weather?

### ITERATIVE RESOLUTION CYCLE
Thought: I need to check.
Action choice: final answer

### FINAL ANSWER
It is sunny.
"""
        data = {
            "corrected_agent_trace": [[{"value": sample_trace, "status": "submitted"}] for _ in range(10)],
            "user_query": ["What is the weather?"] * 10,
        }
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.dataset_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_format_baseline_creates_text_column(self):
        """Baseline format creates text column."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", self.output_path, "--no-augment"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

        # Load output and verify
        output_ds = datasets.load_from_disk(self.output_path)
        self.assertIn("text", output_ds.column_names)
        self.assertEqual(len(output_ds), 10)

    def test_format_baseline_text_contains_prompt_parts(self):
        """Baseline format includes all prompt parts."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", self.output_path, "--no-augment"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        output_ds = datasets.load_from_disk(self.output_path)
        text = output_ds[0]["text"]

        self.assertIn("### INSTRUCTIONS", text)
        self.assertIn("### FUNCTIONS AVAILABLE", text)
        self.assertIn("### USER QUERY", text)
        self.assertIn("What is the weather?", text)

    def test_format_baseline_reproducible(self):
        """Baseline format is reproducible."""
        runner = CliRunner()

        # First run
        result1 = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", self.output_path, "--no-augment", "--seed", "42"],
            obj={"verbose": False},
            catch_exceptions=False,
        )
        self.assertEqual(result1.exit_code, 0)
        output1 = datasets.load_from_disk(self.output_path)
        text1 = output1[0]["text"]

        # Second run
        output_path2 = os.path.join(self.test_dir, "formatted_dataset2")
        result2 = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", output_path2, "--no-augment", "--seed", "42"],
            obj={"verbose": False},
            catch_exceptions=False,
        )
        self.assertEqual(result2.exit_code, 0)
        output2 = datasets.load_from_disk(output_path2)
        text2 = output2[0]["text"]

        self.assertEqual(text1, text2)


class TestFormatCLIWithConfig(unittest.TestCase):
    """Tests for format CLI with augmentation config."""

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.test_dir, "test_dataset")
        self.output_path = os.path.join(self.test_dir, "formatted_dataset")
        self.config_path = os.path.join(self.test_dir, "config.json")

        # Create sample dataset
        sample_trace = """
### INSTRUCTIONS
Test.

EXAMPLE:


The functions available to you are described below.

### FUNCTIONS AVAILABLE
[{"name": "func_a"}, {"name": "func_b"}, {"name": "func_c"}]

### USER QUERY
Query.

### ITERATIVE RESOLUTION CYCLE
Completion.
"""
        data = {
            "corrected_agent_trace": [[{"value": sample_trace, "status": "submitted"}] for _ in range(5)],
        }
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.dataset_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_format_with_shuffle_augmentation(self):
        """Format with shuffle_functions augmentation."""
        config = {
            "seed": 42,
            "formatting": [{"type": "shuffle_functions", "probability": 1.0}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", self.output_path, "-c", self.config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
        self.assertIn("shuffle_functions", result.output)

    def test_format_with_multiply(self):
        """Format with dataset multiplication."""
        config = {
            "seed": 42,
            "multiply": 3,
            "formatting": [{"type": "shuffle_functions", "probability": 1.0}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", self.output_path, "-c", self.config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

        # Should have more samples (5 * 3 = 15, minus deduplication)
        output_ds = datasets.load_from_disk(self.output_path)
        self.assertGreater(len(output_ds), 5)

    def test_no_augment_flag_overrides_config(self):
        """--no-augment flag disables config augmentations."""
        config = {
            "seed": 42,
            "formatting": [{"type": "shuffle_functions", "probability": 1.0}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", self.output_path, "-c", self.config_path, "--no-augment"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("Augmentations disabled", result.output)


class TestFormatCLIErrors(unittest.TestCase):
    """Tests for format CLI error handling."""

    def setUp(self):
        """Create test directory."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_missing_dataset(self):
        """Error for non-existent dataset."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["format", "-i", "/nonexistent/path", "-o", "/tmp/out"],
            obj={"verbose": False},
        )

        self.assertNotEqual(result.exit_code, 0)
        # Error message could be "not found" or "neither a dataset directory"
        output_lower = result.output.lower()
        self.assertTrue(
            "not found" in output_lower or "neither a" in output_lower or "error" in output_lower,
            f"Expected error message, got: {result.output}",
        )

    def test_missing_source_column(self):
        """Error when source column missing."""
        # Create dataset without corrected_agent_trace
        dataset_path = os.path.join(self.test_dir, "bad_dataset")
        data = {"other_column": ["value"]}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(dataset_path)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["format", "-i", dataset_path, "-o", os.path.join(self.test_dir, "out")],
            obj={"verbose": False},
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("corrected_agent_trace", result.output)


class TestFormatDiversityWorkflow(unittest.TestCase):
    """Tests for format → diversity workflow (FR-DATA-28)."""

    def setUp(self):
        """Create test dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.test_dir, "raw_dataset")
        self.baseline_path = os.path.join(self.test_dir, "baseline")
        self.augmented_path = os.path.join(self.test_dir, "augmented")

        sample_trace = """
### INSTRUCTIONS
Test instructions.

EXAMPLE:
Example.

The functions available to you are described below.

### FUNCTIONS AVAILABLE
[{"name": "func_a"}, {"name": "func_b"}]

### USER QUERY
Test query.

### ITERATIVE RESOLUTION CYCLE
Test completion.
"""
        data = {
            "corrected_agent_trace": [[{"value": sample_trace}] for _ in range(20)],
        }
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.dataset_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_format_then_diversity_single(self):
        """Format then run diversity on single dataset."""
        runner = CliRunner()

        # Format
        result = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", self.baseline_path, "--no-augment"],
            obj={"verbose": False},
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

        # Diversity
        result = runner.invoke(
            dataset,
            ["diversity", "-d", self.baseline_path, "--quick"],
            obj={"verbose": False},
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, f"Diversity failed: {result.output}")
        self.assertIn("Distinct-1", result.output)

    def test_format_then_diversity_comparison(self):
        """Format baseline and augmented, then compare diversity."""
        runner = CliRunner()

        # Format baseline
        result = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", self.baseline_path, "--no-augment"],
            obj={"verbose": False},
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

        # Create augment config
        config_path = os.path.join(self.test_dir, "config.json")
        config = {
            "seed": 42,
            "formatting": [{"type": "shuffle_functions", "probability": 1.0}],
            "presentation": [{"type": "newline_variation", "probability": 0.5}],
        }
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Format augmented
        result = runner.invoke(
            dataset,
            ["format", "-i", self.dataset_path, "-o", self.augmented_path, "-c", config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0)

        # Compare diversity
        result = runner.invoke(
            dataset,
            ["diversity", "-o", self.baseline_path, "-a", self.augmented_path, "--quick"],
            obj={"verbose": False},
            catch_exceptions=False,
        )
        self.assertEqual(result.exit_code, 0, f"Comparison failed: {result.output}")
        self.assertIn("Original", result.output)
        self.assertIn("Augmented", result.output)
        self.assertIn("Change", result.output)


if __name__ == "__main__":
    unittest.main()
