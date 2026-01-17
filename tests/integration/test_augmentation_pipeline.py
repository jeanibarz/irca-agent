"""
Integration tests for augmentation pipeline CLI.

Tests FR-DATA-11 through FR-DATA-17.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

import datasets

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from cli.commands.dataset import dataset
from core.prompt_builder import build_full_prompt


class TestConfigDrivenCLI(unittest.TestCase):
    """
    Tests for FR-DATA-17: Config-Driven CLI.
    """

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.test_dir, "input_dataset")
        self.output_path = os.path.join(self.test_dir, "output_dataset")
        self.config_path = os.path.join(self.test_dir, "config.json")

        # Create sample dataset
        sample_parts = {
            "system_instructions": "You are an AI assistant.",
            "example": "Example<|wait|>",
            "available_functions_json": '[{"name": "test_func"}]',
            "user_query": "What is the weather?",
            "assistant_completion": "<thought>Processing.</thought>\n\n### FINAL ANSWER\nThe weather is sunny.",
        }
        sample_prompt = build_full_prompt(sample_parts)

        data = {"text": [sample_prompt] * 10}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.input_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_config_required(self):
        """CLI requires --config argument."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment-pipeline", "-i", self.input_path, "-o", self.output_path],
            obj={"verbose": False},
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--config", result.output)

    def test_config_file_not_found(self):
        """CLI reports error for missing config file."""
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment-pipeline", "-c", "/nonexistent/config.json"],
            obj={"verbose": False},
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_cli_paths_override_config(self):
        """CLI --input and --output override config values."""
        # Config with different paths
        config = {
            "input": "/wrong/input/path",
            "output": "/wrong/output/path",
            "multiply": 2,
            "seed": 42,
            "pipeline": [{"type": "shuffle_functions", "probability": 0.0}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            [
                "augment-pipeline",
                "-c",
                self.config_path,
                "-i",
                self.input_path,  # Override
                "-o",
                self.output_path,  # Override
            ],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
        self.assertTrue(os.path.exists(self.output_path))

    def test_config_paths_used_when_no_cli_override(self):
        """Config paths used when not overridden by CLI."""
        config = {
            "input": self.input_path,
            "output": self.output_path,
            "multiply": 2,
            "seed": 42,
            "pipeline": [{"type": "shuffle_functions", "probability": 0.0}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment-pipeline", "-c", self.config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")
        self.assertTrue(os.path.exists(self.output_path))


class TestDatasetMultiplication(unittest.TestCase):
    """
    Tests for FR-DATA-11: Dataset Size Multiplication.
    """

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.test_dir, "input_dataset")
        self.output_path = os.path.join(self.test_dir, "output_dataset")
        self.config_path = os.path.join(self.test_dir, "config.json")

        # Create sample dataset with 10 samples
        data = {"text": [f"Sample text {i}" for i in range(10)]}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.input_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multiply_three(self):
        """Dataset is multiplied by 3."""
        config = {
            "input": self.input_path,
            "output": self.output_path,
            "multiply": 3,
            "seed": 42,
            "deduplicate": False,
            "pipeline": [{"type": "newline_variation", "probability": 0.5}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment-pipeline", "-c", self.config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

        # Load and verify output
        output_ds = datasets.load_from_disk(self.output_path)
        self.assertEqual(len(output_ds), 30)  # 10 * 3

    def test_original_preserved(self):
        """Original samples have variant_id=0."""
        config = {
            "input": self.input_path,
            "output": self.output_path,
            "multiply": 2,
            "seed": 42,
            "deduplicate": False,
            "pipeline": [{"type": "shuffle_functions", "probability": 1.0}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment-pipeline", "-c", self.config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        output_ds = datasets.load_from_disk(self.output_path)
        originals = [r for r in output_ds if r["variant_id"] == 0]
        self.assertEqual(len(originals), 10)

        for original in originals:
            self.assertEqual(original["augmentations"], [])


class TestFeatureCombination(unittest.TestCase):
    """
    Tests for FR-DATA-13: Feature Combination.
    """

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.test_dir, "input_dataset")
        self.output_path = os.path.join(self.test_dir, "output_dataset")
        self.config_path = os.path.join(self.test_dir, "config.json")

        data = {"text": ["Sample with functions: " + json.dumps([{"name": "func1"}])] * 20}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.input_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multiple_augmentations_combined(self):
        """Multiple augmentations applied to same variant."""
        config = {
            "input": self.input_path,
            "output": self.output_path,
            "multiply": 2,
            "seed": 42,
            "deduplicate": False,
            "pipeline": [
                {"type": "shuffle_functions", "probability": 1.0},
                {"type": "newline_variation", "probability": 1.0},
            ],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment-pipeline", "-c", self.config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        output_ds = datasets.load_from_disk(self.output_path)
        variants = [r for r in output_ds if r["variant_id"] > 0]

        # All variants should have both augmentations
        for variant in variants:
            self.assertEqual(len(variant["augmentations"]), 2)


class TestAugmentationMetadata(unittest.TestCase):
    """
    Tests for FR-DATA-15: Augmentation Metadata.
    """

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.test_dir, "input_dataset")
        self.output_path = os.path.join(self.test_dir, "output_dataset")
        self.config_path = os.path.join(self.test_dir, "config.json")

        data = {"text": ["Sample text"] * 10}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.input_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_metadata_file_created(self):
        """Metadata sidecar file is created."""
        config = {
            "input": self.input_path,
            "output": self.output_path,
            "multiply": 2,
            "seed": 42,
            "pipeline": [{"type": "shuffle_functions", "probability": 0.5}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment-pipeline", "-c", self.config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        metadata_path = os.path.join(self.output_path, "_augmentation_metadata.json")
        self.assertTrue(os.path.exists(metadata_path))

    def test_metadata_contains_required_fields(self):
        """Metadata contains all required fields."""
        config = {
            "input": self.input_path,
            "output": self.output_path,
            "multiply": 2,
            "seed": 42,
            "pipeline": [{"type": "shuffle_functions", "probability": 0.5}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        runner.invoke(
            dataset,
            ["augment-pipeline", "-c", self.config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        metadata_path = os.path.join(self.output_path, "_augmentation_metadata.json")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Check required fields
        self.assertIn("schema_version", metadata)
        self.assertIn("timestamp", metadata)
        self.assertIn("git", metadata)
        self.assertIn("config", metadata)
        self.assertIn("input", metadata)
        self.assertIn("output", metadata)
        self.assertIn("statistics", metadata)
        self.assertIn("environment", metadata)

        # Check nested fields
        self.assertIn("commit_hash", metadata["git"])
        self.assertIn("multiply", metadata["config"])
        self.assertIn("num_samples", metadata["input"])
        self.assertIn("augmentation_counts", metadata["statistics"])
        self.assertIn("python_version", metadata["environment"])


class TestWandbTracking(unittest.TestCase):
    """
    Tests for FR-DATA-16: W&B Augmentation Tracking.
    """

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.test_dir, "input_dataset")
        self.output_path = os.path.join(self.test_dir, "output_dataset")
        self.config_path = os.path.join(self.test_dir, "config.json")

        data = {"text": ["Sample text"] * 5}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.input_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_wandb_not_called_without_flag(self):
        """W&B is not called when --track-wandb is not provided."""
        config = {
            "input": self.input_path,
            "output": self.output_path,
            "multiply": 2,
            "seed": 42,
            "pipeline": [{"type": "shuffle_functions", "probability": 0.5}],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        with patch("wandb.init") as mock_init:
            runner = CliRunner()
            result = runner.invoke(
                dataset,
                ["augment-pipeline", "-c", self.config_path],
                obj={"verbose": False},
                catch_exceptions=False,
            )

            self.assertEqual(result.exit_code, 0)
            mock_init.assert_not_called()


class TestTranslationIntegration(unittest.TestCase):
    """
    Integration tests for translation step with mocked translator.
    """

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.test_dir, "input_dataset")
        self.output_path = os.path.join(self.test_dir, "output_dataset")
        self.config_path = os.path.join(self.test_dir, "config.json")

        # Create dataset with FINAL ANSWER section
        sample_parts = {
            "system_instructions": "You are an AI assistant.",
            "example": "Example<|wait|>",
            "available_functions_json": "[]",
            "user_query": "Show me the data",
            "assistant_completion": "<thought>Processing.</thought>\n\n### FINAL ANSWER\nHere is your data.",
        }
        sample_prompt = build_full_prompt(sample_parts)

        data = {"text": [sample_prompt] * 10}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.input_path)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("src.dataset_generation.translator.TranslationService")
    def test_translation_applied(self, mock_service_cls):
        """Translation step applies translation."""
        # Mock translator
        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = lambda x: [f"[FR] {s}" for s in x]
        mock_service_cls.return_value = mock_translator

        config = {
            "input": self.input_path,
            "output": self.output_path,
            "multiply": 2,
            "seed": 42,
            "deduplicate": False,
            "pipeline": [
                {"type": "translate", "probability": 1.0, "params": {"languages": ["fr"]}},
            ],
        }
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment-pipeline", "-c", self.config_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

        output_ds = datasets.load_from_disk(self.output_path)
        variants = [r for r in output_ds if r["variant_id"] > 0]

        # Check that translation was applied
        for variant in variants:
            self.assertIn("translate:fr", variant["augmentations"])


if __name__ == "__main__":
    unittest.main()
