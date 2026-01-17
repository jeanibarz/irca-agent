"""
Unit tests for augmentation configuration.

Tests FR-DATA-12: Pipeline Configuration.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from augmentation.config import (
    AugmentationConfig,
    PipelineStep,
    TranslateParams,
    load_config,
    validate_config,
)


class TestPipelineStep(unittest.TestCase):
    """Tests for PipelineStep validation."""

    def test_valid_translate_step(self):
        """Valid translate step with languages."""
        step = PipelineStep(
            type="translate",
            probability=0.5,
            params={"languages": ["fr", "es"]},
        )
        self.assertEqual(step.type, "translate")
        self.assertEqual(step.probability, 0.5)

    def test_valid_shuffle_step(self):
        """Valid shuffle_functions step without params."""
        step = PipelineStep(
            type="shuffle_functions",
            probability=0.8,
        )
        self.assertEqual(step.type, "shuffle_functions")
        self.assertIsNone(step.params)

    def test_invalid_step_type(self):
        """Invalid step type raises error."""
        with self.assertRaises(ValueError) as ctx:
            PipelineStep(type="invalid_type", probability=0.5)
        self.assertIn("Invalid step type", str(ctx.exception))

    def test_probability_out_of_range(self):
        """Probability outside 0-1 raises error."""
        with self.assertRaises(ValueError):
            PipelineStep(type="shuffle_functions", probability=1.5)

        with self.assertRaises(ValueError):
            PipelineStep(type="shuffle_functions", probability=-0.1)

    def test_translate_requires_params(self):
        """Translate step requires params with languages."""
        with self.assertRaises(ValueError) as ctx:
            PipelineStep(type="translate", probability=0.5)
        self.assertIn("requires 'params'", str(ctx.exception))


class TestTranslateParams(unittest.TestCase):
    """Tests for TranslateParams validation."""

    def test_valid_params(self):
        """Valid translation params."""
        params = TranslateParams(languages=["fr", "es", "de"])
        self.assertEqual(len(params.languages), 3)
        self.assertIsNone(params.weights)

    def test_valid_params_with_weights(self):
        """Valid translation params with weights."""
        params = TranslateParams(
            languages=["fr", "es"],
            weights=[0.6, 0.4],
        )
        self.assertEqual(params.weights, [0.6, 0.4])

    def test_weights_must_match_languages(self):
        """Weights length must match languages length."""
        with self.assertRaises(ValueError) as ctx:
            TranslateParams(
                languages=["fr", "es"],
                weights=[0.5],  # Only one weight for two languages
            )
        self.assertIn("must match", str(ctx.exception))

    def test_weights_must_sum_to_one(self):
        """Weights must sum to 1.0."""
        with self.assertRaises(ValueError) as ctx:
            TranslateParams(
                languages=["fr", "es"],
                weights=[0.3, 0.3],  # Sums to 0.6
            )
        self.assertIn("must sum to 1.0", str(ctx.exception))

    def test_empty_languages_rejected(self):
        """Empty languages list is rejected."""
        with self.assertRaises(ValueError):
            TranslateParams(languages=[])


class TestAugmentationConfig(unittest.TestCase):
    """Tests for AugmentationConfig validation."""

    def test_valid_config(self):
        """Valid complete configuration."""
        config = AugmentationConfig(
            input="datasets/input",
            output="datasets/output",
            multiply=3,
            seed=42,
            pipeline=[
                PipelineStep(
                    type="translate",
                    probability=0.5,
                    params={"languages": ["fr"]},
                ),
            ],
        )
        self.assertEqual(config.multiply, 3)
        self.assertEqual(config.seed, 42)
        self.assertTrue(config.deduplicate)  # Default value

    def test_config_without_paths(self):
        """Config without input/output is valid (paths from CLI)."""
        config = AugmentationConfig(
            multiply=2,
            seed=42,
            pipeline=[
                PipelineStep(type="shuffle_functions", probability=0.8),
            ],
        )
        self.assertIsNone(config.input)
        self.assertIsNone(config.output)

    def test_multiply_must_be_positive(self):
        """Multiply must be >= 1."""
        with self.assertRaises(ValueError):
            AugmentationConfig(
                multiply=0,
                seed=42,
                pipeline=[PipelineStep(type="shuffle_functions", probability=0.8)],
            )

    def test_pipeline_required(self):
        """Pipeline is required and must have at least one step."""
        with self.assertRaises(ValueError):
            AugmentationConfig(
                multiply=2,
                seed=42,
                pipeline=[],
            )

    def test_deduplicate_options(self):
        """Deduplicate accepts true, false, or 'fuzzy'."""
        # Boolean values
        config1 = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=True,
            pipeline=[PipelineStep(type="shuffle_functions", probability=0.8)],
        )
        self.assertTrue(config1.deduplicate)

        config2 = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate=False,
            pipeline=[PipelineStep(type="shuffle_functions", probability=0.8)],
        )
        self.assertFalse(config2.deduplicate)

        # String 'fuzzy'
        config3 = AugmentationConfig(
            multiply=2,
            seed=42,
            deduplicate="fuzzy",
            pipeline=[PipelineStep(type="shuffle_functions", probability=0.8)],
        )
        self.assertEqual(config3.deduplicate, "fuzzy")


class TestLoadConfig(unittest.TestCase):
    """Tests for loading config from file."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary files."""
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_load_valid_config(self):
        """Load a valid JSON config file."""
        config_data = {
            "multiply": 3,
            "seed": 42,
            "pipeline": [
                {"type": "translate", "probability": 0.5, "params": {"languages": ["fr"]}},
                {"type": "shuffle_functions", "probability": 0.8},
            ],
        }

        config_path = Path(self.test_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        config = load_config(config_path)
        self.assertEqual(config.multiply, 3)
        self.assertEqual(len(config.pipeline), 2)

    def test_load_missing_file(self):
        """Loading non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")

    def test_load_invalid_json(self):
        """Loading invalid JSON raises ValueError."""
        config_path = Path(self.test_dir) / "invalid.json"
        with open(config_path, "w") as f:
            f.write("{ invalid json }")

        with self.assertRaises(ValueError) as ctx:
            load_config(config_path)
        self.assertIn("Invalid JSON", str(ctx.exception))

    def test_load_invalid_config(self):
        """Loading JSON with invalid config raises ValueError."""
        config_data = {
            "multiply": 3,
            # Missing required 'seed' and 'pipeline'
        }

        config_path = Path(self.test_dir) / "incomplete.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        with self.assertRaises(ValueError) as ctx:
            load_config(config_path)
        self.assertIn("Config validation failed", str(ctx.exception))


class TestValidateConfig(unittest.TestCase):
    """Tests for validate_config function."""

    def test_validate_dict(self):
        """Validate a dictionary configuration."""
        data = {
            "multiply": 2,
            "seed": 123,
            "pipeline": [
                {"type": "newline_variation", "probability": 0.3},
            ],
        }
        config = validate_config(data)
        self.assertIsInstance(config, AugmentationConfig)
        self.assertEqual(config.multiply, 2)

    def test_validate_invalid_dict(self):
        """Validate invalid dictionary raises ValueError."""
        data = {"invalid": "data"}
        with self.assertRaises(ValueError):
            validate_config(data)


if __name__ == "__main__":
    unittest.main()
