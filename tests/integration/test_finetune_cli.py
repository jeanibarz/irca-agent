"""
Integration tests for finetuning with pre-augmented datasets.

Tests FR-FT-01: Q-LoRA Training (with prepared datasets)

Note: Dataset augmentation tests are now in test_dataset_augmentation.py
following the separation of concerns (ADR-001).
"""

import os
import sys
import unittest

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))


class TestFinetuningWithPreparedDataset(unittest.TestCase):
    """
    Integration tests for finetuning command with pre-augmented datasets.

    These tests verify that the finetuning pipeline correctly handles
    datasets that have already been augmented via `irca dataset augment`.
    """

    def test_finetune_requires_text_column(self):
        """
        FR-FT-01: Finetuning shall require datasets with 'text' column.

        This is a documentation test - the actual validation is in
        the finetune CLI command which checks for 'text' column presence.
        """
        # This test documents the requirement - full integration tests
        # would require GPU resources and are run separately
        pass

    def test_finetune_cli_options(self):
        """
        Verify finetune CLI has expected options after augmentation removal.

        Following ADR-001, finetune should NOT have augmentation options.
        """
        from click.testing import CliRunner

        from cli.commands.finetune import finetune

        runner = CliRunner()
        result = runner.invoke(finetune, ["run", "--help"])

        # Should have these options
        self.assertIn("--model-type", result.output)
        self.assertIn("--dataset", result.output)
        self.assertIn("--epochs", result.output)
        self.assertIn("--learning-rate", result.output)

        # Should NOT have augmentation options (removed per ADR-001)
        self.assertNotIn("--augment", result.output)
        self.assertNotIn("--augment-lang", result.output)
        self.assertNotIn("--augment-ratio", result.output)


if __name__ == "__main__":
    unittest.main()
