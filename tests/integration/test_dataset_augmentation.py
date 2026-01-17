"""
Integration tests for dataset augmentation CLI command.

Tests FR-DATA-08: Separated Augmentation Pipeline
Tests FR-DATA-09: Markdown Link Preservation
Tests FR-DATA-10: Dataset Inspection

These tests verify the `irca dataset augment` command produces correct output
with proper handling of markdown links and multilingual translation.
"""

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


class TestDatasetAugmentationCLI(unittest.TestCase):
    """
    Integration tests for FR-DATA-08: Separated Augmentation Pipeline.

    Verifies that `irca dataset augment` command:
    - Produces a dataset with 'text' column
    - Preserves original columns
    - Applies translation to correct ratio of samples
    """

    def setUp(self):
        """Create a temporary directory and synthetic dataset for testing."""
        self.test_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.test_dir, "input_dataset")
        self.output_path = os.path.join(self.test_dir, "output_dataset")

        # Create realistic sample prompts
        self.sample_parts = {
            "system_instructions": "You are an AI assistant.",
            "example": "Example interaction<|wait|>",
            "available_functions_json": '[{"name": "test_func"}]',
            "user_query": "What is the weather today?",
            "assistant_completion": "<thought>I need to check the weather.</thought>\n\n### FINAL ANSWER\nThe weather is sunny.",
        }
        sample_prompt = build_full_prompt(self.sample_parts)

        # Create synthetic dataset with 20 samples
        self.n_samples = 20
        data = {
            "corrected_agent_trace": [[{"value": sample_prompt}]] * self.n_samples,
            "user_query": ["What is the weather today?"] * self.n_samples,
        }
        self.input_dataset = datasets.Dataset.from_dict(data)
        self.input_dataset.save_to_disk(self.input_path)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("src.dataset_generation.translator.TranslationService")
    def test_augment_creates_text_column(self, mock_service_cls):
        """
        FR-DATA-08: Augmented dataset shall have a 'text' column.

        Given: A dataset without 'text' column
        When: Running `irca dataset augment`
        Then: Output dataset has 'text' column ready for finetuning
        """
        # Mock translation to return prefixed text
        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = lambda x: [f"[FR] {s}" for s in x]
        mock_service_cls.return_value = mock_translator

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment", "-i", self.input_path, "-o", self.output_path, "-l", "fr", "-r", "1.0"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

        # Load output and verify
        output_ds = datasets.load_from_disk(self.output_path)
        self.assertIn("text", output_ds.column_names)
        self.assertEqual(len(output_ds), self.n_samples)

    @patch("src.dataset_generation.translator.TranslationService")
    def test_augment_preserves_original_columns(self, mock_service_cls):
        """
        FR-DATA-08: Augmentation shall preserve original dataset columns.

        Given: A dataset with columns ['corrected_agent_trace', 'user_query']
        When: Running augmentation
        Then: Output has all original columns plus 'text'
        """
        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = lambda x: x
        mock_service_cls.return_value = mock_translator

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment", "-i", self.input_path, "-o", self.output_path, "-l", "fr", "-r", "0.0"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        output_ds = datasets.load_from_disk(self.output_path)
        self.assertIn("corrected_agent_trace", output_ds.column_names)
        self.assertIn("user_query", output_ds.column_names)
        self.assertIn("text", output_ds.column_names)

    @patch("src.dataset_generation.translator.TranslationService")
    def test_augment_respects_ratio(self, mock_service_cls):
        """
        FR-DATA-04: Stochastic configuration with configurable ratio.

        Given: A dataset with 20 samples and ratio=0.5
        When: Running augmentation with seed=42
        Then: Approximately 50% of samples are translated (deterministic with seed)
        """

        # Mock translation that adds a clear marker
        def mock_translate(texts):
            return [f"[TRANSLATED] {t}" for t in texts]

        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = mock_translate
        mock_service_cls.return_value = mock_translator

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment", "-i", self.input_path, "-o", self.output_path, "-l", "fr", "-r", "0.5", "--seed", "42"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0, f"CLI failed: {result.output}")

        # With ratio=0.5 and 20 samples, expect ~10 translations
        output_ds = datasets.load_from_disk(self.output_path)

        # Count samples with translated content
        translated_count = sum(1 for row in output_ds if "[TRANSLATED]" in row["text"])

        # Should be between 5 and 15 (reasonable range for 50% with some variance)
        # With seed=42 and 20 samples at 50%, we expect deterministic behavior
        self.assertGreaterEqual(translated_count, 5, "At least 25% should be translated")
        self.assertLessEqual(translated_count, 15, "At most 75% should be translated")


class TestMarkdownLinkPreservation(unittest.TestCase):
    """
    Integration tests for FR-DATA-09: Markdown Link Preservation.

    Verifies that markdown link URLs are preserved during translation
    while link text is translated.
    """

    def setUp(self):
        """Create test directory and dataset with markdown links."""
        self.test_dir = tempfile.mkdtemp()
        self.input_path = os.path.join(self.test_dir, "input_dataset")
        self.output_path = os.path.join(self.test_dir, "output_dataset")

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_dataset_with_links(self, final_answer: str, n_samples: int = 10):
        """Helper to create a dataset with specific final answer content."""
        sample_parts = {
            "system_instructions": "You are an AI assistant.",
            "example": "Example<|wait|>",
            "available_functions_json": "[]",
            "user_query": "Show me the data",
            "assistant_completion": f"<thought>Processing.</thought>\n\n### FINAL ANSWER\n{final_answer}",
        }
        sample_prompt = build_full_prompt(sample_parts)

        data = {"corrected_agent_trace": [[{"value": sample_prompt}]] * n_samples}
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.input_path)
        return ds

    @patch("src.dataset_generation.translator.TranslationService")
    def test_output_uuid_links_preserved(self, mock_service_cls):
        """
        FR-DATA-09: Output[uuid] links shall be preserved.

        Given: Final answer with "[here](Output[abc123])"
        When: Translating to French
        Then: URL "Output[abc123]" is unchanged, text "here" is translated
        """
        self._create_dataset_with_links("Check the results [here](Output[abc123def456]).")

        # Mock translation that would break links if not protected
        def mock_translate(texts):
            result = []
            for t in texts:
                # Simulate what a real translator might do
                t = t.replace("Check the results", "Vérifiez les résultats")
                t = t.replace("here", "ici")
                t = t.replace("Output", "Sortie")  # This should NOT happen to URLs
                result.append(t)
            return result

        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = mock_translate
        mock_service_cls.return_value = mock_translator

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment", "-i", self.input_path, "-o", self.output_path, "-l", "fr", "-r", "1.0", "--seed", "42"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        output_ds = datasets.load_from_disk(self.output_path)
        final_answer_section = output_ds[0]["text"]

        # URL must be preserved exactly
        self.assertIn("Output[abc123def456]", final_answer_section, "Output[uuid] reference must be preserved")
        # Should NOT contain translated URL
        self.assertNotIn("Sortie[", final_answer_section, "URL should not be translated")

    @patch("src.dataset_generation.translator.TranslationService")
    def test_multiple_links_all_preserved(self, mock_service_cls):
        """
        FR-DATA-09: Multiple markdown links shall all be preserved.

        Given: Final answer with multiple Output references
        When: Translating
        Then: All URLs are preserved
        """
        self._create_dataset_with_links("See [weather](Output[weather123]) and [forecast](Output[forecast456]).")

        def mock_translate(texts):
            return [f"[FR] {t}" for t in texts]

        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = mock_translate
        mock_service_cls.return_value = mock_translator

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment", "-i", self.input_path, "-o", self.output_path, "-l", "fr", "-r", "1.0"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        output_ds = datasets.load_from_disk(self.output_path)
        text = output_ds[0]["text"]

        # Both URLs must be preserved
        self.assertIn("Output[weather123]", text)
        self.assertIn("Output[forecast456]", text)

    @patch("src.dataset_generation.translator.TranslationService")
    def test_link_text_is_translated(self, mock_service_cls):
        """
        FR-DATA-09: Link text (description) shall be translated.

        Given: "[click here](Output[id])"
        When: Translating to French
        Then: "click here" becomes French, but URL is unchanged
        """
        self._create_dataset_with_links("Results are [available here](Output[results789]).")

        def mock_translate(texts):
            result = []
            for t in texts:
                t = t.replace("available here", "disponible ici")
                t = t.replace("Results are", "Les résultats sont")
                result.append(t)
            return result

        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = mock_translate
        mock_service_cls.return_value = mock_translator

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment", "-i", self.input_path, "-o", self.output_path, "-l", "fr", "-r", "1.0"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        output_ds = datasets.load_from_disk(self.output_path)
        text = output_ds[0]["text"]

        # Link text should be translated
        self.assertIn("disponible ici", text, "Link text should be translated")
        # URL must be preserved
        self.assertIn("Output[results789]", text, "URL must be preserved")

    @patch("src.dataset_generation.translator.TranslationService")
    def test_http_links_preserved(self, mock_service_cls):
        """
        FR-DATA-09: HTTP/HTTPS URLs shall be preserved.

        Given: Final answer with http URLs
        When: Translating
        Then: Full URL is unchanged
        """
        self._create_dataset_with_links("See documentation at [docs](https://example.com/api/v1).")

        def mock_translate(texts):
            return [t.replace("See documentation at", "Voir la documentation à") for t in texts]

        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = mock_translate
        mock_service_cls.return_value = mock_translator

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment", "-i", self.input_path, "-o", self.output_path, "-l", "fr", "-r", "1.0"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        output_ds = datasets.load_from_disk(self.output_path)
        text = output_ds[0]["text"]

        self.assertIn("https://example.com/api/v1", text)

    @patch("src.dataset_generation.translator.TranslationService")
    def test_complex_uuid_patterns_preserved(self, mock_service_cls):
        """
        FR-DATA-09: Complex Output reference patterns shall be preserved.

        Given: Output references with array indices like Output[id][0]["name"]
        When: Translating
        Then: Full reference path is preserved
        """
        self._create_dataset_with_links('The name is [Coffee House](Output[cLFHPMNtdm8vcsJFwjqgY8][0]["name"]).')

        def mock_translate(texts):
            return [t.replace("The name is", "Le nom est") for t in texts]

        mock_translator = MagicMock()
        mock_translator.translate_batch.side_effect = mock_translate
        mock_service_cls.return_value = mock_translator

        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["augment", "-i", self.input_path, "-o", self.output_path, "-l", "fr", "-r", "1.0"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)

        output_ds = datasets.load_from_disk(self.output_path)
        text = output_ds[0]["text"]

        self.assertIn('Output[cLFHPMNtdm8vcsJFwjqgY8][0]["name"]', text)


class TestDatasetInspect(unittest.TestCase):
    """
    Integration tests for FR-DATA-10: Dataset Inspection.
    """

    def setUp(self):
        """Create test directory and sample dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.test_dir, "test_dataset")

        # Create a simple dataset
        data = {
            "text": ["Sample text 1", "Sample text 2"],
            "label": [0, 1],
        }
        ds = datasets.Dataset.from_dict(data)
        ds.save_to_disk(self.dataset_path)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_inspect_shows_columns(self):
        """
        FR-DATA-10: Inspect command shall display column names.

        Given: A dataset with columns ['text', 'label']
        When: Running `irca dataset inspect`
        Then: Output shows column names
        """
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["inspect", "-p", self.dataset_path, "-n", "1"],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("text", result.output)
        self.assertIn("label", result.output)
        self.assertIn("2 examples", result.output)

    def test_inspect_indicates_finetuning_readiness(self):
        """
        FR-DATA-10: Inspect shall indicate if dataset is ready for finetuning.

        Given: A dataset with 'text' column
        When: Running inspect
        Then: Shows "ready for finetuning" indicator
        """
        runner = CliRunner()
        result = runner.invoke(
            dataset,
            ["inspect", "-p", self.dataset_path],
            obj={"verbose": False},
            catch_exceptions=False,
        )

        self.assertEqual(result.exit_code, 0)
        self.assertIn("text", result.output)
        self.assertIn("ready for finetuning", result.output.lower())


if __name__ == "__main__":
    unittest.main()
