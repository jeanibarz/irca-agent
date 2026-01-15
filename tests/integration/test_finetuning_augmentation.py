import os
import sys
import unittest
from unittest.mock import patch

import datasets

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from finetuning.model_finetuning import augment_dataset


class TestFinetuningAugmentationIntegration(unittest.TestCase):
    """
    Integration tests for dataset augmentation.
    Verifies deterministic behavior of augmentation probabilities.
    """

    def setUp(self):
        # Create a synthetic dataset
        # We need a large enough dataset for ratios to coincide with integer counts cleanly
        self.n_samples = 100
        self.dataset = datasets.Dataset.from_dict(
            {
                "corrected_agent_trace": [
                    [{"value": f"Query {i}\n\n### FINAL ANSWER\nAnswer {i}"}] for i in range(self.n_samples)
                ]
            }
        )

    @patch("dataset_generation.translator.TranslationService")
    @patch("core.prompt_builder.parse_corrected_agent_trace")
    @patch("core.prompt_builder.build_full_prompt")
    def test_augmentation_determinism(self, mock_build, mock_parse, mock_service_cls):
        """
        Verify that augmentation adds the exact expected number of examples
        based on ratio, regardless of randomness (due to fixed seeds).
        """
        # --- Setup Mocks ---
        mock_translator = mock_service_cls.return_value
        mock_translator.translate_batch.side_effect = lambda x: [f"Translated_{s}" for s in x]

        mock_parse.side_effect = lambda x: {
            "user_query": "Query",
            "assistant_completion": "Trace\n\n### FINAL ANSWER\nAnswer",
            "system_instructions": "Sys",
            "available_functions_json": "[]",
            "example": "",
        }
        mock_build.return_value = "NEW_PROMPT_VAL"

        # --- Scenario 1: Ratio 0.0 (No Augmentation) ---
        config_zero = {"augment_enabled": True, "augment_languages": ["fr"], "augment_ratio": 0.0}
        ds_zero = augment_dataset(self.dataset, config_zero)
        self.assertEqual(len(ds_zero), self.n_samples, "Ratio 0 should not change size")

        # --- Scenario 2: Ratio 1.0 (Same Size, but all translated) ---
        config_full = {"augment_enabled": True, "augment_languages": ["fr"], "augment_ratio": 1.0}
        ds_full = augment_dataset(self.dataset, config_full)
        self.assertEqual(len(ds_full), self.n_samples, "Ratio 1.0 should keep same size (diversification)")

        # --- Scenario 3: Ratio 1.0 with 2 Languages (Same Size) ---
        config_multi = {"augment_enabled": True, "augment_languages": ["fr", "es"], "augment_ratio": 1.0}
        ds_multi = augment_dataset(self.dataset, config_multi)
        self.assertEqual(len(ds_multi), self.n_samples, "Ratio 1.0 with 2 langs should keep same size")

        # Verify both langs were initialized
        calls = mock_service_cls.call_args_list
        langs_called = [c.kwargs["target_lang"] for c in calls]
        self.assertIn("fr", langs_called)
        self.assertIn("es", langs_called)

    def test_augmentation_disabled(self):
        config = {"augment_enabled": False}
        ds = augment_dataset(self.dataset, config)
        self.assertEqual(len(ds), self.n_samples)


if __name__ == "__main__":
    unittest.main()
