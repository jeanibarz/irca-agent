import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import datasets

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from dataset_generation.translator import TranslationService
from finetuning.model_finetuning import augment_dataset


class TestAugmentation(unittest.TestCase):
    def test_translation_service_init(self):
        service = TranslationService("fr")
        self.assertEqual(service.model_name, "Helsinki-NLP/opus-mt-en-fr")

    @patch("dataset_generation.translator.AutoTokenizer.from_pretrained")
    @patch("dataset_generation.translator.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("dataset_generation.translator.torch.cuda.is_available", return_value=False)
    def test_translation_service_batch(self, mock_cuda, mock_model, mock_tok):
        # Mock tokens
        mock_tokenizer_instance = mock_tok.return_value
        mock_tokenizer_instance.return_value = MagicMock()  # mock encoded inputs
        mock_tokenizer_instance.batch_decode.return_value = ["Bonjour", "Monde"]

        # Mock model instance and its .to() method
        mock_model_instance = mock_model.return_value
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.generate.return_value = ["outputs"]

        service = TranslationService("fr")
        res = service.translate_batch(["Hello", "World"])

        self.assertEqual(res, ["Bonjour", "Monde"])
        mock_model.return_value.generate.assert_called()

    @patch("dataset_generation.translator.TranslationService")
    @patch("core.prompt_builder.parse_corrected_agent_trace")
    @patch("core.prompt_builder.build_full_prompt")
    def test_augment_dataset_logic(self, mock_build, mock_parse, mock_service_cls):
        # Setup mocks
        mock_translator = mock_service_cls.return_value
        # Translate appends "Trans_"
        mock_translator.translate_batch.side_effect = lambda x: [f"Trans_{s}" for s in x]

        # Mock parser to return a fixed structure
        mock_parse.return_value = {
            "user_query": "Query",
            "assistant_completion": "Thought: ...\n\n### FINAL ANSWER\nAnswer",
            "system_instructions": "Sys",
            "available_functions_json": "[]",
            "example": "",
        }

        mock_build.return_value = "NEW_PROMPT_STRING"

        # Setup fake dataset
        # We simulate the structure used in model_finetuning
        # "corrected_agent_trace" is a list of dicts
        data = {"corrected_agent_trace": [[{"value": "Original Prompt"}]] * 100}
        original_ds = datasets.Dataset.from_dict(data)

        config = {
            "augment_enabled": True,
            "augment_languages": ["fr"],
            "augment_ratio": 0.1,  # Should augment 10 examples
        }

        # Run augmentation
        augmented_ds = augment_dataset(original_ds, config)

        # 100 original -> 100 diversified (replacement)
        self.assertEqual(len(augmented_ds), 100)

        # Verify mocks were called
        self.assertTrue(mock_translator.translate_batch.called)
        self.assertTrue(mock_parse.called)
        self.assertTrue(mock_build.called)

    def test_augment_disabled(self):
        original_ds = datasets.Dataset.from_dict({"a": [1]})
        config = {"augment_enabled": False}
        res = augment_dataset(original_ds, config)
        self.assertEqual(len(res), 1)
        self.assertEqual(res, original_ds)


if __name__ == "__main__":
    unittest.main()
