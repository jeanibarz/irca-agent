"""
Unit tests for translation service and augmentation utilities.

Tests for FR-DATA-03, FR-DATA-09: Translation and link preservation.

Note: Full augmentation pipeline tests are in tests/integration/test_dataset_augmentation.py
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from dataset_generation.translator import TranslationService


class TestTranslationService(unittest.TestCase):
    """Unit tests for TranslationService class."""

    def test_translation_service_init_standard_lang(self):
        """Test initialization with standard language code."""
        service = TranslationService("fr")
        self.assertEqual(service.model_name, "Helsinki-NLP/opus-mt-en-fr")
        self.assertEqual(service.target_lang, "fr")
        self.assertIsNone(service.prefix)

    def test_translation_service_init_special_lang(self):
        """Test initialization with special language (group model)."""
        service = TranslationService("bn")  # Bengali uses en-inc group model
        self.assertEqual(service.model_name, "Helsinki-NLP/opus-mt-en-inc")
        self.assertEqual(service.prefix, ">>bn<<")

    def test_translation_service_skip_english(self):
        """Test that English target returns input unchanged."""
        service = TranslationService("en")
        result = service.translate_batch(["Hello", "World"])
        self.assertEqual(result, ["Hello", "World"])

    def test_translation_service_empty_input(self):
        """Test that empty input returns empty output."""
        service = TranslationService("fr")
        result = service.translate_batch([])
        self.assertEqual(result, [])

    @patch("dataset_generation.translator.AutoTokenizer.from_pretrained")
    @patch("dataset_generation.translator.AutoModelForSeq2SeqLM.from_pretrained")
    @patch("dataset_generation.translator.torch.cuda.is_available", return_value=False)
    def test_translation_service_batch(self, mock_cuda, mock_model, mock_tok):
        """Test batch translation with mocked model."""
        # Mock tokenizer
        mock_tokenizer_instance = mock_tok.return_value
        mock_tokenizer_instance.return_value = MagicMock()
        mock_tokenizer_instance.batch_decode.return_value = ["Bonjour", "Monde"]

        # Mock model
        mock_model_instance = mock_model.return_value
        mock_model_instance.to.return_value = mock_model_instance
        mock_model_instance.generate.return_value = ["outputs"]

        service = TranslationService("fr")
        res = service.translate_batch(["Hello", "World"])

        self.assertEqual(res, ["Bonjour", "Monde"])
        mock_model.return_value.generate.assert_called()

    def test_model_cache_is_class_level(self):
        """Test that model cache is shared across instances."""
        # Clear cache first
        TranslationService._model_cache.clear()
        TranslationService._tokenizer_cache.clear()

        # Verify cache is class-level (same dict object)
        service1 = TranslationService("fr")
        service2 = TranslationService("es")

        self.assertIs(service1._model_cache, service2._model_cache)
        self.assertIs(service1._tokenizer_cache, service2._tokenizer_cache)


class TestSpecialLanguageModels(unittest.TestCase):
    """Test special language model mappings."""

    def test_bengali_uses_indic_model(self):
        """Bengali should use the en-inc group model."""
        service = TranslationService("bn")
        self.assertIn("en-inc", service.model_name)

    def test_portuguese_uses_big_model(self):
        """Portuguese should use the tc-big model."""
        service = TranslationService("pt")
        self.assertIn("tc-big", service.model_name)

    def test_japanese_uses_jap_model(self):
        """Japanese should use the en-jap model."""
        service = TranslationService("ja")
        self.assertIn("en-jap", service.model_name)


if __name__ == "__main__":
    unittest.main()
