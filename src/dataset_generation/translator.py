import logging
from typing import Any, cast

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class TranslationService:
    """
    Handle translations using local HuggingFace models (Opus-MT).
    Includes a class-level cache to avoid reloading models in online training loops.
    """

    _model_cache: dict[str, Any] = {}
    _tokenizer_cache: dict[str, Any] = {}

    # Mapping for languages with non-standard Helsinki-NLP model names or group models
    # format: lang_code -> (model_name, prefix)
    _SPECIAL_MODELS: dict[str, tuple[str, str | None]] = {
        "bn": ("Helsinki-NLP/opus-mt-en-inc", ">>bn<<"),
        "te": ("Helsinki-NLP/opus-mt-en-inc", ">>te<<"),
        "ta": ("Helsinki-NLP/opus-mt-en-inc", ">>ta<<"),
        "pt": ("Helsinki-NLP/opus-mt-tc-big-en-pt", None),
        "tr": ("Helsinki-NLP/opus-mt-tc-big-en-tr", None),
        "ja": ("Helsinki-NLP/opus-mt-en-jap", None),
    }

    def __init__(
        self,
        target_lang: str = "fr",
        model_name_template: str = "Helsinki-NLP/opus-mt-en-{lang}",
        max_length: int = 512,
        device: str | None = None,
    ):
        self.target_lang = target_lang
        self.max_length = max_length

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine model name and prefix
        if target_lang in self._SPECIAL_MODELS:
            self.model_name, self.prefix = self._SPECIAL_MODELS[target_lang]
        else:
            self.model_name = model_name_template.format(lang=target_lang)
            self.prefix = None

    def load_model(self) -> None:
        """Load model and tokenizer if not already loaded into cache."""
        if self.model_name in self._model_cache:
            return

        logger.info(f"Loading translation model: {self.model_name} on {self.device}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            model = model.to(self.device)

            TranslationService._model_cache[self.model_name] = model
            TranslationService._tokenizer_cache[self.model_name] = tokenizer
        except Exception as e:
            logger.error(f"Failed to load translation model: {e}")
            raise RuntimeError(
                f"Could not load translation model {self.model_name}. Ensure 'sacremoses' is installed."
            ) from e

    @property
    def model(self) -> Any:
        self.load_model()
        return self._model_cache[self.model_name]

    @property
    def tokenizer(self) -> Any:
        self.load_model()
        return self._tokenizer_cache[self.model_name]

    def translate_batch(self, texts: list[str]) -> list[str]:
        """
        Translate a batch of texts from English to target language.
        If target_lang is 'en', returns the texts as is.
        """
        if self.target_lang == "en" or not texts:
            return texts

        # Ensure model/tokenizer are loaded via properties
        self.load_model()

        # Prepare texts with prefix if necessary (for group models like en-inc)
        inputs_texts = [f"{self.prefix} {t}" for t in texts] if self.prefix else texts

        # Handle splitting if batch is too huge?
        # For now rely on caller to pass reasonable batches (dataset.map default is 1000)
        # We might want to sub-batch if GPU OOMs, but text is small.

        try:
            inputs = self.tokenizer(
                inputs_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=self.max_length)

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return cast(list[str], decoded)
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Fallback to original text? Or raise?
            # Raising is safer to detect issues.
            raise
