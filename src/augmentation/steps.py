"""
Augmentation step implementations.

Wraps existing augmentation functions and provides a unified interface.
"""

import json
import random
import re
from abc import ABC, abstractmethod
from typing import Any

from src.augmentation.config import PipelineStep


class AugmentationStep(ABC):
    """Base class for augmentation steps."""

    def __init__(self, step_config: PipelineStep, rng: random.Random):
        self.config = step_config
        self.rng = rng

    @abstractmethod
    def apply(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply augmentation to a sample."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the name of this augmentation for metadata."""
        pass


class TranslateStep(AugmentationStep):
    """
    Translation augmentation step.

    Translates user query and final answer while preserving reasoning traces.
    """

    def __init__(self, step_config: PipelineStep, rng: random.Random):
        super().__init__(step_config, rng)
        self._translator = None
        self._current_lang = None

        # Parse params
        params = step_config.params or {}
        self.languages = params.get("languages", ["fr"])
        self.weights = params.get("weights")

    def _get_translator(self, lang: str):
        """Lazy load translator for the target language."""
        from src.dataset_generation.translator import TranslationService

        if self._translator is None or self._current_lang != lang:
            self._translator = TranslationService(lang)
            self._current_lang = lang
        return self._translator

    def _select_language(self) -> str:
        """Select a language based on weights or uniform random."""
        if self.weights:
            return self.rng.choices(self.languages, weights=self.weights, k=1)[0]
        return self.rng.choice(self.languages)

    def _translate_preserving_links(self, text: str, translator) -> str:
        """
        Translate text while preserving markdown link URLs.

        Implements FR-DATA-09: Markdown Link Preservation.
        """
        # Pattern to match markdown links: [text](url)
        pattern = r"(\[[^\]]*\]\()([^\)]+)(\))"

        # Find all links and their positions
        links = []
        for match in re.finditer(pattern, text):
            links.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "prefix": match.group(1),  # "[text]("
                    "url": match.group(2),  # the URL
                    "suffix": match.group(3),  # ")"
                    "full": match.group(0),
                }
            )

        if not links:
            # No links, translate entire text
            result = translator.translate_batch([text])
            return result[0] if result else text

        # Split text into segments around links
        segments = []
        last_end = 0

        for link in links:
            # Add text before this link
            if link["start"] > last_end:
                segments.append({"type": "text", "content": text[last_end : link["start"]]})

            # Add the link (only translate the link text, preserve URL)
            link_text_match = re.match(r"\[([^\]]*)\]", link["prefix"])
            if link_text_match:
                link_text = link_text_match.group(1)
                segments.append({"type": "link", "text": link_text, "url": link["url"]})

            last_end = link["end"]

        # Add remaining text after last link
        if last_end < len(text):
            segments.append({"type": "text", "content": text[last_end:]})

        # Translate text segments
        texts_to_translate = [s["content"] for s in segments if s["type"] == "text"]
        if texts_to_translate:
            translated_texts = translator.translate_batch(texts_to_translate)
        else:
            translated_texts = []

        # Also translate link texts
        link_texts = [s["text"] for s in segments if s["type"] == "link"]
        if link_texts:
            translated_link_texts = translator.translate_batch(link_texts)
        else:
            translated_link_texts = []

        # Reassemble with translated text and original URLs
        result = []
        text_idx = 0
        link_idx = 0

        for segment in segments:
            if segment["type"] == "text":
                result.append(translated_texts[text_idx])
                text_idx += 1
            else:  # link
                translated_link_text = translated_link_texts[link_idx]
                result.append(f"[{translated_link_text}]({segment['url']})")
                link_idx += 1

        return "".join(result)

    def apply(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply translation to the sample."""
        from src.core.generation.constants import FINAL_ANSWER_PROMPT

        lang = self._select_language()
        self._selected_lang = lang  # Store for get_name()
        translator = self._get_translator(lang)

        text = sample.get("text", "")
        if not text:
            return sample

        # Find the FINAL ANSWER section
        final_answer_marker = FINAL_ANSWER_PROMPT
        if final_answer_marker in text:
            parts = text.split(final_answer_marker, 1)
            prefix = parts[0]
            final_answer = parts[1] if len(parts) > 1 else ""

            # Translate the final answer (preserving links)
            translated_answer = self._translate_preserving_links(final_answer, translator)

            # Reconstruct
            text = prefix + final_answer_marker + translated_answer

        return {**sample, "text": text}

    def get_name(self) -> str:
        lang = getattr(self, "_selected_lang", self.languages[0])
        return f"translate:{lang}"


class ShuffleFunctionsStep(AugmentationStep):
    """
    Shuffle functions augmentation step.

    Randomizes the order of functions in the JSON function list.
    Implements FR-GEN-04: Function Shuffling.
    """

    def apply(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Shuffle functions in the sample."""
        text = sample.get("text", "")
        if not text:
            return sample

        # Find JSON function array in the text
        # Look for patterns like: ### FUNCTIONS AVAILABLE\n[...]
        pattern = r"(\[[\s\S]*?\{[\s\S]*?\"name\"[\s\S]*?\}[\s\S]*?\])"

        def shuffle_json_match(match):
            try:
                json_str = match.group(1)
                functions = json.loads(json_str)
                if isinstance(functions, list):
                    self.rng.shuffle(functions)
                    return json.dumps(functions)
            except (json.JSONDecodeError, TypeError):
                pass
            return match.group(0)

        # Apply shuffling to first JSON array found (the functions list)
        text = re.sub(pattern, shuffle_json_match, text, count=1)

        return {**sample, "text": text}

    def get_name(self) -> str:
        return "shuffle_functions"


class FormatVariationStep(AugmentationStep):
    """
    Format variation augmentation step.

    Randomizes instruction markers and whitespace.
    Implements FR-GEN-05: Prompt Randomization.
    """

    def apply(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply format variations to the sample."""
        text = sample.get("text", "")
        if not text:
            return sample

        # Randomly choose a formatting style
        format_style = self.rng.choice([1, 2])

        if format_style == 2:
            replacements = {
                "### INSTRUCTIONS": "",
                "### FUNCTIONS AVAILABLE": "<|FUNCTIONS AVAILABLE|>",
                "### USER QUERY": "<|USER QUERY|>",
            }
            for old, new in replacements.items():
                newline_count = self.rng.choice(["", "\n", "\n\n"])
                text = text.replace(old, newline_count + new)

        return {**sample, "text": text}

    def get_name(self) -> str:
        return "format_variation"


class NewlineVariationStep(AugmentationStep):
    """
    Newline variation augmentation step.

    Randomly replaces newlines with either \\n or \\r\\n.
    """

    def apply(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply newline variations to the sample."""
        text = sample.get("text", "")
        if not text:
            return sample

        newline_choice = self.rng.choice(["\n", "\r\n"])
        text = text.replace("\n", newline_choice)

        return {**sample, "text": text}

    def get_name(self) -> str:
        return "newline_variation"


# Step registry
# NOTE: format_variation removed - see ADR-006 (breaks chat template parsing)
STEP_REGISTRY: dict[str, type[AugmentationStep]] = {
    "translate": TranslateStep,
    "shuffle_functions": ShuffleFunctionsStep,
    "newline_variation": NewlineVariationStep,
}


def create_step(step_config: PipelineStep, rng: random.Random) -> AugmentationStep:
    """
    Create an augmentation step from config.

    Args:
        step_config: Pipeline step configuration
        rng: Random number generator

    Returns:
        Instantiated augmentation step

    Raises:
        ValueError: If step type is unknown
    """
    step_class = STEP_REGISTRY.get(step_config.type)
    if step_class is None:
        raise ValueError(f"Unknown step type: {step_config.type}")
    return step_class(step_config, rng)
