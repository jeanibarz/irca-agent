"""
Augmentation step implementations for dataset formatting.

Steps are organized by stage:
- Structural: Operate on parsed components before formatting
- Formatting: Operate during the formatting process
- Presentation: Operate on final formatted text

Implements FR-DATA-25: Staged Augmentation.
"""

import random
import re
from abc import ABC, abstractmethod
from typing import Any

from src.formatting.config import StepConfig


class AugmentationStep(ABC):
    """Base class for augmentation steps."""

    def __init__(self, config: StepConfig, rng: random.Random):
        self.config = config
        self.rng = rng
        self._applied_name: str | None = None

    @abstractmethod
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply augmentation to data."""
        pass

    def get_name(self) -> str:
        """Get the name of this augmentation for metadata."""
        return self._applied_name or self.config.type

    @property
    def stage(self) -> str:
        """Get the stage this step belongs to."""
        raise NotImplementedError("Subclasses must define stage")


# =============================================================================
# STRUCTURAL STEPS (operate on parsed components before formatting)
# =============================================================================


class TranslateStep(AugmentationStep):
    """
    Translate user query and/or final answer.

    Operates on parsed components (structural stage).

    Params:
        languages: List of target languages (e.g., ["fr", "es", "de"])
        weights: Optional weights for language selection
        targets: What to translate - ["query", "answer"] or subset
    """

    stage = "structural"

    def __init__(self, config: StepConfig, rng: random.Random):
        super().__init__(config, rng)
        params = config.params or {}
        self.languages = params.get("languages", ["fr"])
        self.weights = params.get("weights")
        self.targets = params.get("targets", ["query", "answer"])
        self._translator = None
        self._current_lang = None

    def _get_translator(self, lang: str) -> Any:
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

    def _translate_preserving_links(self, text: str, translator: Any) -> str:
        """Translate text while preserving markdown link URLs."""
        pattern = r"(\[[^\]]*\]\()([^\)]+)(\))"
        links = []
        for match in re.finditer(pattern, text):
            links.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "prefix": match.group(1),
                    "url": match.group(2),
                }
            )

        if not links:
            result = translator.translate_batch([text])
            return result[0] if result else text

        segments = []
        last_end = 0

        for link in links:
            if link["start"] > last_end:
                segments.append({"type": "text", "content": text[last_end : link["start"]]})

            link_text_match = re.match(r"\[([^\]]*)\]", link["prefix"])
            if link_text_match:
                segments.append(
                    {
                        "type": "link",
                        "text": link_text_match.group(1),
                        "url": link["url"],
                    }
                )

            last_end = link["end"]

        if last_end < len(text):
            segments.append({"type": "text", "content": text[last_end:]})

        texts_to_translate = [s["content"] for s in segments if s["type"] == "text"]
        link_texts = [s["text"] for s in segments if s["type"] == "link"]

        translated_texts = translator.translate_batch(texts_to_translate) if texts_to_translate else []
        translated_links = translator.translate_batch(link_texts) if link_texts else []

        result = []
        text_idx = 0
        link_idx = 0

        for segment in segments:
            if segment["type"] == "text":
                result.append(translated_texts[text_idx])
                text_idx += 1
            else:
                result.append(f"[{translated_links[link_idx]}]({segment['url']})")
                link_idx += 1

        return "".join(result)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply translation to parsed components."""
        lang = self._select_language()
        self._applied_name = f"translate:{lang}"
        translator = self._get_translator(lang)

        result = data.copy()

        if "query" in self.targets and result.get("user_query"):
            result["user_query"] = self._translate_preserving_links(result["user_query"], translator)

        if "answer" in self.targets and result.get("final_answer"):
            result["final_answer"] = self._translate_preserving_links(result["final_answer"], translator)

        return result


# =============================================================================
# FORMATTING STEPS (operate during formatting process)
# =============================================================================


class ShuffleFunctionsStep(AugmentationStep):
    """
    Shuffle the order of functions in the available_functions list.

    Operates during formatting stage on the parsed functions list.
    """

    stage = "formatting"

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Shuffle functions in the parsed data."""
        self._applied_name = "shuffle_functions"
        result = data.copy()

        functions = result.get("available_functions")
        if functions and isinstance(functions, list):
            functions = functions.copy()
            self.rng.shuffle(functions)
            result["available_functions"] = functions

        return result


class IndentFunctionsStep(AugmentationStep):
    """
    Randomize JSON indentation for functions.

    Operates during formatting stage.

    Params:
        choices: List of indent values (e.g., [null, 2, 4])
    """

    stage = "formatting"

    def __init__(self, config: StepConfig, rng: random.Random):
        super().__init__(config, rng)
        params = config.params or {}
        self.choices = params.get("choices", [None, 2, 4])

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Set JSON indent preference in parsed data."""
        indent = self.rng.choice(self.choices)
        self._applied_name = f"indent_functions:{indent}"
        result = data.copy()
        result["_json_indent"] = indent
        return result


# =============================================================================
# PRESENTATION STEPS (operate on final formatted text)
# =============================================================================


class FormatVariationStep(AugmentationStep):
    """
    Randomize instruction markers and whitespace in formatted text.

    Operates on final text (presentation stage).

    WARNING (ADR-006): This augmentation replaces standard IRCA markers with
    non-standard variants. This BREAKS:
    - Chat template conversion in finetuning (parse_corrected_agent_trace fails)
    - Completion extraction in diversity evaluation

    Only use if:
    - Training without chat templates (not recommended for new models)
    - Testing parser robustness (experimental)

    Recommended alternatives:
    - translate: Adds linguistic diversity
    - shuffle_functions: Changes function order
    - indent_functions: Varies JSON formatting
    - newline_variation: Changes \\n vs \\r\\n (less impactful)

    See: docs/adrs/ADR-006-chat-template-consistency.md
    """

    stage = "presentation"

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply format variations to text."""
        self._applied_name = "format_variation"
        result = data.copy()

        text = result.get("text", "")
        if not text:
            return result

        format_style = self.rng.choice([1, 2])

        if format_style == 2:
            replacements = {
                "### INSTRUCTIONS": "",
                "### FUNCTIONS AVAILABLE": "<|FUNCTIONS AVAILABLE|>",
                "### USER QUERY": "<|USER QUERY|>",
                "### ITERATIVE RESOLUTION CYCLE": "<|ITERATIVE RESOLUTION CYCLE|>",
                "### FINAL ANSWER": "<|FINAL ANSWER|>",
            }
            for old, new in replacements.items():
                newline_count = self.rng.choice(["", "\n", "\n\n"])
                text = text.replace(old, newline_count + new)

        result["text"] = text
        return result


class NewlineVariationStep(AugmentationStep):
    """
    Randomize newline characters (\\n vs \\r\\n).

    Operates on final text (presentation stage).
    """

    stage = "presentation"

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply newline variation to text."""
        self._applied_name = "newline_variation"
        result = data.copy()

        text = result.get("text", "")
        if not text:
            return result

        newline_choice = self.rng.choice(["\n", "\r\n"])
        result["text"] = text.replace("\n", newline_choice)
        return result


# =============================================================================
# STEP REGISTRY
# =============================================================================


STEP_REGISTRY: dict[str, type[AugmentationStep]] = {
    # Structural
    "translate": TranslateStep,
    # Formatting
    "shuffle_functions": ShuffleFunctionsStep,
    "indent_functions": IndentFunctionsStep,
    # Presentation
    "newline_variation": NewlineVariationStep,
    # NOTE: format_variation removed - see ADR-006 (breaks chat template parsing)
}


def create_step(config: StepConfig, rng: random.Random) -> AugmentationStep:
    """
    Create an augmentation step from config.

    Args:
        config: Step configuration
        rng: Random number generator

    Returns:
        Instantiated augmentation step

    Raises:
        ValueError: If step type is unknown
    """
    step_class = STEP_REGISTRY.get(config.type)
    if step_class is None:
        raise ValueError(f"Unknown step type: {config.type}. Available: {list(STEP_REGISTRY.keys())}")
    return step_class(config, rng)


def get_step_stage(step_type: str) -> str:
    """Get the stage for a step type."""
    step_class = STEP_REGISTRY.get(step_type)
    if step_class is None:
        raise ValueError(f"Unknown step type: {step_type}")
    return step_class.stage
