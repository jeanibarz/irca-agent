"""
Completion marker detection and validation.

Validates that completion boundaries can be properly detected for training loss masking.
Different models use different markers to separate prompt from completion.

Implements:
- FR-VAL-003: Completion marker detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.constants import (
    ALL_COMPLETION_MARKERS,
    LLAMA3_COMPLETION_MARKERS,
    MISTRAL_COMPLETION_MARKERS,
    MODEL_FAMILY_CHATML,
    MODEL_FAMILY_IRCA,
    MODEL_FAMILY_LLAMA3,
    MODEL_FAMILY_MARKERS,
    MODEL_FAMILY_MISTRAL,
    MODEL_FAMILY_QWEN,
    MODEL_FAMILY_UNKNOWN,
    QWEN_COMPLETION_MARKERS,
)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class CompletionDetectionResult:
    """Result of completion marker detection."""

    found: bool
    marker: str | None
    position: int  # Character position, -1 if not found
    token_boundary: int  # Token index where completion starts, 0 if not found
    model_family: str | None

    def __str__(self) -> str:
        if self.found:
            return f"Found '{self.marker}' at position {self.position} (token {self.token_boundary})"
        return "No completion marker found"


def detect_completion_marker(
    text: str,
    markers: list[str] | None = None,
) -> CompletionDetectionResult:
    """
    Detect the completion marker in a formatted text sample.

    Searches for known markers that separate prompt from completion.
    Returns the first marker found.

    Args:
        text: Full formatted text (prompt + completion).
        markers: List of markers to search for. If None, uses all known markers.

    Returns:
        CompletionDetectionResult with marker info.

    Example:
        >>> result = detect_completion_marker("<|im_start|>assistant\\nHello")
        >>> result.found
        True
        >>> result.marker
        '<|im_start|>assistant\\n'
    """
    if markers is None:
        markers = ALL_COMPLETION_MARKERS

    for marker in markers:
        pos = text.find(marker)
        if pos >= 0:
            return CompletionDetectionResult(
                found=True,
                marker=marker,
                position=pos,
                token_boundary=0,  # Will be computed by caller if needed
                model_family=_marker_to_family(marker),
            )

    return CompletionDetectionResult(
        found=False,
        marker=None,
        position=-1,
        token_boundary=0,
        model_family=None,
    )


def _marker_to_family(marker: str) -> str:
    """Map a marker to its model family."""
    if marker in QWEN_COMPLETION_MARKERS:
        return MODEL_FAMILY_QWEN
    elif marker in MISTRAL_COMPLETION_MARKERS:
        return MODEL_FAMILY_MISTRAL
    elif marker in LLAMA3_COMPLETION_MARKERS:
        return MODEL_FAMILY_LLAMA3
    elif marker.startswith("### "):
        return MODEL_FAMILY_IRCA
    else:
        return MODEL_FAMILY_CHATML


def get_model_family(tokenizer: PreTrainedTokenizer) -> str:
    """
    Detect the model family from a tokenizer.

    Uses special tokens to identify the model's chat template format.

    Args:
        tokenizer: Model's tokenizer.

    Returns:
        Model family string (e.g., "qwen", "mistral", "llama3", "unknown").

    Example:
        >>> family = get_model_family(qwen_tokenizer)
        >>> family
        'qwen'
    """
    # Get all tokens
    vocab = getattr(tokenizer, "vocab", {}) or {}
    added_tokens = getattr(tokenizer, "added_tokens_encoder", {}) or {}
    all_tokens = {**vocab, **added_tokens}

    # Check for Qwen/ChatML markers
    if "<|im_start|>" in all_tokens or "im_start" in str(tokenizer.special_tokens_map):
        return MODEL_FAMILY_QWEN

    # Check for Llama3 markers
    if "<|start_header_id|>" in all_tokens:
        return MODEL_FAMILY_LLAMA3

    # Check for Mistral/Llama2 markers
    if "[INST]" in all_tokens or hasattr(tokenizer, "inst_token"):
        return MODEL_FAMILY_MISTRAL

    # Check for generic ChatML
    if "<|assistant|>" in all_tokens:
        return MODEL_FAMILY_CHATML

    return MODEL_FAMILY_UNKNOWN


def get_completion_markers_for_model(
    tokenizer: PreTrainedTokenizer | None = None,
    model_family: str | None = None,
) -> list[str]:
    """
    Get the appropriate completion markers for a model.

    Args:
        tokenizer: Model's tokenizer (used to detect family if not provided).
        model_family: Model family string (overrides tokenizer detection).

    Returns:
        List of completion markers for this model family.
    """
    if model_family is None and tokenizer is not None:
        model_family = get_model_family(tokenizer)

    if model_family and model_family in MODEL_FAMILY_MARKERS:
        return MODEL_FAMILY_MARKERS[model_family]

    # Default to all markers
    return ALL_COMPLETION_MARKERS


def validate_completion_detection(
    texts: list[str],
    expected_family: str | None = None,
    markers: list[str] | None = None,
) -> tuple[bool, list[dict]]:
    """
    Validate that completion markers can be detected in all texts.

    This catches cases where:
    - Augmentation corrupted the completion markers
    - Model family mismatch (wrong markers used)
    - Text doesn't follow expected format

    Args:
        texts: List of formatted text samples to validate.
        expected_family: Expected model family (optional, for stricter validation).
        markers: Specific markers to search for (optional).

    Returns:
        Tuple of (all_passed, list_of_issues).

    Example:
        >>> passed, issues = validate_completion_detection(dataset["text"])
        >>> if not passed:
        ...     for issue in issues:
        ...         print(f"Sample {issue['index']}: {issue['error']}")
    """
    issues = []

    for idx, text in enumerate(texts):
        result = detect_completion_marker(text, markers)

        if not result.found:
            issues.append(
                {
                    "index": idx,
                    "error": "No completion marker found",
                    "text_preview": text[:200] + "..." if len(text) > 200 else text,
                }
            )
        elif expected_family and result.model_family != expected_family:
            issues.append(
                {
                    "index": idx,
                    "error": f"Family mismatch: expected {expected_family}, got {result.model_family}",
                    "marker": result.marker,
                }
            )

    return len(issues) == 0, issues


def find_completion_token_boundary(
    text: str,
    tokenizer: PreTrainedTokenizer,
    markers: list[str] | None = None,
) -> int:
    """
    Find the token index where the completion starts.

    This is used to mask prompt tokens during loss computation.
    Returns 0 if no marker found (evaluate all tokens).

    Args:
        text: Full text containing prompt and completion.
        tokenizer: Tokenizer to use for encoding.
        markers: Completion markers to search for.

    Returns:
        Token index where completion starts (0 if not found).
    """
    result = detect_completion_marker(text, markers)

    if not result.found:
        return 0

    # Get text up to end of marker
    prompt_end = result.position + len(result.marker)
    prompt_text = text[:prompt_end]

    # Tokenize to find boundary
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=True)
    return len(prompt_tokens["input_ids"])


def compute_completion_token_boundaries(
    texts: list[str],
    tokenizer: PreTrainedTokenizer,
    markers: list[str] | None = None,
) -> list[int]:
    """
    Compute completion token boundaries for a list of texts.

    Args:
        texts: List of formatted texts.
        tokenizer: Tokenizer for encoding.
        markers: Completion markers to search for.

    Returns:
        List of token boundaries (one per text).
    """
    return [find_completion_token_boundary(text, tokenizer, markers) for text in texts]
