"""
Unit tests for completion marker detection.

Tests:
- FR-VAL-003: Completion marker detection
- FM-8: Completion marker detection fails for some models
"""

import pytest

from src.core.constants import (
    ALL_COMPLETION_MARKERS,
    IRCA_COMPLETION_MARKERS,
    LLAMA3_COMPLETION_MARKERS,
    MISTRAL_COMPLETION_MARKERS,
    MODEL_FAMILY_IRCA,
    MODEL_FAMILY_LLAMA3,
    MODEL_FAMILY_MISTRAL,
    MODEL_FAMILY_QWEN,
    MODEL_FAMILY_UNKNOWN,
    QWEN_COMPLETION_MARKERS,
)
from src.validation.completion import (
    CompletionDetectionResult,
    detect_completion_marker,
    get_completion_markers_for_model,
    get_model_family,
    validate_completion_detection,
)


# ============================================
# Sample Formatted Texts
# ============================================


@pytest.fixture
def irca_formatted_text() -> str:
    """Text formatted with IRCA markers (raw format)."""
    return """### INSTRUCTIONS
You are helpful.

### FUNCTIONS AVAILABLE
[]

### USER QUERY
Hello

### ITERATIVE RESOLUTION CYCLE
Thought: I should say hello.

### FINAL ANSWER
Hello there!"""


@pytest.fixture
def qwen_formatted_text() -> str:
    """Text formatted with Qwen/ChatML markers."""
    return """<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hello there!<|im_end|>"""


@pytest.fixture
def mistral_formatted_text() -> str:
    """Text formatted with Mistral markers."""
    return """<s>[INST] You are helpful.

Hello [/INST] Hello there!</s>"""


@pytest.fixture
def llama3_formatted_text() -> str:
    """Text formatted with Llama3 markers."""
    return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>
Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Hello there!<|eot_id|>"""


# ============================================
# Test Completion Detection (FR-VAL-003, FM-8)
# ============================================


class TestCompletionDetection:
    """Tests for completion marker detection."""

    def test_ut_val_010_detect_irca_completion_boundary(self, irca_formatted_text: str):
        """UT-VAL-010: Detect IRCA format completion boundary."""
        result = detect_completion_marker(irca_formatted_text)

        assert result.found
        assert result.marker == "### ITERATIVE RESOLUTION CYCLE"
        assert result.model_family == MODEL_FAMILY_IRCA
        assert result.position > 0

    def test_ut_val_011_detect_qwen_completion_boundary(self, qwen_formatted_text: str):
        """UT-VAL-011: Detect Qwen/ChatML completion boundary."""
        result = detect_completion_marker(qwen_formatted_text)

        assert result.found
        assert "<|im_start|>assistant" in result.marker
        assert result.model_family == MODEL_FAMILY_QWEN

    def test_ut_val_012_detect_mistral_completion_boundary(self, mistral_formatted_text: str):
        """UT-VAL-012: Detect Mistral completion boundary."""
        result = detect_completion_marker(mistral_formatted_text)

        assert result.found
        assert result.marker == "[/INST]"
        assert result.model_family == MODEL_FAMILY_MISTRAL

    def test_ut_val_013_detect_llama3_completion_boundary(self, llama3_formatted_text: str):
        """UT-VAL-013: Detect Llama3 completion boundary."""
        result = detect_completion_marker(llama3_formatted_text)

        assert result.found
        assert "<|start_header_id|>assistant" in result.marker
        assert result.model_family == MODEL_FAMILY_LLAMA3

    def test_ut_val_014_return_0_when_no_marker_found(self):
        """UT-VAL-014: Return appropriate result when no completion marker found."""
        text_without_markers = "This is just plain text without any markers."
        result = detect_completion_marker(text_without_markers)

        assert not result.found
        assert result.marker is None
        assert result.position == -1
        assert result.token_boundary == 0

    def test_custom_markers(self, irca_formatted_text: str):
        """Test detection with custom markers."""
        # Only look for Qwen markers (won't find any)
        result = detect_completion_marker(irca_formatted_text, markers=QWEN_COMPLETION_MARKERS)
        assert not result.found

        # Look for IRCA markers (will find)
        result = detect_completion_marker(irca_formatted_text, markers=IRCA_COMPLETION_MARKERS)
        assert result.found

    def test_first_marker_wins(self):
        """Test that the first marker in the text is used."""
        # Text with multiple marker types
        text = """### ITERATIVE RESOLUTION CYCLE
Some content
<|im_start|>assistant
More content"""

        result = detect_completion_marker(text)
        assert result.found
        assert "ITERATIVE" in result.marker  # IRCA markers come first in the list


class TestValidateCompletionDetection:
    """Tests for batch completion validation."""

    def test_all_texts_valid(self, irca_formatted_text: str, qwen_formatted_text: str):
        """Test validation passes when all texts have valid markers."""
        texts = [irca_formatted_text, qwen_formatted_text]
        passed, issues = validate_completion_detection(texts)

        assert passed
        assert len(issues) == 0

    def test_detect_missing_markers(self):
        """Test validation catches texts without markers."""
        texts = ["No markers here", "Still no markers"]
        passed, issues = validate_completion_detection(texts)

        assert not passed
        assert len(issues) == 2
        for issue in issues:
            assert "No completion marker found" in issue["error"]

    def test_family_mismatch(self, irca_formatted_text: str):
        """Test validation catches model family mismatches."""
        texts = [irca_formatted_text]
        passed, issues = validate_completion_detection(texts, expected_family=MODEL_FAMILY_QWEN)

        assert not passed
        assert len(issues) == 1
        assert "mismatch" in issues[0]["error"].lower()


class TestModelFamilyDetection:
    """Tests for model family detection from tokenizer."""

    def test_unknown_family_for_mock(self):
        """Test that mock/basic tokenizer returns unknown family."""
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab = {}
        mock_tokenizer.added_tokens_encoder = {}
        mock_tokenizer.special_tokens_map = {}
        # Ensure hasattr returns False for inst_token
        del mock_tokenizer.inst_token

        family = get_model_family(mock_tokenizer)
        assert family == MODEL_FAMILY_UNKNOWN

    def test_qwen_family_detection(self):
        """Test Qwen family detection from special tokens."""
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab = {"<|im_start|>": 151644}
        mock_tokenizer.added_tokens_encoder = {}
        mock_tokenizer.special_tokens_map = {}

        family = get_model_family(mock_tokenizer)
        assert family == MODEL_FAMILY_QWEN

    def test_mistral_family_detection(self):
        """Test Mistral family detection from special tokens."""
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab = {"[INST]": 3}
        mock_tokenizer.added_tokens_encoder = {}
        mock_tokenizer.special_tokens_map = {}

        family = get_model_family(mock_tokenizer)
        assert family == MODEL_FAMILY_MISTRAL

    def test_llama3_family_detection(self):
        """Test Llama3 family detection from special tokens."""
        from unittest.mock import MagicMock

        mock_tokenizer = MagicMock()
        mock_tokenizer.vocab = {"<|start_header_id|>": 128006}
        mock_tokenizer.added_tokens_encoder = {}
        mock_tokenizer.special_tokens_map = {}

        family = get_model_family(mock_tokenizer)
        assert family == MODEL_FAMILY_LLAMA3


class TestCompletionMarkersForModel:
    """Tests for getting markers based on model family."""

    def test_get_qwen_markers(self):
        """Test getting markers for Qwen family."""
        markers = get_completion_markers_for_model(model_family=MODEL_FAMILY_QWEN)
        assert markers == QWEN_COMPLETION_MARKERS

    def test_get_mistral_markers(self):
        """Test getting markers for Mistral family."""
        markers = get_completion_markers_for_model(model_family=MODEL_FAMILY_MISTRAL)
        assert markers == MISTRAL_COMPLETION_MARKERS

    def test_get_all_markers_for_unknown(self):
        """Test getting all markers for unknown family."""
        markers = get_completion_markers_for_model(model_family=MODEL_FAMILY_UNKNOWN)
        assert markers == ALL_COMPLETION_MARKERS


# ============================================
# Test Completion Markers Constants
# ============================================


class TestCompletionMarkerConstants:
    """Tests for completion marker constants."""

    def test_all_markers_contains_families(self):
        """Test that ALL_COMPLETION_MARKERS includes all family markers."""
        for marker in QWEN_COMPLETION_MARKERS:
            assert marker in ALL_COMPLETION_MARKERS
        for marker in MISTRAL_COMPLETION_MARKERS:
            assert marker in ALL_COMPLETION_MARKERS
        for marker in IRCA_COMPLETION_MARKERS:
            assert marker in ALL_COMPLETION_MARKERS

    def test_markers_are_unique(self):
        """Test that all markers are unique."""
        # Note: Duplicates between lists might exist by design
        for family_markers in [
            QWEN_COMPLETION_MARKERS,
            MISTRAL_COMPLETION_MARKERS,
            LLAMA3_COMPLETION_MARKERS,
            IRCA_COMPLETION_MARKERS,
        ]:
            assert len(family_markers) == len(set(family_markers))

    def test_markers_are_nonempty_strings(self):
        """Test that all markers are non-empty strings."""
        for marker in ALL_COMPLETION_MARKERS:
            assert isinstance(marker, str)
            assert len(marker) > 0
