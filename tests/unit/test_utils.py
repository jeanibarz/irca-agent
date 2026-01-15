"""
Tests for Utility Functions

Tests the utility functions in core.utils.
Covers Requirements:
- FR-GEN-03 (Function Augmentation helper)
- FR-GEN-04 (Function Shuffling)
"""

import json

from core.utils import (
    extract_and_remove,
    format_generate_user_query,
    get_trainable_param_count,
    shuffle_json_functions,
)


class TestExtractAndRemove:
    """Tests for extract_and_remove function."""

    def test_extract_section_basic(self):
        """Extract a section between two markers."""
        text = "START content here END rest of text"
        extracted, remaining = extract_and_remove(
            start_marker="START",
            end_marker="END",
            full_prompt=text,
            include_start_marker=False,
            include_end_marker=False,
        )
        assert extracted == "content here"
        assert "rest of text" in remaining

    def test_extract_with_start_marker_included(self):
        """Extract with start marker included in result."""
        text = "### HEADER\ncontent\n### NEXT"
        extracted, _ = extract_and_remove(
            start_marker="### HEADER",
            end_marker="### NEXT",
            full_prompt=text,
            include_start_marker=True,
            include_end_marker=False,
        )
        assert extracted.startswith("### HEADER")
        assert "content" in extracted

    def test_extract_with_end_marker_included(self):
        """Extract with end marker included in result."""
        text = "START content END rest"
        extracted, _ = extract_and_remove(
            start_marker="START",
            end_marker="END",
            full_prompt=text,
            include_start_marker=False,
            include_end_marker=True,
        )
        assert extracted.endswith("END")

    def test_extract_with_no_start_marker(self):
        """Extract from beginning when start_marker is None."""
        text = "beginning content END rest"
        extracted, remaining = extract_and_remove(
            start_marker=None,
            end_marker="END",
            full_prompt=text,
            include_start_marker=False,
            include_end_marker=False,
        )
        assert extracted == "beginning content"

    def test_extract_with_no_end_marker(self):
        """Extract to end when end_marker is None."""
        text = "before START rest of the text"
        extracted, _ = extract_and_remove(
            start_marker="START",
            end_marker=None,
            full_prompt=text,
            include_start_marker=False,
            include_end_marker=False,
        )
        assert extracted == "rest of the text"

    def test_extract_start_marker_not_found(self):
        """Return empty string and original when start marker not found."""
        text = "some text without the expected marker"
        extracted, remaining = extract_and_remove(
            start_marker="NOT_FOUND",
            end_marker="END",
            full_prompt=text,
        )
        assert extracted == ""
        assert remaining == text

    def test_extract_end_marker_not_found(self):
        """Extract to end when end marker not found."""
        text = "START content but no end marker"
        extracted, remaining = extract_and_remove(
            start_marker="START",
            end_marker="NOT_FOUND",
            full_prompt=text,
            include_start_marker=False,
        )
        assert "content but no end marker" in extracted


class TestShuffleJsonFunctions:
    """
    Tests for shuffle_json_functions function.
    Req: FR-GEN-04 (Function Shuffling)
    """

    def test_shuffle_returns_valid_json(self, sample_functions_json):
        """Shuffled output is valid JSON."""
        result = shuffle_json_functions(sample_functions_json)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_shuffle_preserves_all_functions(self, sample_functions_json):
        """All functions are preserved after shuffle."""
        original = json.loads(sample_functions_json)
        result = shuffle_json_functions(sample_functions_json)
        shuffled = json.loads(result)

        assert len(shuffled) == len(original)

        original_names = {f["name"] for f in original}
        shuffled_names = {f["name"] for f in shuffled}
        assert original_names == shuffled_names

    def test_shuffle_empty_list(self):
        """Shuffling empty list returns empty list."""
        result = shuffle_json_functions("[]")
        assert result == "[]"

    def test_shuffle_single_function(self):
        """Shuffling single function returns same function."""
        single = '[{"name": "test"}]'
        result = shuffle_json_functions(single)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "test"


class TestFormatGenerateUserQuery:
    """Tests for format_generate_user_query function."""

    def test_format_without_query(self, sample_functions_json):
        """Format prompt without user query."""
        result = format_generate_user_query(sample_functions_json)
        assert "functions are available" in result
        assert "generate a potential user query" in result
        assert sample_functions_json in result
        assert "</s>" not in result

    def test_format_with_query(self, sample_functions_json):
        """Format prompt with user query for training."""
        query = "What's the weather?"
        result = format_generate_user_query(sample_functions_json, user_query=query)
        assert query in result
        assert result.endswith("</s>")


class TestGetTrainableParamCount:
    """Tests for get_trainable_param_count function."""

    def test_param_count_mock_model(self):
        """Get parameter counts from a mock model."""
        from unittest.mock import MagicMock

        # Create mock parameters
        class MockParam:
            def __init__(self, numel_val, requires_grad):
                self._numel = numel_val
                self.requires_grad = requires_grad

            def numel(self):
                return self._numel

        mock_model = MagicMock()
        mock_model.parameters.return_value = [
            MockParam(1000, True),  # trainable
            MockParam(2000, True),  # trainable
            MockParam(5000, False),  # frozen
        ]

        trainable, total = get_trainable_param_count(mock_model)
        assert trainable == 3000
        assert total == 8000
