"""
Unit tests for perplexity-based diversity metrics.

Tests FR-DATA-18: Perplexity Diversity Metrics.
Tests FR-DATA-21: Diversity Comparison.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.diversity.perplexity import (
    compare_datasets,
    compute_perplexity_profile,
    compute_quick_diversity,
    compute_sample_perplexity,
)


class TestSamplePerplexity:
    """Tests for single sample perplexity computation."""

    def test_sample_perplexity_basic(self):
        """Sample perplexity computes exp(loss)."""
        # Create mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer to return tensor
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }

        # Mock model to return loss
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(2.0)  # exp(2) ≈ 7.39
        mock_model.return_value = mock_output
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])

        ppl = compute_sample_perplexity("test text", mock_model, mock_tokenizer)

        assert ppl == pytest.approx(np.exp(2.0), rel=0.01)

    def test_sample_perplexity_empty_text(self):
        """Sample perplexity returns inf for empty text."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        ppl = compute_sample_perplexity("", mock_model, mock_tokenizer)
        assert ppl == float("inf")

    def test_sample_perplexity_whitespace_only(self):
        """Sample perplexity returns inf for whitespace-only text."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        ppl = compute_sample_perplexity("   \n\t  ", mock_model, mock_tokenizer)
        assert ppl == float("inf")


class TestPerplexityProfile:
    """Tests for perplexity profile computation."""

    @patch("src.diversity.perplexity.load_model_and_tokenizer")
    @patch("src.diversity.perplexity.compute_sample_perplexity")
    def test_perplexity_profile_basic(self, mock_compute_ppl, mock_load):
        """Perplexity profile computes all statistics."""
        # Setup mock
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # Mock perplexity computation with varying values
        perplexities = [7.39, 20.09, 12.18]  # exp(2), exp(3), exp(2.5)
        mock_compute_ppl.side_effect = perplexities

        texts = ["text 1", "text 2", "text 3"]
        result = compute_perplexity_profile(
            texts,
            model_name="mock-model",
            show_progress=False,
            use_cache=False,
        )

        assert "mean_perplexity" in result
        assert "std_perplexity" in result
        assert "min_perplexity" in result
        assert "max_perplexity" in result
        assert "median_perplexity" in result
        assert "p90_perplexity" in result
        assert "sample_count" in result
        assert result["sample_count"] == 3

        # Check statistics are reasonable
        assert result["mean_perplexity"] == pytest.approx(np.mean(perplexities), rel=0.01)
        assert result["min_perplexity"] == pytest.approx(min(perplexities), rel=0.01)
        assert result["max_perplexity"] == pytest.approx(max(perplexities), rel=0.01)

    def test_perplexity_profile_empty(self):
        """Perplexity profile handles empty input."""
        result = compute_perplexity_profile([], model_name="mock", show_progress=False)
        assert result["sample_count"] == 0
        assert result["mean_perplexity"] == 0.0

    @patch("src.diversity.perplexity.load_model_and_tokenizer")
    @patch("src.diversity.perplexity.compute_sample_perplexity")
    def test_perplexity_profile_sampling(self, mock_compute_ppl, mock_load):
        """Perplexity profile with sampling."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        mock_compute_ppl.return_value = 10.0

        texts = [f"text {i}" for i in range(100)]
        result = compute_perplexity_profile(
            texts,
            model_name="mock",
            sample_size=10,
            show_progress=False,
            use_cache=False,
        )

        # Should have processed only 10 samples
        assert result["sample_count"] == 10
        # Mock should have been called 10 times
        assert mock_compute_ppl.call_count == 10


class TestCompareDatasets:
    """Tests for dataset comparison."""

    def test_compare_datasets_quick_mode(self):
        """Compare datasets in quick mode (no model)."""
        original = ["hello world", "foo bar"]
        augmented = ["hello world", "foo bar", "baz qux", "test sample"]

        result = compare_datasets(original, augmented, model_name=None, quick=True)

        assert "original" in result
        assert "augmented" in result
        assert "delta" in result
        assert "summary" in result

        # Check lexical metrics present
        assert "lexical" in result["original"]
        assert "lexical" in result["augmented"]

        # Check deltas
        assert "distinct_1_change" in result["delta"]
        assert "distinct_2_change" in result["delta"]

    def test_compare_datasets_requires_model(self):
        """Compare datasets requires model when not quick mode."""
        with pytest.raises(ValueError, match="model_name is required"):
            compare_datasets(["a"], ["b"], model_name=None, quick=False)

    def test_compare_datasets_sample_count(self):
        """Compare datasets records sample counts."""
        original = ["text"] * 10
        augmented = ["text"] * 30

        result = compare_datasets(original, augmented, quick=True)

        assert result["original"]["sample_count"] == 10
        assert result["augmented"]["sample_count"] == 30


class TestQuickDiversity:
    """Tests for quick diversity computation."""

    def test_quick_diversity_basic(self):
        """Quick diversity computes lexical metrics."""
        texts = ["hello world", "foo bar baz"]
        result = compute_quick_diversity(texts)

        assert "lexical" in result
        assert "sample_count" in result
        assert result["sample_count"] == 2

    def test_quick_diversity_no_semantic(self):
        """Quick diversity without semantic by default."""
        texts = ["hello world"]
        result = compute_quick_diversity(texts, include_semantic=False)

        assert "lexical" in result
        assert "semantic" not in result

    def test_quick_diversity_with_semantic(self):
        """Quick diversity with semantic metrics."""
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            pytest.skip("sentence-transformers not installed")

        texts = ["hello world", "foo bar"]
        result = compute_quick_diversity(texts, include_semantic=True)

        assert "lexical" in result
        assert "semantic" in result


class TestPerplexityInvariants:
    """Tests for perplexity metric invariants."""

    @patch("src.diversity.perplexity.load_model_and_tokenizer")
    def test_perplexity_positive(self, mock_load):
        """Perplexity is always positive."""
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_model.return_value.loss = torch.tensor(1.5)
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.eval = MagicMock()

        result = compute_perplexity_profile(["test"], model_name="mock", show_progress=False, use_cache=False)

        assert result["mean_perplexity"] > 0
        assert result["min_perplexity"] > 0

    def test_delta_format(self):
        """Delta formatting works correctly."""
        from src.diversity.utils import format_delta

        assert format_delta(100, 125) == "+25.0%"
        assert format_delta(100, 75) == "-25.0%"
        assert format_delta(100, 100) == "+0.0%"
        assert format_delta(0, 100) == "N/A"
