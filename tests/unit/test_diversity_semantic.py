"""
Unit tests for semantic diversity metrics.

Tests FR-DATA-23: Quick Diversity Mode (semantic component).
"""

import pytest

# Skip all tests if sentence-transformers is not installed
pytest.importorskip("sentence_transformers")


from src.diversity.semantic import (
    compute_embedding_coverage,
    compute_semantic_diversity,
    compute_vendi_score,
)


class TestVendiScore:
    """Tests for Vendi Score computation."""

    def test_vendi_score_identical_texts(self):
        """Vendi Score ≈ 1 for identical texts."""
        texts = ["hello world"] * 10
        score = compute_vendi_score(texts, show_progress=False)
        # Should be close to 1 since all texts are identical
        assert score < 2.0

    def test_vendi_score_unique_texts(self):
        """Vendi Score approaches n for very different texts."""
        # Use semantically diverse texts
        texts = [
            "The weather is sunny today.",
            "I love programming in Python.",
            "The cat sleeps on the couch.",
            "Mathematics is a beautiful subject.",
            "The ocean waves crash on the shore.",
        ]
        score = compute_vendi_score(texts, show_progress=False)
        # Should be closer to n (5) since texts are different
        assert score > 2.0
        assert score <= len(texts)

    def test_vendi_score_duplicates_invariant(self):
        """Vendi Score doesn't change significantly with duplicates."""
        texts = [
            "Hello world, how are you?",
            "The quick brown fox jumps.",
            "Python is a programming language.",
        ]

        score_original = compute_vendi_score(texts, show_progress=False)

        # Duplicate the dataset
        texts_duplicated = texts * 3
        score_duplicated = compute_vendi_score(texts_duplicated, show_progress=False)

        # Score should be similar (within 20%)
        assert abs(score_original - score_duplicated) / score_original < 0.2

    def test_vendi_score_empty(self):
        """Vendi Score handles empty input."""
        assert compute_vendi_score([], show_progress=False) == 0.0

    def test_vendi_score_single(self):
        """Vendi Score = 1 for single text."""
        score = compute_vendi_score(["hello world"], show_progress=False)
        assert score == pytest.approx(1.0)

    def test_vendi_score_bounded(self):
        """Vendi Score is bounded by dataset size."""
        texts = ["text " + str(i) for i in range(20)]
        score = compute_vendi_score(texts, show_progress=False)
        assert 1.0 <= score <= len(texts)

    def test_vendi_score_sampling(self):
        """Vendi Score with sampling works."""
        texts = ["sample text " + str(i) for i in range(100)]
        score = compute_vendi_score(texts, sample_size=50, show_progress=False)
        assert score > 0


class TestEmbeddingCoverage:
    """Tests for embedding space coverage metrics."""

    def test_embedding_coverage_basic(self):
        """Embedding coverage returns all metrics."""
        texts = ["hello world", "foo bar", "baz qux"]
        result = compute_embedding_coverage(texts, show_progress=False)

        assert "avg_pairwise_distance" in result
        assert "std_pairwise_distance" in result
        assert "embedding_spread" in result
        assert "centroid_distance" in result
        assert "sample_count" in result

    def test_embedding_coverage_identical(self):
        """Embedding coverage metrics for identical texts."""
        texts = ["hello world"] * 5
        result = compute_embedding_coverage(texts, show_progress=False)

        # Identical texts should have near-zero pairwise distance
        assert result["avg_pairwise_distance"] < 0.01

    def test_embedding_coverage_diverse(self):
        """Embedding coverage metrics for diverse texts."""
        texts = [
            "The sun shines brightly in the sky.",
            "I love to code in various programming languages.",
            "The ocean is vast and mysterious.",
            "Mathematics reveals the universe's patterns.",
            "Music brings joy to people everywhere.",
        ]
        result = compute_embedding_coverage(texts, show_progress=False)

        # Diverse texts should have higher pairwise distance
        assert result["avg_pairwise_distance"] > 0.1

    def test_embedding_coverage_empty(self):
        """Embedding coverage handles empty input."""
        result = compute_embedding_coverage([], show_progress=False)
        assert result["sample_count"] == 0
        assert result["avg_pairwise_distance"] == 0.0


class TestSemanticDiversity:
    """Tests for comprehensive semantic diversity."""

    def test_semantic_diversity_basic(self):
        """Semantic diversity returns all metrics."""
        texts = ["hello world", "foo bar baz", "test sample"]
        result = compute_semantic_diversity(texts, show_progress=False)

        assert "vendi_score" in result
        assert "effective_diversity_ratio" in result
        assert "avg_pairwise_distance" in result
        assert "embedding_spread" in result
        assert "embedding_model" in result

    def test_semantic_diversity_effective_ratio(self):
        """Effective diversity ratio is correctly computed."""
        texts = ["sample " + str(i) for i in range(10)]
        result = compute_semantic_diversity(texts, show_progress=False)

        expected_ratio = result["vendi_score"] / len(texts)
        assert result["effective_diversity_ratio"] == pytest.approx(expected_ratio)

    def test_semantic_diversity_empty(self):
        """Semantic diversity handles empty input."""
        result = compute_semantic_diversity([], show_progress=False)
        assert result["sample_count"] == 0
        assert result["vendi_score"] == 0.0


class TestSemanticInvariants:
    """Tests for semantic diversity invariants."""

    def test_vendi_increases_with_diversity(self):
        """Vendi Score increases when adding diverse samples."""
        base_texts = ["The cat sits on the mat."] * 5

        score_base = compute_vendi_score(base_texts, show_progress=False)

        # Add diverse texts
        diverse_texts = base_texts + [
            "Programming is a valuable skill.",
            "The ocean is blue and vast.",
        ]
        score_diverse = compute_vendi_score(diverse_texts, show_progress=False)

        assert score_diverse > score_base

    def test_pairwise_distance_symmetric(self):
        """Pairwise distance is non-negative."""
        texts = ["hello", "world", "test"]
        result = compute_embedding_coverage(texts, show_progress=False)
        assert result["avg_pairwise_distance"] >= 0.0

    def test_coverage_bounded(self):
        """Coverage metrics are bounded."""
        texts = ["text " + str(i) for i in range(20)]
        result = compute_embedding_coverage(texts, show_progress=False)

        # Cosine distance is bounded [0, 2] for normalized vectors
        assert 0.0 <= result["avg_pairwise_distance"] <= 2.0
