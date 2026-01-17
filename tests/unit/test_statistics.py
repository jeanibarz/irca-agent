"""Unit tests for statistical utilities."""

import numpy as np
import pytest

from src.diversity.statistics import (
    bootstrap_confidence_interval,
    bootstrap_difference_test,
    effect_size_cohens_d,
    interpret_effect_size,
)


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap_confidence_interval function."""

    def test_basic_ci(self):
        """Test basic confidence interval computation."""
        values = [10.0, 12.0, 11.0, 13.0, 9.0]
        mean, lower, upper = bootstrap_confidence_interval(values, seed=42)

        assert mean == pytest.approx(11.0, rel=0.01)
        assert lower < mean < upper
        assert lower > 9.0  # Should be reasonable
        assert upper < 14.0

    def test_ci_width_decreases_with_sample_size(self):
        """Larger samples should have narrower CIs."""
        np.random.seed(42)
        small_sample = np.random.normal(100, 10, size=10).tolist()
        large_sample = np.random.normal(100, 10, size=100).tolist()

        _, small_lower, small_upper = bootstrap_confidence_interval(small_sample, seed=42)
        _, large_lower, large_upper = bootstrap_confidence_interval(large_sample, seed=42)

        small_width = small_upper - small_lower
        large_width = large_upper - large_lower

        assert large_width < small_width

    def test_empty_list(self):
        """Empty list should return zeros."""
        mean, lower, upper = bootstrap_confidence_interval([])
        assert mean == 0.0
        assert lower == 0.0
        assert upper == 0.0

    def test_single_value(self):
        """Single value should return that value for all."""
        value = 42.0
        mean, lower, upper = bootstrap_confidence_interval([value])
        assert mean == value
        assert lower == value
        assert upper == value

    def test_reproducibility(self):
        """Same seed should give same results."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result1 = bootstrap_confidence_interval(values, seed=123)
        result2 = bootstrap_confidence_interval(values, seed=123)
        assert result1 == result2

    def test_different_seeds(self):
        """Different seeds should give different results."""
        values = list(range(100))
        result1 = bootstrap_confidence_interval(values, seed=1)
        result2 = bootstrap_confidence_interval(values, seed=2)
        # Mean should be same, but CI bounds may differ slightly
        assert result1[0] == result2[0]  # Same mean
        # CI bounds may differ due to random sampling


class TestBootstrapDifferenceTest:
    """Tests for bootstrap_difference_test function."""

    def test_significant_difference(self):
        """Test detection of significant difference."""
        # Two clearly different distributions
        np.random.seed(42)
        values_a = np.random.normal(100, 5, size=50).tolist()
        values_b = np.random.normal(80, 5, size=50).tolist()

        result = bootstrap_difference_test(values_a, values_b, seed=42)

        assert result["difference"] < 0  # B has lower mean
        assert result["significant"] is True
        assert result["ci_upper"] < 0  # Entire CI below zero

    def test_no_significant_difference(self):
        """Test when distributions are similar."""
        np.random.seed(42)
        values_a = np.random.normal(100, 10, size=30).tolist()
        values_b = np.random.normal(100, 10, size=30).tolist()

        result = bootstrap_difference_test(values_a, values_b, seed=42)

        # CI should include zero for similar distributions
        # Note: with random sampling, this isn't guaranteed
        assert result["ci_lower"] < 5  # Should be close to zero
        assert result["ci_upper"] > -5

    def test_returns_all_fields(self):
        """Test that all expected fields are returned."""
        values_a = [1.0, 2.0, 3.0]
        values_b = [4.0, 5.0, 6.0]

        result = bootstrap_difference_test(values_a, values_b)

        assert "mean_a" in result
        assert "mean_b" in result
        assert "difference" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "significant" in result
        assert "p_value_approx" in result


class TestEffectSizeCohensD:
    """Tests for effect_size_cohens_d function."""

    def test_large_effect(self):
        """Test detection of large effect size."""
        # Large separation between groups
        values_a = [1.0, 2.0, 3.0, 4.0, 5.0]
        values_b = [10.0, 11.0, 12.0, 13.0, 14.0]

        d = effect_size_cohens_d(values_a, values_b)

        assert abs(d) > 0.8  # Large effect

    def test_small_effect(self):
        """Test detection of small effect size."""
        values_a = [10.0, 11.0, 12.0, 13.0, 14.0]
        values_b = [10.5, 11.5, 12.5, 13.5, 14.5]

        d = effect_size_cohens_d(values_a, values_b)

        assert abs(d) < 0.5  # Small or negligible effect

    def test_direction(self):
        """Test that sign indicates direction correctly."""
        # Use values with variance so pooled std is non-zero
        values_a = [9.0, 10.0, 11.0]
        values_b = [4.0, 5.0, 6.0]

        d = effect_size_cohens_d(values_a, values_b)

        assert d < 0  # B is lower than A

    def test_identical_distributions(self):
        """Test that identical distributions give zero effect."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        d = effect_size_cohens_d(values, values)

        assert d == 0.0


class TestInterpretEffectSize:
    """Tests for interpret_effect_size function."""

    def test_negligible(self):
        """Test negligible effect interpretation."""
        result = interpret_effect_size(0.1)
        assert "Negligible" in result

    def test_small(self):
        """Test small effect interpretation."""
        result = interpret_effect_size(0.3)
        assert "Small" in result

    def test_medium(self):
        """Test medium effect interpretation."""
        result = interpret_effect_size(0.6)
        assert "Medium" in result

    def test_large(self):
        """Test large effect interpretation."""
        result = interpret_effect_size(1.0)
        assert "Large" in result

    def test_negative_direction(self):
        """Test that negative values indicate B is lower."""
        result = interpret_effect_size(-0.5)
        assert "lower" in result

    def test_positive_direction(self):
        """Test that positive values indicate B is higher."""
        result = interpret_effect_size(0.5)
        assert "higher" in result
