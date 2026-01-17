"""
Statistical utilities for diversity evaluation.

This module provides statistical functions for computing confidence intervals
and significance tests, particularly for the ADR-008 experiment methodology.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    values: Sequence[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Uses non-parametric bootstrap resampling to estimate the sampling
    distribution of the mean, then computes percentile-based CI.

    Args:
        values: Sequence of observed values (e.g., perplexities).
        n_bootstrap: Number of bootstrap resamples. Default: 1000.
        confidence: Confidence level (e.g., 0.95 for 95% CI). Default: 0.95.
        seed: Random seed for reproducibility. Default: 42.

    Returns:
        Tuple of (mean, ci_lower, ci_upper) where:
        - mean: Sample mean of the original values
        - ci_lower: Lower bound of the confidence interval
        - ci_upper: Upper bound of the confidence interval

    Example:
        >>> perplexities = [12.5, 15.2, 11.8, 14.1, 13.3]
        >>> mean, lower, upper = bootstrap_confidence_interval(perplexities)
        >>> print(f"Mean: {mean:.2f} [{lower:.2f}, {upper:.2f}]")
        Mean: 13.38 [12.24, 14.52]
    """
    values = np.array(values)
    n = len(values)

    if n == 0:
        return 0.0, 0.0, 0.0

    if n == 1:
        return float(values[0]), float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)

    # Generate bootstrap samples and compute means
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Compute percentiles
    alpha = 1 - confidence
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = float(np.percentile(bootstrap_means, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_means, upper_percentile))
    mean = float(np.mean(values))

    return mean, ci_lower, ci_upper


def bootstrap_difference_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Test whether two samples have significantly different means using bootstrap.

    Computes the bootstrap distribution of the difference in means and
    reports whether zero is within the confidence interval.

    Args:
        values_a: First sample (e.g., baseline perplexities).
        values_b: Second sample (e.g., augmented perplexities).
        n_bootstrap: Number of bootstrap resamples. Default: 1000.
        seed: Random seed for reproducibility. Default: 42.

    Returns:
        Dictionary with:
        - mean_a: Mean of first sample
        - mean_b: Mean of second sample
        - difference: mean_b - mean_a (negative means B is better)
        - ci_lower: Lower 95% CI of difference
        - ci_upper: Upper 95% CI of difference
        - significant: Whether CI excludes zero
        - p_value_approx: Approximate p-value (proportion of bootstrap
          samples with opposite sign to observed difference)
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    mean_a = float(np.mean(values_a))
    mean_b = float(np.mean(values_b))
    observed_diff = mean_b - mean_a

    rng = np.random.default_rng(seed)

    # Bootstrap the difference
    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        sample_a = rng.choice(values_a, size=len(values_a), replace=True)
        sample_b = rng.choice(values_b, size=len(values_b), replace=True)
        bootstrap_diffs.append(np.mean(sample_b) - np.mean(sample_a))

    bootstrap_diffs = np.array(bootstrap_diffs)

    ci_lower = float(np.percentile(bootstrap_diffs, 2.5))
    ci_upper = float(np.percentile(bootstrap_diffs, 97.5))

    # Check if CI excludes zero
    significant = (ci_lower > 0) or (ci_upper < 0)

    # Approximate p-value: proportion of bootstrap samples with opposite sign
    if observed_diff >= 0:
        p_value = float(np.mean(bootstrap_diffs < 0))
    else:
        p_value = float(np.mean(bootstrap_diffs > 0))
    p_value = 2 * min(p_value, 1 - p_value)  # Two-tailed

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "difference": observed_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
        "p_value_approx": p_value,
    }


def effect_size_cohens_d(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> float:
    """
    Compute Cohen's d effect size for the difference between two samples.

    Cohen's d is the standardized difference between means:
    d = (mean_b - mean_a) / pooled_std

    Interpretation guidelines:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        values_a: First sample.
        values_b: Second sample.

    Returns:
        Cohen's d effect size. Negative means B has lower mean than A.
    """
    values_a = np.array(values_a)
    values_b = np.array(values_b)

    n_a = len(values_a)
    n_b = len(values_b)

    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)

    var_a = np.var(values_a, ddof=1)
    var_b = np.var(values_b, ddof=1)

    # Pooled standard deviation
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
    pooled_std = np.sqrt(pooled_var)

    if pooled_std == 0:
        return 0.0

    return float((mean_b - mean_a) / pooled_std)


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value.

    Returns:
        Human-readable interpretation.
    """
    abs_d = abs(d)
    direction = "lower" if d < 0 else "higher"

    if abs_d < 0.2:
        return f"Negligible effect ({d:.2f})"
    elif abs_d < 0.5:
        return f"Small effect ({d:.2f}): B is {direction}"
    elif abs_d < 0.8:
        return f"Medium effect ({d:.2f}): B is {direction}"
    else:
        return f"Large effect ({d:.2f}): B is {direction}"
