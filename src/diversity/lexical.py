"""
Lexical diversity metrics based on n-gram analysis.

These metrics capture surface-level syntactic variation - different words,
different n-gram patterns. They are fast to compute and model-agnostic.

Metrics:
- Distinct-n: Ratio of unique n-grams to total n-grams
- Type-Token Ratio (TTR): Unique words / total words
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable


def tokenize_simple(text: str) -> list[str]:
    """
    Simple whitespace tokenizer with basic normalization.

    Splits on whitespace and removes empty tokens.
    Preserves case and punctuation for n-gram diversity.
    """
    return text.split()


def tokenize_words(text: str) -> list[str]:
    """
    Word tokenizer that extracts alphanumeric words.

    Better for measuring true lexical diversity without punctuation noise.
    """
    return re.findall(r"\b\w+\b", text.lower())


def compute_distinct_n(
    texts: list[str],
    n: int,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> float:
    """
    Compute Distinct-n metric: ratio of unique n-grams to total n-grams.

    Higher values indicate more lexical diversity.
    - 1.0 = all n-grams are unique (maximum diversity)
    - 0.0 = all n-grams are identical (no diversity)

    Args:
        texts: List of text samples to analyze.
        n: Size of n-grams (1 for unigrams, 2 for bigrams, etc.)
        tokenizer: Optional tokenizer function. Defaults to simple whitespace split.

    Returns:
        Ratio of unique n-grams to total n-grams (0.0 to 1.0).

    Example:
        >>> texts = ["hello world", "hello there", "goodbye world"]
        >>> compute_distinct_n(texts, 1)  # Unigrams
        0.8  # 4 unique out of 5 total words (hello appears twice)
    """
    if not texts:
        return 0.0

    if tokenizer is None:
        tokenizer = tokenize_simple

    all_ngrams: list[tuple[str, ...]] = []

    for text in texts:
        tokens = tokenizer(text)
        if len(tokens) < n:
            continue
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    unique_ngrams = len(set(all_ngrams))
    total_ngrams = len(all_ngrams)

    return unique_ngrams / total_ngrams


def compute_type_token_ratio(
    texts: list[str],
    tokenizer: Callable[[str], list[str]] | None = None,
) -> float:
    """
    Compute Type-Token Ratio (TTR): unique words / total words.

    A classic measure of lexical diversity.
    Higher TTR = more diverse vocabulary.

    Args:
        texts: List of text samples to analyze.
        tokenizer: Optional tokenizer function. Defaults to word tokenizer.

    Returns:
        Type-token ratio (0.0 to 1.0).
    """
    if not texts:
        return 0.0

    if tokenizer is None:
        tokenizer = tokenize_words

    all_tokens: list[str] = []
    for text in texts:
        all_tokens.extend(tokenizer(text))

    if not all_tokens:
        return 0.0

    unique_tokens = len(set(all_tokens))
    total_tokens = len(all_tokens)

    return unique_tokens / total_tokens


def compute_lexical_diversity(
    texts: list[str],
    max_n: int = 3,
    include_ttr: bool = True,
) -> dict[str, float]:
    """
    Compute comprehensive lexical diversity metrics.

    Returns Distinct-1, Distinct-2, Distinct-3 (configurable), and optionally TTR.

    Args:
        texts: List of text samples to analyze.
        max_n: Maximum n-gram size to compute (default: 3).
        include_ttr: Whether to include Type-Token Ratio (default: True).

    Returns:
        Dictionary with diversity metrics:
        {
            "distinct_1": float,
            "distinct_2": float,
            "distinct_3": float,
            "type_token_ratio": float,  # if include_ttr
            "sample_count": int,
            "total_tokens": int,
        }

    Example:
        >>> texts = ["The quick brown fox", "The lazy dog sleeps"]
        >>> metrics = compute_lexical_diversity(texts)
        >>> print(metrics["distinct_2"])
        0.875
    """
    if not texts:
        return {f"distinct_{i}": 0.0 for i in range(1, max_n + 1)} | {
            "type_token_ratio": 0.0,
            "sample_count": 0,
            "total_tokens": 0,
        }

    # Count total tokens for reporting
    total_tokens = sum(len(tokenize_simple(text)) for text in texts)

    result: dict[str, float | int] = {
        "sample_count": len(texts),
        "total_tokens": total_tokens,
    }

    # Compute Distinct-n for each n
    for n in range(1, max_n + 1):
        result[f"distinct_{n}"] = compute_distinct_n(texts, n)

    # Optionally compute TTR
    if include_ttr:
        result["type_token_ratio"] = compute_type_token_ratio(texts)

    return result


def compute_ngram_frequency(
    texts: list[str],
    n: int,
    top_k: int = 10,
    tokenizer: Callable[[str], list[str]] | None = None,
) -> list[tuple[tuple[str, ...], int]]:
    """
    Compute most frequent n-grams in the dataset.

    Useful for understanding what patterns are repeated.

    Args:
        texts: List of text samples.
        n: Size of n-grams.
        top_k: Number of top n-grams to return.
        tokenizer: Optional tokenizer function.

    Returns:
        List of (n-gram, count) tuples, sorted by frequency descending.
    """
    if not texts:
        return []

    if tokenizer is None:
        tokenizer = tokenize_simple

    ngram_counts: Counter[tuple[str, ...]] = Counter()

    for text in texts:
        tokens = tokenizer(text)
        if len(tokens) < n:
            continue
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        ngram_counts.update(ngrams)

    return ngram_counts.most_common(top_k)


def compute_vocabulary_coverage(
    texts: list[str],
    tokenizer: Callable[[str], list[str]] | None = None,
) -> dict[str, float | int]:
    """
    Compute vocabulary statistics for the dataset.

    Args:
        texts: List of text samples.
        tokenizer: Optional tokenizer function.

    Returns:
        Dictionary with vocabulary statistics:
        {
            "vocabulary_size": int,
            "total_tokens": int,
            "hapax_legomena": int,  # Words appearing exactly once
            "hapax_ratio": float,   # Ratio of hapax to vocabulary
        }
    """
    if not texts:
        return {
            "vocabulary_size": 0,
            "total_tokens": 0,
            "hapax_legomena": 0,
            "hapax_ratio": 0.0,
        }

    if tokenizer is None:
        tokenizer = tokenize_words

    token_counts: Counter[str] = Counter()
    for text in texts:
        token_counts.update(tokenizer(text))

    vocabulary_size = len(token_counts)
    total_tokens = sum(token_counts.values())
    hapax_legomena = sum(1 for count in token_counts.values() if count == 1)

    return {
        "vocabulary_size": vocabulary_size,
        "total_tokens": total_tokens,
        "hapax_legomena": hapax_legomena,
        "hapax_ratio": hapax_legomena / vocabulary_size if vocabulary_size > 0 else 0.0,
    }
