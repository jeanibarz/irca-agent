"""
Unit tests for lexical diversity metrics.

Tests FR-DATA-23: Quick Diversity Mode (lexical component).
"""

import pytest

from src.diversity.lexical import (
    compute_distinct_n,
    compute_lexical_diversity,
    compute_ngram_frequency,
    compute_type_token_ratio,
    compute_vocabulary_coverage,
    tokenize_simple,
    tokenize_words,
)


class TestTokenizers:
    """Tests for tokenizer functions."""

    def test_tokenize_simple_basic(self):
        """Simple tokenizer splits on whitespace."""
        text = "hello world foo bar"
        tokens = tokenize_simple(text)
        assert tokens == ["hello", "world", "foo", "bar"]

    def test_tokenize_simple_preserves_punctuation(self):
        """Simple tokenizer preserves punctuation."""
        text = "hello, world!"
        tokens = tokenize_simple(text)
        assert tokens == ["hello,", "world!"]

    def test_tokenize_simple_empty(self):
        """Simple tokenizer handles empty string."""
        assert tokenize_simple("") == []

    def test_tokenize_words_basic(self):
        """Word tokenizer extracts alphanumeric words."""
        text = "Hello, World! This is a test."
        tokens = tokenize_words(text)
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    def test_tokenize_words_lowercase(self):
        """Word tokenizer lowercases."""
        text = "HELLO World"
        tokens = tokenize_words(text)
        assert tokens == ["hello", "world"]


class TestDistinctN:
    """Tests for Distinct-n metric."""

    def test_distinct_1_basic(self):
        """Distinct-1 computes correctly."""
        # 5 total words, 4 unique ("hello" appears twice)
        texts = ["hello world", "hello there"]
        score = compute_distinct_n(texts, 1)
        # unique: hello, world, there = 3; total = 4
        # But wait: "hello world" has 2 tokens, "hello there" has 2 tokens
        # Total tokens: hello, world, hello, there = 4
        # Unique: hello, world, there = 3
        assert score == pytest.approx(3 / 4, rel=0.01)

    def test_distinct_2_basic(self):
        """Distinct-2 computes correctly."""
        texts = ["hello world foo", "hello world bar"]
        score = compute_distinct_n(texts, 2)
        # Bigrams from first: (hello, world), (world, foo)
        # Bigrams from second: (hello, world), (world, bar)
        # Total: 4, Unique: 3 (hello world appears twice)
        assert score == pytest.approx(3 / 4, rel=0.01)

    def test_distinct_n_all_unique(self):
        """Distinct-n with all unique n-grams returns 1.0."""
        texts = ["a b c", "d e f", "g h i"]
        score = compute_distinct_n(texts, 1)
        assert score == pytest.approx(1.0)

    def test_distinct_n_all_same(self):
        """Distinct-n with all same n-grams returns low score."""
        texts = ["hello hello hello"] * 5
        score = compute_distinct_n(texts, 1)
        # All "hello", so unique=1, total=15
        assert score == pytest.approx(1 / 15, rel=0.01)

    def test_distinct_n_empty_texts(self):
        """Distinct-n handles empty input."""
        assert compute_distinct_n([], 1) == 0.0

    def test_distinct_n_short_texts(self):
        """Distinct-n handles texts shorter than n."""
        texts = ["a", "b"]  # Single words, can't form bigrams
        score = compute_distinct_n(texts, 2)
        assert score == 0.0

    def test_distinct_n_custom_tokenizer(self):
        """Distinct-n with custom tokenizer."""
        texts = ["hello, world!", "hello, there!"]
        # With word tokenizer (removes punctuation)
        score = compute_distinct_n(texts, 1, tokenizer=tokenize_words)
        # Words: hello, world, hello, there = 4 total, 3 unique
        assert score == pytest.approx(3 / 4, rel=0.01)


class TestTypeTokenRatio:
    """Tests for Type-Token Ratio."""

    def test_ttr_basic(self):
        """TTR computes correctly."""
        texts = ["the cat sat", "the dog ran"]
        ttr = compute_type_token_ratio(texts)
        # Words: the, cat, sat, the, dog, ran = 6 total
        # Unique: the, cat, sat, dog, ran = 5
        assert ttr == pytest.approx(5 / 6, rel=0.01)

    def test_ttr_all_unique(self):
        """TTR = 1.0 when all words are unique."""
        texts = ["apple banana cherry"]
        ttr = compute_type_token_ratio(texts)
        assert ttr == pytest.approx(1.0)

    def test_ttr_all_same(self):
        """TTR approaches 0 when all words are same."""
        texts = ["hello " * 100]
        ttr = compute_type_token_ratio(texts)
        assert ttr == pytest.approx(1 / 100, rel=0.01)

    def test_ttr_empty(self):
        """TTR handles empty input."""
        assert compute_type_token_ratio([]) == 0.0


class TestLexicalDiversity:
    """Tests for comprehensive lexical diversity."""

    def test_lexical_diversity_basic(self):
        """Lexical diversity returns all metrics."""
        texts = ["hello world", "foo bar baz"]
        result = compute_lexical_diversity(texts)

        assert "distinct_1" in result
        assert "distinct_2" in result
        assert "distinct_3" in result
        assert "type_token_ratio" in result
        assert "sample_count" in result
        assert "total_tokens" in result

        assert result["sample_count"] == 2
        assert result["total_tokens"] == 5

    def test_lexical_diversity_empty(self):
        """Lexical diversity handles empty input."""
        result = compute_lexical_diversity([])
        assert result["sample_count"] == 0
        assert result["distinct_1"] == 0.0

    def test_lexical_diversity_custom_max_n(self):
        """Lexical diversity with custom max_n."""
        texts = ["a b c d e"]
        result = compute_lexical_diversity(texts, max_n=5)
        assert "distinct_5" in result

    def test_lexical_diversity_no_ttr(self):
        """Lexical diversity without TTR."""
        texts = ["hello world"]
        result = compute_lexical_diversity(texts, include_ttr=False)
        assert "type_token_ratio" not in result


class TestNgramFrequency:
    """Tests for n-gram frequency analysis."""

    def test_ngram_frequency_basic(self):
        """N-gram frequency returns most common."""
        texts = ["the cat sat", "the cat ran", "the dog sat"]
        freq = compute_ngram_frequency(texts, n=1, top_k=3)

        # "the" appears 3 times, should be first
        assert freq[0] == (("the",), 3)

    def test_ngram_frequency_bigrams(self):
        """N-gram frequency for bigrams."""
        texts = ["a b c", "a b d", "a b e"]
        freq = compute_ngram_frequency(texts, n=2, top_k=2)

        # "a b" appears 3 times
        assert freq[0] == (("a", "b"), 3)

    def test_ngram_frequency_empty(self):
        """N-gram frequency handles empty input."""
        assert compute_ngram_frequency([], n=1) == []


class TestVocabularyCoverage:
    """Tests for vocabulary coverage statistics."""

    def test_vocab_coverage_basic(self):
        """Vocabulary coverage computes correctly."""
        texts = ["the cat", "a cat", "the dog"]
        result = compute_vocabulary_coverage(texts)

        # Words: the, cat, a, cat, the, dog
        # Vocabulary: the, cat, a, dog = 4
        # Hapax (appear once): a, dog = 2
        assert result["vocabulary_size"] == 4
        assert result["total_tokens"] == 6
        assert result["hapax_legomena"] == 2
        assert result["hapax_ratio"] == pytest.approx(2 / 4, rel=0.01)

    def test_vocab_coverage_empty(self):
        """Vocabulary coverage handles empty input."""
        result = compute_vocabulary_coverage([])
        assert result["vocabulary_size"] == 0
        assert result["hapax_ratio"] == 0.0


class TestDiversityInvariants:
    """Tests for diversity metric invariants."""

    def test_distinct_n_bounded(self):
        """Distinct-n is always between 0 and 1."""
        import random

        random.seed(42)
        for _ in range(10):
            texts = [" ".join(random.choices("abcdefghij", k=10)) for _ in range(5)]
            for n in [1, 2, 3]:
                score = compute_distinct_n(texts, n)
                assert 0.0 <= score <= 1.0

    def test_ttr_bounded(self):
        """TTR is always between 0 and 1."""
        import random

        random.seed(42)
        for _ in range(10):
            texts = [" ".join(random.choices("abcdefghij", k=10)) for _ in range(5)]
            ttr = compute_type_token_ratio(texts)
            assert 0.0 <= ttr <= 1.0

    def test_duplicates_reduce_diversity(self):
        """Adding duplicates reduces or maintains diversity."""
        unique_texts = ["hello world", "foo bar", "baz qux"]
        d1_unique = compute_distinct_n(unique_texts, 1)

        # Add duplicates
        duplicated_texts = unique_texts * 3
        d1_duplicated = compute_distinct_n(duplicated_texts, 1)

        # Duplicates should reduce distinct-n
        assert d1_duplicated <= d1_unique
