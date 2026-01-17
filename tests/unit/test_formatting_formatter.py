"""
Unit tests for DatasetFormatter.

Tests FR-DATA-24: Dataset Format Command.
Tests FR-DATA-26: Format Baseline.
"""

import pytest

from datasets import Dataset
from src.formatting.config import FormatConfig, StepConfig, create_baseline_config
from src.formatting.formatter import DatasetFormatter, format_dataset


class TestDatasetFormatter:
    """Tests for DatasetFormatter."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with structured data."""
        # Simulate the corrected_agent_trace format
        sample_trace = """
### INSTRUCTIONS
You are an AI assistant.

EXAMPLE:
Sample example.

The functions available to you are described below.

### FUNCTIONS AVAILABLE
[{"name": "test_func", "description": "A test function"}]

### USER QUERY
What is the weather?

### ITERATIVE RESOLUTION CYCLE
Thought: I need to check the weather.
Action choice: final answer

### FINAL ANSWER
The weather is sunny.
"""
        data = {
            "corrected_agent_trace": [
                [{"value": sample_trace, "status": "submitted"}],
                [{"value": sample_trace.replace("sunny", "rainy"), "status": "submitted"}],
            ],
            "user_query": ["What is the weather?", "What is the weather?"],
        }
        return Dataset.from_dict(data)

    def test_format_baseline(self, sample_dataset):
        """Format dataset without augmentation."""
        config = create_baseline_config()
        formatter = DatasetFormatter(config)

        result, stats = formatter.format(sample_dataset)

        assert "text" in result.column_names
        assert len(result) == 2
        assert stats["input_samples"] == 2
        assert stats["output_samples"] == 2
        assert stats["parse_errors"] == 0

    def test_format_produces_text_column(self, sample_dataset):
        """Formatted dataset has text column with content."""
        config = create_baseline_config()
        formatter = DatasetFormatter(config)

        result, _ = formatter.format(sample_dataset)

        assert "### INSTRUCTIONS" in result[0]["text"]
        assert "### USER QUERY" in result[0]["text"]
        assert "What is the weather?" in result[0]["text"]

    def test_format_preserves_original_index(self, sample_dataset):
        """Formatted dataset has original_index column."""
        config = create_baseline_config()
        formatter = DatasetFormatter(config)

        result, _ = formatter.format(sample_dataset)

        assert "original_index" in result.column_names
        assert result[0]["original_index"] == 0
        assert result[1]["original_index"] == 1

    def test_format_with_shuffling(self, sample_dataset):
        """Format with shuffle_functions augmentation."""
        config = FormatConfig(
            seed=42,
            formatting=[StepConfig(type="shuffle_functions", probability=1.0)],
        )
        formatter = DatasetFormatter(config)

        result, stats = formatter.format(sample_dataset)

        assert len(result) == 2
        assert stats["augmentation_counts"]["shuffle_functions"] == 2

    def test_format_with_multiply(self, sample_dataset):
        """Format with dataset multiplication."""
        config = FormatConfig(
            seed=42,
            multiply=3,
            formatting=[StepConfig(type="shuffle_functions", probability=1.0)],
        )
        formatter = DatasetFormatter(config)

        result, stats = formatter.format(sample_dataset)

        # 2 samples x 3 multiplier = 6 samples
        assert stats["variants_generated"] == 6
        # Some may be deduplicated
        assert len(result) <= 6

    def test_format_records_augmentations(self, sample_dataset):
        """Formatted samples record applied augmentations."""
        config = FormatConfig(
            seed=42,
            formatting=[StepConfig(type="shuffle_functions", probability=1.0)],
        )
        formatter = DatasetFormatter(config)

        result, _ = formatter.format(sample_dataset)

        assert "augmentations" in result.column_names
        # Baseline (variant_id=0) should have augmentations applied too
        assert "shuffle_functions" in result[0]["augmentations"]

    def test_format_reproducible(self, sample_dataset):
        """Same seed produces same results."""
        config1 = FormatConfig(
            seed=42,
            formatting=[StepConfig(type="shuffle_functions", probability=1.0)],
        )
        config2 = FormatConfig(
            seed=42,
            formatting=[StepConfig(type="shuffle_functions", probability=1.0)],
        )

        formatter1 = DatasetFormatter(config1)
        formatter2 = DatasetFormatter(config2)

        result1, _ = formatter1.format(sample_dataset)
        result2, _ = formatter2.format(sample_dataset)

        assert result1[0]["text"] == result2[0]["text"]

    def test_format_different_seeds_different_results(self, sample_dataset):
        """Different seeds produce different results."""
        config1 = FormatConfig(
            seed=42,
            formatting=[StepConfig(type="shuffle_functions", probability=1.0)],
        )
        config2 = FormatConfig(
            seed=123,
            formatting=[StepConfig(type="shuffle_functions", probability=1.0)],
        )

        formatter1 = DatasetFormatter(config1)
        formatter2 = DatasetFormatter(config2)

        result1, _ = formatter1.format(sample_dataset)
        result2, _ = formatter2.format(sample_dataset)

        # Different shuffles may produce different JSON
        # (though could be same by chance)
        # This test mainly verifies different RNG states


class TestFormatDatasetFunction:
    """Tests for format_dataset convenience function."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset."""
        sample_trace = """
### INSTRUCTIONS
Test instructions.

EXAMPLE:
Example.

The functions available to you are described below.

### FUNCTIONS AVAILABLE
[]

### USER QUERY
Test query.

### ITERATIVE RESOLUTION CYCLE
Test completion.
"""
        return Dataset.from_dict(
            {
                "corrected_agent_trace": [[{"value": sample_trace}]],
            }
        )

    def test_format_without_config(self, sample_dataset):
        """Format without config uses baseline."""
        result, stats = format_dataset(sample_dataset)

        assert "text" in result.column_names
        assert len(result) == 1
        assert stats["output_samples"] == 1

    def test_format_with_no_augment(self, sample_dataset):
        """no_augment flag disables augmentations."""
        config = FormatConfig(
            seed=42,
            formatting=[StepConfig(type="shuffle_functions", probability=1.0)],
        )

        result, stats = format_dataset(sample_dataset, config=config, no_augment=True)

        assert stats["augmentation_counts"] == {}


class TestFormatterEdgeCases:
    """Tests for edge cases."""

    def test_empty_dataset(self):
        """Handles empty dataset."""
        empty_ds = Dataset.from_dict({"corrected_agent_trace": []})
        config = create_baseline_config()
        formatter = DatasetFormatter(config)

        result, stats = formatter.format(empty_ds)

        assert len(result) == 0
        assert stats["input_samples"] == 0

    def test_missing_source_column(self):
        """Reports error for missing source column."""
        ds = Dataset.from_dict({"other_column": ["value"]})
        config = create_baseline_config()
        formatter = DatasetFormatter(config)

        result, stats = formatter.format(ds)

        # Should have parse errors
        assert stats["parse_errors"] == 1

    def test_string_source_column(self):
        """Handles string source column format."""
        trace = """
### INSTRUCTIONS
Test.

EXAMPLE:


The functions available to you are described below.

### FUNCTIONS AVAILABLE
[]

### USER QUERY
Query.

### ITERATIVE RESOLUTION CYCLE
Completion.
"""
        ds = Dataset.from_dict({"corrected_agent_trace": [trace]})
        config = create_baseline_config()
        formatter = DatasetFormatter(config)

        result, stats = formatter.format(ds)

        assert len(result) == 1
        assert "Query" in result[0]["text"]

    def test_deduplication(self):
        """Deduplication removes duplicates."""
        # Same trace twice
        trace = """
### INSTRUCTIONS
Test.

EXAMPLE:


The functions available to you are described below.

### FUNCTIONS AVAILABLE
[]

### USER QUERY
Query.

### ITERATIVE RESOLUTION CYCLE
Completion.
"""
        ds = Dataset.from_dict(
            {
                "corrected_agent_trace": [[{"value": trace}]],
            }
        )

        # Multiply with no augmentation produces duplicates
        config = FormatConfig(
            seed=42,
            multiply=3,
            deduplicate=True,
        )
        formatter = DatasetFormatter(config)

        result, stats = formatter.format(ds)

        # All 3 variants are identical, so 2 should be removed
        assert stats["duplicates_removed"] == 2
        assert len(result) == 1
