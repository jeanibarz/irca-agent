"""
Unit tests for dataset validation.

Tests:
- FR-VAL-001: Pre-flight dataset validation
- FR-VAL-002: Marker integrity verification
- FM-12: Augmented data may corrupt structural markers
"""

import pytest
from datasets import Dataset

from src.core.constants import (
    MARKER_FINAL_ANSWER,
    MARKER_FUNCTIONS,
    MARKER_INSTRUCTIONS,
    MARKER_ITERATIVE_CYCLE,
    MARKER_USER_QUERY,
    REQUIRED_MARKERS,
)
from src.validation.dataset import (
    ValidationResult,
    _check_marker_integrity,
    validate_augmentation_distribution,
    validate_dataset_structure,
    validate_marker_integrity,
    validate_text_column,
)


# ============================================
# Sample Data Fixtures
# ============================================


@pytest.fixture
def valid_irca_text() -> str:
    """Valid IRCA-formatted text with all markers."""
    return """
### INSTRUCTIONS
You are an AI assistant that uses functions.

EXAMPLE:
User: Test
Assistant: OK

The functions available to you are described below.

### FUNCTIONS AVAILABLE
[{"name": "get_weather", "description": "Get weather"}]

### USER QUERY
What is the weather in Paris?

### ITERATIVE RESOLUTION CYCLE
Thought: I should get the weather.
Action: call function
Call: get_weather(location="Paris")
Output: {"temp": 20}
Thought: I have the answer.

### FINAL ANSWER
The weather in Paris is 20 degrees.
""".strip()


@pytest.fixture
def valid_dataset(valid_irca_text: str) -> Dataset:
    """Dataset with valid IRCA-formatted samples."""
    return Dataset.from_list(
        [
            {
                "text": valid_irca_text,
                "original_index": 0,
                "variant_id": 0,
                "augmentations": [],
            },
            {
                "text": valid_irca_text.replace("Paris", "London"),
                "original_index": 1,
                "variant_id": 0,
                "augmentations": ["translate:fr"],
            },
        ]
    )


@pytest.fixture
def corrupted_marker_text() -> str:
    """Text with corrupted markers (simulating augmentation corruption)."""
    return """
## INSTRUCTIONS
You are an AI assistant.

FUNCTIONS:
[{"name": "test"}]

USER:
What is the weather?

RESPONSE:
The weather is nice.
""".strip()


# ============================================
# Test Marker Integrity (FR-VAL-002, FM-12)
# ============================================


class TestMarkerIntegrity:
    """Tests for IRCA marker integrity validation."""

    def test_ut_val_001_markers_preserved_after_shuffle(self, valid_irca_text: str):
        """UT-VAL-001: Markers preserved after shuffle_functions."""
        # Simulate shuffle_functions augmentation (just reorder functions)
        text = valid_irca_text.replace(
            '[{"name": "get_weather", "description": "Get weather"}]',
            '[{"name": "get_weather", "description": "Get weather"}, {"name": "search"}]',
        )

        issues = _check_marker_integrity(text, sample_idx=0)
        errors = [i for i in issues if i["severity"] == "error"]

        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_ut_val_002_markers_preserved_after_translate(self, valid_irca_text: str):
        """UT-VAL-002: Markers preserved after translate augmentation."""
        # Simulate translate augmentation (translate user query and answer)
        text = valid_irca_text.replace(
            "What is the weather in Paris?",
            "Quel temps fait-il à Paris?",
        ).replace(
            "The weather in Paris is 20 degrees.",
            "Le temps à Paris est de 20 degrés.",
        )

        issues = _check_marker_integrity(text, sample_idx=0)
        errors = [i for i in issues if i["severity"] == "error"]

        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_ut_val_003_markers_preserved_after_indent(self, valid_irca_text: str):
        """UT-VAL-003: Markers preserved after indent_functions augmentation."""
        # Simulate indent_functions (indent JSON)
        text = valid_irca_text.replace(
            '[{"name": "get_weather", "description": "Get weather"}]',
            '[\n  {\n    "name": "get_weather",\n    "description": "Get weather"\n  }\n]',
        )

        issues = _check_marker_integrity(text, sample_idx=0)
        errors = [i for i in issues if i["severity"] == "error"]

        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_ut_val_004_detect_broken_markers_from_format_variation(
        self, corrupted_marker_text: str
    ):
        """UT-VAL-004: Detect broken markers from format_variation."""
        issues = _check_marker_integrity(corrupted_marker_text, sample_idx=0)
        errors = [i for i in issues if i["severity"] == "error"]

        # Should detect missing required markers
        assert len(errors) >= 4, "Should detect all missing required markers"
        error_markers = [e.get("marker") for e in errors if "marker" in e]
        for marker in REQUIRED_MARKERS:
            assert marker in error_markers, f"Should detect missing {marker}"

    def test_ut_val_005_final_answer_marker_present(self, valid_irca_text: str):
        """UT-VAL-005: FINAL ANSWER marker present in completion."""
        issues = _check_marker_integrity(valid_irca_text, sample_idx=0)

        # Should not have warning about missing FINAL ANSWER
        warnings = [i for i in issues if i["severity"] == "warning"]
        final_answer_warnings = [w for w in warnings if MARKER_FINAL_ANSWER in w.get("issue", "")]

        assert len(final_answer_warnings) == 0, "Should not warn about missing FINAL ANSWER"

    def test_ut_val_005b_missing_final_answer_warning(self, valid_irca_text: str):
        """UT-VAL-005b: Warning when FINAL ANSWER marker missing."""
        # Remove FINAL ANSWER marker
        text = valid_irca_text.replace(MARKER_FINAL_ANSWER, "ANSWER:")

        issues = _check_marker_integrity(text, sample_idx=0)
        warnings = [i for i in issues if i["severity"] == "warning"]

        assert len(warnings) >= 1, "Should warn about missing FINAL ANSWER"

    def test_marker_order_validation(self, valid_irca_text: str):
        """Test that marker order is validated."""
        # Swap INSTRUCTIONS and FUNCTIONS markers (invalid order)
        text = valid_irca_text.replace(
            "### INSTRUCTIONS",
            "PLACEHOLDER_INSTRUCTIONS",
        ).replace(
            "### FUNCTIONS AVAILABLE",
            "### INSTRUCTIONS",  # This is wrong
        ).replace(
            "PLACEHOLDER_INSTRUCTIONS",
            "### FUNCTIONS AVAILABLE",  # This is wrong
        )

        issues = _check_marker_integrity(text, sample_idx=0)
        errors = [i for i in issues if i["severity"] == "error"]

        assert len(errors) >= 1, "Should detect out-of-order markers"


class TestTextColumnValidation:
    """Tests for text column validation."""

    def test_valid_text_column(self, valid_dataset: Dataset):
        """Test validation passes for valid text column."""
        result = validate_text_column(valid_dataset)
        assert result.passed
        assert result.check_name == "text_column_valid"

    def test_missing_text_column(self):
        """Test validation fails for missing text column."""
        dataset = Dataset.from_list([{"not_text": "foo"}])
        result = validate_text_column(dataset)

        assert not result.passed
        assert "missing" in result.message.lower()

    def test_empty_text_values(self):
        """Test validation fails for empty text values."""
        dataset = Dataset.from_list([{"text": ""}, {"text": "valid"}])
        result = validate_text_column(dataset)

        assert not result.passed
        assert "empty" in result.message.lower()

    def test_parse_error_markers(self):
        """Test validation detects PARSE ERROR markers."""
        dataset = Dataset.from_list(
            [{"text": "[PARSE ERROR: something went wrong]"}, {"text": "valid text"}]
        )
        result = validate_text_column(dataset)

        assert not result.passed
        assert "PARSE ERROR" in result.message


class TestDatasetStructureValidation:
    """Tests for dataset structure validation."""

    def test_valid_structure(self, valid_dataset: Dataset):
        """Test validation passes for valid structure."""
        result = validate_dataset_structure(valid_dataset)
        assert result.passed

    def test_empty_dataset(self):
        """Test validation fails for empty dataset."""
        # An empty dataset with proper columns
        dataset = Dataset.from_list([])
        # Add text column schema
        dataset = dataset.add_column("text", [])
        result = validate_dataset_structure(dataset)

        assert not result.passed
        assert "empty" in result.message.lower()

    def test_missing_expected_columns_warning(self):
        """Test warning for missing expected columns."""
        dataset = Dataset.from_list([{"text": "some text"}])
        result = validate_dataset_structure(dataset)

        assert result.passed  # Should pass (not required)
        assert result.severity == "warning" or result.details.get("warnings")


class TestMarkerIntegrityDataset:
    """Tests for marker integrity on full datasets."""

    def test_valid_dataset_markers(self, valid_dataset: Dataset):
        """Test marker validation passes for valid dataset."""
        result = validate_marker_integrity(valid_dataset, sample_size=10)
        assert result.passed

    def test_corrupted_dataset_markers(self, corrupted_marker_text: str):
        """Test marker validation fails for corrupted dataset."""
        dataset = Dataset.from_list([{"text": corrupted_marker_text}])
        result = validate_marker_integrity(dataset, sample_size=10)

        assert not result.passed
        assert "integrity" in result.check_name


class TestAugmentationDistribution:
    """Tests for augmentation distribution validation."""

    def test_augmented_dataset(self, valid_dataset: Dataset):
        """Test augmentation distribution for augmented dataset."""
        result = validate_augmentation_distribution(valid_dataset)
        assert result.passed

    def test_baseline_dataset(self):
        """Test validation for baseline dataset without augmentations column."""
        dataset = Dataset.from_list([{"text": "some text"}])
        result = validate_augmentation_distribution(dataset)

        assert result.passed
        assert "baseline" in result.message.lower()


# ============================================
# Test Constants (centralized markers)
# ============================================


class TestCentralizedConstants:
    """Tests for centralized marker constants."""

    def test_required_markers_defined(self):
        """Test that all required markers are defined."""
        assert MARKER_INSTRUCTIONS is not None
        assert MARKER_FUNCTIONS is not None
        assert MARKER_USER_QUERY is not None
        assert MARKER_ITERATIVE_CYCLE is not None
        assert MARKER_FINAL_ANSWER is not None

    def test_required_markers_list(self):
        """Test that REQUIRED_MARKERS contains expected markers."""
        assert MARKER_INSTRUCTIONS in REQUIRED_MARKERS
        assert MARKER_FUNCTIONS in REQUIRED_MARKERS
        assert MARKER_USER_QUERY in REQUIRED_MARKERS
        assert MARKER_ITERATIVE_CYCLE in REQUIRED_MARKERS

    def test_markers_are_strings(self):
        """Test that all markers are non-empty strings."""
        for marker in REQUIRED_MARKERS:
            assert isinstance(marker, str)
            assert len(marker) > 0
            assert marker.startswith("###")
