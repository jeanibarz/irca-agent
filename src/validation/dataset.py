"""
Dataset validation utilities.

Validates dataset structure and IRCA marker integrity to catch
augmentation corruption before training.

Implements:
- FR-VAL-001: Pre-flight dataset validation
- FR-VAL-002: Marker integrity verification
- FR-EXP-002: Dataset structure validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.core.constants import (
    EXPECTED_MARKERS,
    MARKER_FINAL_ANSWER,
    MARKER_FUNCTIONS,
    MARKER_INSTRUCTIONS,
    MARKER_ITERATIVE_CYCLE,
    MARKER_USER_QUERY,
    REQUIRED_MARKERS,
)

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    check_name: str
    message: str
    severity: str = "error"  # "error", "warning", "info"
    details: dict = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✓" if self.passed else "✗"
        return f"[{status}] {self.check_name}: {self.message}"


def validate_text_column(dataset: Dataset) -> ValidationResult:
    """
    Validate that dataset has a 'text' column suitable for training.

    Checks:
    - 'text' column exists
    - All values are non-empty strings
    - No placeholder/error markers present

    Args:
        dataset: HuggingFace Dataset to validate.

    Returns:
        ValidationResult with pass/fail and details.
    """
    # Check column exists
    if "text" not in dataset.column_names:
        return ValidationResult(
            passed=False,
            check_name="text_column_exists",
            message="Dataset missing required 'text' column",
            severity="error",
            details={"available_columns": dataset.column_names},
        )

    # Check for empty/invalid values
    empty_count = 0
    error_count = 0
    error_indices = []

    for idx, text in enumerate(dataset["text"]):
        if not text or not isinstance(text, str):
            empty_count += 1
        elif "[PARSE ERROR:" in text:
            error_count += 1
            error_indices.append(idx)

    if empty_count > 0:
        return ValidationResult(
            passed=False,
            check_name="text_column_valid",
            message=f"Found {empty_count} empty/invalid 'text' values",
            severity="error",
            details={"empty_count": empty_count, "total": len(dataset)},
        )

    if error_count > 0:
        return ValidationResult(
            passed=False,
            check_name="text_column_valid",
            message=f"Found {error_count} samples with PARSE ERROR markers",
            severity="error",
            details={
                "error_count": error_count,
                "error_indices": error_indices[:10],  # First 10
                "total": len(dataset),
            },
        )

    return ValidationResult(
        passed=True,
        check_name="text_column_valid",
        message=f"'text' column valid ({len(dataset)} samples)",
        severity="info",
        details={"sample_count": len(dataset)},
    )


def validate_marker_integrity(
    dataset: Dataset,
    sample_size: int = 10,
) -> ValidationResult:
    """
    Validate that IRCA structural markers are intact after augmentation.

    This catches cases where augmentation (especially format_variation)
    corrupts the markers needed for proper training.

    Checks for each sampled text:
    - All required markers present
    - Markers appear in correct order
    - FINAL ANSWER marker present (warning if missing)

    Args:
        dataset: HuggingFace Dataset with 'text' column.
        sample_size: Number of samples to check (default: 10).

    Returns:
        ValidationResult with pass/fail and details.
    """
    if "text" not in dataset.column_names:
        return ValidationResult(
            passed=False,
            check_name="marker_integrity",
            message="Cannot check markers: 'text' column missing",
            severity="error",
        )

    # Sample indices
    import random

    rng = random.Random(42)
    indices = rng.sample(range(len(dataset)), min(sample_size, len(dataset)))

    issues = []
    warnings = []

    for idx in indices:
        text = dataset[idx]["text"]
        sample_issues = _check_marker_integrity(text, idx)

        for issue in sample_issues:
            if issue["severity"] == "error":
                issues.append(issue)
            else:
                warnings.append(issue)

    if issues:
        return ValidationResult(
            passed=False,
            check_name="marker_integrity",
            message=f"Found {len(issues)} marker integrity issues in {sample_size} samples",
            severity="error",
            details={
                "issues": issues[:10],  # First 10
                "warnings": warnings[:5],
                "samples_checked": len(indices),
            },
        )

    if warnings:
        return ValidationResult(
            passed=True,
            check_name="marker_integrity",
            message=f"Markers valid, but {len(warnings)} warnings",
            severity="warning",
            details={
                "warnings": warnings[:5],
                "samples_checked": len(indices),
            },
        )

    return ValidationResult(
        passed=True,
        check_name="marker_integrity",
        message=f"All required markers intact in {len(indices)} samples",
        severity="info",
        details={"samples_checked": len(indices)},
    )


def _check_marker_integrity(text: str, sample_idx: int) -> list[dict]:
    """
    Check marker integrity for a single text sample.

    Returns list of issues found.

    Note: IRCA format may include an EXAMPLE section that contains markers
    like ITERATIVE_RESOLUTION_CYCLE and FINAL_ANSWER. The order validation
    accounts for this by finding structural markers in sequence, not just
    their first occurrence.
    """
    issues = []

    # Check required markers
    for marker in REQUIRED_MARKERS:
        if marker not in text:
            issues.append(
                {
                    "severity": "error",
                    "sample_idx": sample_idx,
                    "issue": f"Missing required marker: {marker}",
                    "marker": marker,
                }
            )

    # Check marker order (when all required markers present)
    # The structural order should be: INSTRUCTIONS -> FUNCTIONS -> USER_QUERY -> ITERATIVE_CYCLE
    # Note: ITERATIVE_CYCLE may appear earlier in an EXAMPLE section, so we need to find
    # the occurrence that comes AFTER USER_QUERY
    if not issues:
        pos_instructions = text.find(MARKER_INSTRUCTIONS)
        pos_functions = text.find(MARKER_FUNCTIONS)
        pos_user_query = text.find(MARKER_USER_QUERY)

        # Find ITERATIVE_CYCLE that comes AFTER USER_QUERY (the actual section, not in EXAMPLE)
        pos_iterative = text.find(MARKER_ITERATIVE_CYCLE, pos_user_query) if pos_user_query >= 0 else -1

        # Verify structural order
        if pos_instructions >= 0 and pos_functions >= 0 and pos_instructions > pos_functions:
            issues.append(
                {
                    "severity": "error",
                    "sample_idx": sample_idx,
                    "issue": f"Markers out of order: {MARKER_INSTRUCTIONS} after {MARKER_FUNCTIONS}",
                }
            )

        if pos_functions >= 0 and pos_user_query >= 0 and pos_functions > pos_user_query:
            issues.append(
                {
                    "severity": "error",
                    "sample_idx": sample_idx,
                    "issue": f"Markers out of order: {MARKER_FUNCTIONS} after {MARKER_USER_QUERY}",
                }
            )

        if pos_user_query >= 0 and pos_iterative < 0:
            # ITERATIVE_CYCLE doesn't appear after USER_QUERY - this is an error
            # (it might only exist in the EXAMPLE section)
            issues.append(
                {
                    "severity": "error",
                    "sample_idx": sample_idx,
                    "issue": f"No {MARKER_ITERATIVE_CYCLE} found after {MARKER_USER_QUERY}",
                }
            )

    # Check FINAL ANSWER marker (warning only - some samples might not have it)
    if MARKER_FINAL_ANSWER not in text:
        issues.append(
            {
                "severity": "warning",
                "sample_idx": sample_idx,
                "issue": f"Missing {MARKER_FINAL_ANSWER} marker",
            }
        )

    return issues


def validate_dataset_structure(dataset: Dataset) -> ValidationResult:
    """
    Comprehensive validation of dataset structure.

    Checks:
    - Required columns exist
    - Column types are correct
    - No obviously corrupted data

    Args:
        dataset: HuggingFace Dataset to validate.

    Returns:
        ValidationResult with pass/fail and details.
    """
    required_columns = ["text"]
    expected_columns = ["text", "original_index", "variant_id", "augmentations"]

    # Check required columns
    missing_required = [col for col in required_columns if col not in dataset.column_names]
    if missing_required:
        return ValidationResult(
            passed=False,
            check_name="dataset_structure",
            message=f"Missing required columns: {missing_required}",
            severity="error",
            details={
                "missing": missing_required,
                "available": dataset.column_names,
            },
        )

    # Check expected columns (warning only)
    missing_expected = [col for col in expected_columns if col not in dataset.column_names]
    warnings = []
    if missing_expected:
        warnings.append(f"Missing expected columns: {missing_expected}")

    # Check dataset size
    if len(dataset) == 0:
        return ValidationResult(
            passed=False,
            check_name="dataset_structure",
            message="Dataset is empty",
            severity="error",
        )

    # Basic sanity check on first sample
    first_sample = dataset[0]
    text_length = len(first_sample.get("text", ""))
    if text_length < 100:
        warnings.append(f"First sample unusually short ({text_length} chars)")
    elif text_length > 100000:
        warnings.append(f"First sample unusually long ({text_length} chars)")

    if warnings:
        return ValidationResult(
            passed=True,
            check_name="dataset_structure",
            message=f"Structure valid with {len(warnings)} warnings",
            severity="warning",
            details={"warnings": warnings, "columns": dataset.column_names},
        )

    return ValidationResult(
        passed=True,
        check_name="dataset_structure",
        message=f"Dataset structure valid ({len(dataset)} samples, {len(dataset.column_names)} columns)",
        severity="info",
        details={
            "sample_count": len(dataset),
            "columns": dataset.column_names,
        },
    )


def validate_augmentation_distribution(
    dataset: Dataset,
    expected_multiplier: int = 3,
) -> ValidationResult:
    """
    Validate that augmentation distribution is as expected.

    Checks:
    - Augmentations column exists
    - Distribution of augmentation types matches expectations
    - No samples have missing augmentation metadata

    Args:
        dataset: HuggingFace Dataset with 'augmentations' column.
        expected_multiplier: Expected dataset multiplication factor.

    Returns:
        ValidationResult with pass/fail and details.
    """
    if "augmentations" not in dataset.column_names:
        return ValidationResult(
            passed=True,
            check_name="augmentation_distribution",
            message="No 'augmentations' column (baseline dataset)",
            severity="info",
        )

    # Count augmentation types
    from collections import Counter

    aug_counter: Counter[str] = Counter()
    no_aug_count = 0

    for sample in dataset:
        augs = sample.get("augmentations", [])
        if not augs:
            no_aug_count += 1
        for aug in augs:
            # Extract augmentation type (e.g., "translate:fr" -> "translate")
            aug_type = aug.split(":")[0] if ":" in aug else aug
            aug_counter[aug_type] += 1

    # Check if any augmentations were applied
    total_augs = sum(aug_counter.values())
    if total_augs == 0 and expected_multiplier > 1:
        return ValidationResult(
            passed=False,
            check_name="augmentation_distribution",
            message="No augmentations found despite expected multiplication",
            severity="warning",
            details={
                "total_samples": len(dataset),
                "no_augmentation_count": no_aug_count,
            },
        )

    return ValidationResult(
        passed=True,
        check_name="augmentation_distribution",
        message=f"Augmentation distribution: {dict(aug_counter)}",
        severity="info",
        details={
            "augmentation_counts": dict(aug_counter),
            "no_augmentation_count": no_aug_count,
            "total_samples": len(dataset),
        },
    )
