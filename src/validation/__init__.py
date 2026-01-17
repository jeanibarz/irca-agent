"""
Experiment Validation Framework.

This module provides pre-flight validation for experiments to catch
failure modes before expensive GPU operations.

Implements:
- FR-VAL-001: Pre-flight dataset validation
- FR-VAL-002: Marker integrity verification
- FR-VAL-003: Completion marker detection
- FR-VAL-004: System message preservation
- FR-VAL-005: GPU memory compatibility
- FR-VAL-006: Statistics API validation
- FR-EXP-001: Experiment pre-flight command

Example usage:
    from src.validation import run_preflight_checks

    report = run_preflight_checks(experiment_dir, sample_size=10)
    if not report.passed:
        print("Validation failed:", report.errors)
"""

from src.validation.completion import (
    CompletionDetectionResult,
    detect_completion_marker,
    get_model_family,
    validate_completion_detection,
)
from src.validation.dataset import (
    ValidationResult,
    validate_dataset_structure,
    validate_marker_integrity,
    validate_text_column,
)
from src.validation.experiment import (
    PreflightReport,
    run_preflight_checks,
    validate_experiment_config,
)

__all__ = [
    # Dataset validation
    "ValidationResult",
    "validate_text_column",
    "validate_marker_integrity",
    "validate_dataset_structure",
    # Completion detection
    "CompletionDetectionResult",
    "detect_completion_marker",
    "get_model_family",
    "validate_completion_detection",
    # Experiment validation
    "PreflightReport",
    "run_preflight_checks",
    "validate_experiment_config",
]
