"""
Experiment pre-flight validation orchestration.

Provides comprehensive validation before expensive GPU operations,
catching common failure modes early.

Implements:
- FR-EXP-001: Experiment pre-flight command
- FR-EXP-004: Configuration consistency
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.validation.dataset import (
    ValidationResult,
    validate_augmentation_distribution,
    validate_dataset_structure,
    validate_marker_integrity,
    validate_text_column,
)

logger = logging.getLogger(__name__)


@dataclass
class PreflightReport:
    """Complete pre-flight validation report."""

    passed: bool
    errors: list[ValidationResult] = field(default_factory=list)
    warnings: list[ValidationResult] = field(default_factory=list)
    info: list[ValidationResult] = field(default_factory=list)
    experiment_config: dict | None = None
    datasets_validated: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"Pre-flight Validation: {status}"]
        lines.append(f"  Errors: {len(self.errors)}")
        lines.append(f"  Warnings: {len(self.warnings)}")
        lines.append(f"  Datasets: {len(self.datasets_validated)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "passed": self.passed,
            "errors": [
                {"check": e.check_name, "message": e.message, "details": e.details} for e in self.errors
            ],
            "warnings": [
                {"check": w.check_name, "message": w.message, "details": w.details} for w in self.warnings
            ],
            "info": [{"check": i.check_name, "message": i.message, "details": i.details} for i in self.info],
            "experiment_config": self.experiment_config,
            "datasets_validated": self.datasets_validated,
        }


def validate_experiment_config(config_path: Path) -> tuple[bool, dict[str, Any]]:
    """
    Validate experiment configuration file.

    Checks:
    - File exists and is valid JSON
    - Required fields present
    - Statistical test references are valid
    - Dataset paths exist

    Args:
        config_path: Path to experiment.json.

    Returns:
        Tuple of (valid, config_or_errors).
    """
    if not config_path.exists():
        return False, {"error": f"Config file not found: {config_path}"}

    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        return False, {"error": f"Invalid JSON: {e}"}

    issues = []

    # Check required fields
    required_fields = ["experiment_id", "datasets", "evaluations"]
    for field in required_fields:
        if field not in config:
            issues.append(f"Missing required field: {field}")

    # Check statistical_analysis.test value (FM-1: bootstrap_paired doesn't exist)
    stat_config = config.get("statistical_analysis", {})
    test_name = stat_config.get("test", "")
    valid_tests = ["bootstrap_confidence_interval", "bootstrap_difference_test"]

    if test_name and test_name not in valid_tests:
        issues.append(
            f"Invalid statistical test '{test_name}'. "
            f"Valid options: {valid_tests}. "
            "Note: 'bootstrap_paired' does not exist."
        )

    # Check dataset paths exist
    datasets = config.get("datasets", {})
    experiment_dir = config_path.parent

    for name, ds_config in datasets.items():
        if isinstance(ds_config, dict):
            path = ds_config.get("path")
            if path:
                path_obj = Path(path)
                if path_obj.is_absolute():
                    full_path = path_obj
                elif path.startswith("experiments/"):
                    # Path is relative to project root, not experiment dir
                    # Go up from experiment_dir to find project root
                    project_root = experiment_dir.parent.parent
                    full_path = project_root / path
                else:
                    # Path is relative to experiment dir
                    full_path = experiment_dir / path

                if not full_path.exists():
                    issues.append(f"Dataset path not found: {path}")

    if issues:
        return False, {"errors": issues, "config": config}

    return True, config


def run_preflight_checks(
    experiment_dir: Path,
    sample_size: int = 10,
) -> PreflightReport:
    """
    Run comprehensive pre-flight validation checks.

    Validates:
    - Experiment configuration
    - Dataset structure and text column
    - IRCA marker integrity
    - Augmentation distribution (if applicable)

    Args:
        experiment_dir: Path to experiment directory.
        sample_size: Number of samples to check for marker integrity.

    Returns:
        PreflightReport with all validation results.

    Example:
        >>> report = run_preflight_checks(Path("experiments/exp001"))
        >>> if not report.passed:
        ...     for error in report.errors:
        ...         print(f"ERROR: {error}")
    """
    import datasets

    report = PreflightReport(passed=True)

    # Validate experiment config
    config_path = experiment_dir / "experiment.json"
    if config_path.exists():
        valid, config_result = validate_experiment_config(config_path)
        if not valid:
            errors = config_result.get("errors", [config_result.get("error", "Unknown error")])
            for error in errors if isinstance(errors, list) else [errors]:
                report.errors.append(
                    ValidationResult(
                        passed=False,
                        check_name="experiment_config",
                        message=str(error),
                        severity="error",
                    )
                )
            report.passed = False
        else:
            report.experiment_config = config_result
            report.info.append(
                ValidationResult(
                    passed=True,
                    check_name="experiment_config",
                    message=f"Config valid: {config_result.get('experiment_id', 'unknown')}",
                    severity="info",
                )
            )
    else:
        report.warnings.append(
            ValidationResult(
                passed=True,
                check_name="experiment_config",
                message="No experiment.json found",
                severity="warning",
            )
        )

    # Find and validate datasets
    dataset_dirs = _find_dataset_directories(experiment_dir, report.experiment_config)

    for ds_name, ds_path in dataset_dirs.items():
        logger.info(f"Validating dataset: {ds_name} at {ds_path}")

        try:
            ds = datasets.load_from_disk(str(ds_path))
            if isinstance(ds, datasets.DatasetDict):
                ds = ds.get("train", ds.get(list(ds.keys())[0]))
        except Exception as e:
            report.errors.append(
                ValidationResult(
                    passed=False,
                    check_name=f"dataset_load:{ds_name}",
                    message=f"Failed to load dataset: {e}",
                    severity="error",
                )
            )
            report.passed = False
            continue

        report.datasets_validated.append(ds_name)

        # Run dataset validations
        validations = [
            validate_dataset_structure(ds),
            validate_text_column(ds),
            validate_marker_integrity(ds, sample_size),
        ]

        # Add augmentation distribution check if augmented dataset
        if "augment" in ds_name.lower():
            validations.append(validate_augmentation_distribution(ds))

        for result in validations:
            result.check_name = f"{ds_name}:{result.check_name}"
            if not result.passed:
                report.errors.append(result)
                report.passed = False
            elif result.severity == "warning":
                report.warnings.append(result)
            else:
                report.info.append(result)

    return report


def _find_dataset_directories(
    experiment_dir: Path,
    config: dict | None,
) -> dict[str, Path]:
    """
    Find dataset directories to validate.

    Uses experiment config if available, otherwise searches for common patterns.
    """
    dataset_dirs: dict[str, Path] = {}

    # Try to find datasets from config
    if config:
        for name, ds_config in config.get("datasets", {}).items():
            if isinstance(ds_config, dict):
                path = ds_config.get("path", "")
                if path:
                    full_path = experiment_dir / path if not Path(path).is_absolute() else Path(path)
                    # Resolve relative paths that start with experiment dir
                    if str(path).startswith("experiments/"):
                        full_path = experiment_dir.parent.parent / path
                    if full_path.exists():
                        dataset_dirs[name] = full_path

    # If no datasets found from config, search directory
    if not dataset_dirs:
        for subdir in experiment_dir.iterdir():
            if subdir.is_dir() and (subdir / "dataset_info.json").exists():
                dataset_dirs[subdir.name] = subdir

    return dataset_dirs


def validate_gpu_memory_compatibility(
    model_name: str,
    eval_precision: str = "float16",
    training_precision: str = "4bit",
) -> ValidationResult:
    """
    Validate GPU memory compatibility between training and evaluation.

    Warns if evaluation uses higher precision than training (OOM risk).

    Args:
        model_name: Model being evaluated.
        eval_precision: Precision for evaluation.
        training_precision: Precision used during training.

    Returns:
        ValidationResult with compatibility status.
    """
    precision_memory = {
        "float32": 4.0,
        "float16": 2.0,
        "bfloat16": 2.0,
        "8bit": 1.0,
        "4bit": 0.5,
    }

    eval_mem = precision_memory.get(eval_precision, 2.0)
    train_mem = precision_memory.get(training_precision, 0.5)

    if eval_mem > train_mem * 2:
        return ValidationResult(
            passed=True,  # Warning, not failure
            check_name="gpu_memory_compatibility",
            message=f"Evaluation uses {eval_precision} ({eval_mem}x) vs training {training_precision} ({train_mem}x). OOM risk.",
            severity="warning",
            details={
                "model": model_name,
                "eval_precision": eval_precision,
                "training_precision": training_precision,
            },
        )

    return ValidationResult(
        passed=True,
        check_name="gpu_memory_compatibility",
        message=f"Memory compatible: eval={eval_precision}, train={training_precision}",
        severity="info",
    )


def validate_statistics_api() -> ValidationResult:
    """
    Validate that statistical functions are available.

    Addresses FM-1: bootstrap_paired doesn't exist (documentation bug).

    Returns:
        ValidationResult with API validation status.
    """
    from src.diversity import statistics

    issues = []
    available = []

    # Check expected functions exist
    expected = [
        "bootstrap_confidence_interval",
        "bootstrap_difference_test",
        "effect_size_cohens_d",
        "interpret_effect_size",
    ]

    for func_name in expected:
        if hasattr(statistics, func_name):
            available.append(func_name)
        else:
            issues.append(f"Missing function: {func_name}")

    # Explicitly check that bootstrap_paired does NOT exist (it's a doc bug)
    if hasattr(statistics, "bootstrap_paired"):
        issues.append("Unexpected function 'bootstrap_paired' found - should not exist")
    else:
        available.append("(confirmed: bootstrap_paired does not exist)")

    if issues:
        return ValidationResult(
            passed=False,
            check_name="statistics_api",
            message=f"Statistics API issues: {', '.join(issues)}",
            severity="error",
            details={"issues": issues, "available": available},
        )

    return ValidationResult(
        passed=True,
        check_name="statistics_api",
        message=f"Statistics API valid: {len(available)} functions available",
        severity="info",
        details={"available": available},
    )
