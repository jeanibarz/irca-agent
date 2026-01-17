"""
Metadata generation for augmentation reproducibility.

Implements FR-DATA-15: Augmentation Metadata.
"""

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.augmentation.config import AugmentationConfig

METADATA_FILENAME = "_augmentation_metadata.json"
SCHEMA_VERSION = "1.0"


def get_git_info() -> dict[str, Any]:
    """
    Get git repository information.

    Returns:
        Dictionary with commit_hash, branch, and dirty status
    """
    git_info = {
        "commit_hash": None,
        "branch": None,
        "dirty": None,
    }

    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["commit_hash"] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["branch"] = result.stdout.strip()

        # Check if dirty (uncommitted changes)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_info["dirty"] = len(result.stdout.strip()) > 0

    except (subprocess.TimeoutExpired, FileNotFoundError):
        # Git not available or timeout
        pass

    return git_info


def get_environment_info() -> dict[str, Any]:
    """
    Get environment and package version information.

    Returns:
        Dictionary with Python version and key package versions
    """
    env_info = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "key_packages": {},
    }

    # Try to get versions of key packages
    packages_to_check = ["transformers", "datasets", "torch", "pydantic"]

    for package in packages_to_check:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", None)
            if version:
                env_info["key_packages"][package] = version
        except ImportError:
            pass

    # Try to get irca version
    try:
        from src import __version__

        env_info["irca_version"] = __version__
    except (ImportError, AttributeError):
        env_info["irca_version"] = "unknown"

    return env_info


def compute_dataset_hash(dataset_path: str | Path, sample_size: int = 100) -> str | None:
    """
    Compute a hash of the dataset for identity tracking.

    Uses a sample of the dataset to compute a fast hash.

    Args:
        dataset_path: Path to the dataset
        sample_size: Number of samples to use for hashing

    Returns:
        SHA256 hash string or None if unable to compute
    """
    try:
        from datasets import load_from_disk

        dataset = load_from_disk(str(dataset_path))

        # Sample some rows for hashing
        n_samples = min(sample_size, len(dataset))
        sample_data = dataset.select(range(n_samples))

        # Create hash from text column or first column
        hasher = hashlib.sha256()
        for row in sample_data:
            text = row.get("text", str(row))
            hasher.update(text.encode() if isinstance(text, str) else str(text).encode())

        return f"sha256:{hasher.hexdigest()[:16]}"

    except Exception:
        return None


def generate_metadata(
    config: AugmentationConfig,
    input_path: str,
    output_path: str,
    statistics: dict[str, Any],
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Generate comprehensive metadata for augmentation run.

    Implements FR-DATA-15: Augmentation Metadata.

    Args:
        config: Augmentation configuration used
        input_path: Input dataset path
        output_path: Output dataset path
        statistics: Pipeline statistics dictionary
        duration_seconds: Optional duration of augmentation

    Returns:
        Complete metadata dictionary
    """
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git": get_git_info(),
        "config": config.model_dump(),
        "input": {
            "path": str(input_path),
            "hash": compute_dataset_hash(input_path),
            "num_samples": statistics.get("input_samples", 0),
        },
        "output": {
            "path": str(output_path),
            "num_samples": statistics.get("variants_generated", 0) - statistics.get("duplicates_removed", 0),
        },
        "statistics": {
            "augmentation_counts": dict(statistics.get("augmentation_counts", {})),
            "duplicates_removed": statistics.get("duplicates_removed", 0),
            "empty_variants": statistics.get("empty_variants", 0),
        },
        "environment": get_environment_info(),
    }

    if duration_seconds is not None:
        metadata["statistics"]["duration_seconds"] = round(duration_seconds, 2)

    return metadata


def save_metadata(metadata: dict[str, Any], output_path: str | Path) -> Path:
    """
    Save metadata to sidecar file.

    Args:
        metadata: Metadata dictionary
        output_path: Output dataset directory

    Returns:
        Path to the saved metadata file
    """
    output_dir = Path(output_path)
    metadata_path = output_dir / METADATA_FILENAME

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata_path


def load_metadata(dataset_path: str | Path) -> dict[str, Any] | None:
    """
    Load metadata from a dataset directory.

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Metadata dictionary or None if not found
    """
    metadata_path = Path(dataset_path) / METADATA_FILENAME

    if not metadata_path.exists():
        return None

    with open(metadata_path) as f:
        return json.load(f)
