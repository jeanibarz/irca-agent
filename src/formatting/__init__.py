"""
Dataset Formatting Module.

Provides tools for converting raw structured datasets to training-ready text format.
Implements ADR-003: Dataset Format Pipeline.

Key components:
- FormatConfig: Configuration schema for format operations
- DatasetFormatter: Main class for formatting datasets
- Staged augmentation steps (structural, formatting, presentation)
"""

from src.formatting.config import FormatConfig, load_format_config
from src.formatting.formatter import DatasetFormatter

__all__ = [
    "FormatConfig",
    "load_format_config",
    "DatasetFormatter",
]
