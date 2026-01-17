"""
Dataset augmentation module.

Provides pipeline-based dataset augmentation with configurable transformation steps.
"""

from src.augmentation.config import AugmentationConfig, load_config, validate_config
from src.augmentation.pipeline import AugmentationPipeline

__all__ = [
    "AugmentationConfig",
    "AugmentationPipeline",
    "load_config",
    "validate_config",
]
