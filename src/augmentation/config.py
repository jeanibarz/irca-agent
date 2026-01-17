"""
Configuration schema and validation for augmentation pipeline.

Implements FR-DATA-12: Pipeline Configuration.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TranslateParams(BaseModel):
    """Parameters for translation step."""

    languages: list[str] = Field(..., min_length=1, description="Target language codes")
    weights: list[float] | None = Field(default=None, description="Optional weights for language selection")

    @model_validator(mode="after")
    def validate_weights(self) -> "TranslateParams":
        if self.weights is not None:
            if len(self.weights) != len(self.languages):
                raise ValueError(
                    f"weights length ({len(self.weights)}) must match languages length ({len(self.languages)})"
                )
            if abs(sum(self.weights) - 1.0) > 0.01:
                raise ValueError(f"weights must sum to 1.0, got {sum(self.weights)}")
        return self


class PipelineStep(BaseModel):
    """A single step in the augmentation pipeline."""

    type: str = Field(..., description="Step type: translate, shuffle_functions, newline_variation")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of applying this step")
    params: dict[str, Any] | None = Field(default=None, description="Step-specific parameters")

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        # NOTE: format_variation removed - see ADR-006 (breaks chat template parsing)
        valid_types = {"translate", "shuffle_functions", "newline_variation"}
        if v not in valid_types:
            raise ValueError(f"Invalid step type '{v}'. Valid types: {valid_types}")
        return v

    @model_validator(mode="after")
    def validate_params(self) -> "PipelineStep":
        if self.type == "translate":
            if self.params is None:
                raise ValueError("translate step requires 'params' with 'languages'")
            # Validate translate params
            TranslateParams(**self.params)
        return self


class AugmentationConfig(BaseModel):
    """
    Configuration for dataset augmentation pipeline.

    Implements FR-DATA-12: Pipeline Configuration.
    """

    input: str | None = Field(default=None, description="Input dataset path (can be overridden via CLI)")
    output: str | None = Field(default=None, description="Output dataset path (can be overridden via CLI)")
    multiply: int = Field(..., ge=1, description="Size multiplier (e.g., 3 = 3x samples)")
    seed: int = Field(..., description="Random seed for reproducibility")
    deduplicate: bool | str = Field(default=True, description="Deduplicate variants: true, false, or 'fuzzy'")
    pipeline: list[PipelineStep] = Field(..., min_length=1, description="Ordered list of augmentation steps")

    @field_validator("deduplicate")
    @classmethod
    def validate_deduplicate(cls, v: bool | str) -> bool | str:
        if isinstance(v, str) and v not in ("fuzzy",):
            raise ValueError(f"Invalid deduplicate value '{v}'. Use true, false, or 'fuzzy'")
        return v


def load_config(config_path: str | Path) -> AugmentationConfig:
    """
    Load and validate augmentation config from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        Validated AugmentationConfig

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e

    return validate_config(data)


def validate_config(data: dict[str, Any]) -> AugmentationConfig:
    """
    Validate configuration dictionary.

    Args:
        data: Configuration dictionary

    Returns:
        Validated AugmentationConfig

    Raises:
        ValueError: If config is invalid
    """
    try:
        return AugmentationConfig(**data)
    except Exception as e:
        raise ValueError(f"Config validation failed: {e}") from e
