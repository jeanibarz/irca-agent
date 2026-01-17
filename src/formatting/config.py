"""
Configuration schema for dataset formatting.

Implements FR-DATA-27: Format Config.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class StepConfig(BaseModel):
    """Configuration for a single augmentation step."""

    type: str = Field(..., description="Step type identifier")
    probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of applying this step (0.0 to 1.0)",
    )
    params: dict[str, Any] | None = Field(
        default=None,
        description="Step-specific parameters",
    )


class FormatConfig(BaseModel):
    """
    Configuration for dataset formatting with staged augmentation.

    Augmentation stages:
    - structural: Applied before formatting (e.g., translation)
    - formatting: Applied during formatting (e.g., function shuffling)
    - presentation: Applied after formatting (e.g., newline variation)
    """

    seed: int = Field(default=42, description="Random seed for reproducibility")
    multiply: int = Field(
        default=1,
        ge=1,
        description="Dataset size multiplier (1 = no multiplication)",
    )
    deduplicate: bool = Field(
        default=True,
        description="Remove duplicate variants when multiply > 1",
    )
    source_column: str = Field(
        default="corrected_agent_trace",
        description="Column containing raw prompt data",
    )

    # Staged augmentation pipelines
    structural: list[StepConfig] = Field(
        default_factory=list,
        description="Steps applied before formatting (on parsed components)",
    )
    formatting: list[StepConfig] = Field(
        default_factory=list,
        description="Steps applied during formatting (on structure)",
    )
    presentation: list[StepConfig] = Field(
        default_factory=list,
        description="Steps applied after formatting (on final text)",
    )

    @field_validator("structural", "formatting", "presentation", mode="before")
    @classmethod
    def validate_steps(cls, v: Any) -> list[StepConfig]:
        """Convert dict steps to StepConfig objects."""
        if v is None:
            return []
        if isinstance(v, list):
            return [StepConfig(**item) if isinstance(item, dict) else item for item in v]
        return v

    def has_augmentations(self) -> bool:
        """Check if any augmentation steps are configured."""
        return bool(self.structural or self.formatting or self.presentation)

    def get_all_steps(self) -> list[tuple[str, StepConfig]]:
        """Get all steps with their stage names."""
        steps = []
        for step in self.structural:
            steps.append(("structural", step))
        for step in self.formatting:
            steps.append(("formatting", step))
        for step in self.presentation:
            steps.append(("presentation", step))
        return steps


def load_format_config(config_path: str | Path) -> FormatConfig:
    """
    Load format configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        Validated FormatConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(path) as f:
            data = json.load(f)
        return FormatConfig(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}") from e
    except Exception as e:
        raise ValueError(f"Invalid config: {e}") from e


def create_baseline_config(source_column: str = "corrected_agent_trace") -> FormatConfig:
    """
    Create a baseline config with no augmentations.

    Args:
        source_column: Column containing raw prompt data

    Returns:
        FormatConfig with no augmentation steps
    """
    return FormatConfig(
        seed=42,
        multiply=1,
        source_column=source_column,
        structural=[],
        formatting=[],
        presentation=[],
    )
