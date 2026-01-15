"""
Centralized Settings for IRCA-Agent

All configuration is loaded from environment variables or .env file.
Use `get_settings()` to get a cached singleton instance.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseSettings):
    """Configuration for a specific model."""

    base_model: str
    model_name: str


class Settings(BaseSettings):
    """
    Main settings class for IRCA-Agent.

    All settings can be overridden via environment variables.
    Environment variable names are uppercase versions of the field names.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # ===========================================
    # Paths
    # ===========================================
    workspace_dir: Path = Field(
        default=Path("/workspace"),
        description="Base workspace directory",
    )
    models_dir: Path = Field(
        default=Path("models"),
        description="Directory for downloaded/finetuned models (relative to workspace)",
    )
    datasets_dir: Path = Field(
        default=Path("datasets"),
        description="Directory for datasets (relative to workspace)",
    )
    finetuned_models_dir: Path = Field(
        default=Path("finetuned_models"),
        description="Subdirectory for finetuned models (relative to models_dir)",
    )

    # ===========================================
    # API Keys & Tokens
    # ===========================================
    huggingface_token: str | None = Field(
        default=None,
        description="HuggingFace Hub access token",
    )
    argilla_api_url: str = Field(
        default="http://localhost:6900",
        description="Argilla server URL",
    )
    argilla_api_key: str | None = Field(
        default=None,
        description="Argilla API key",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (optional, for GPT-based generation)",
    )
    wandb_api_key: str | None = Field(
        default=None,
        description="Weights & Biases API key",
    )
    wandb_project: str = Field(
        default="irca-agent",
        description="W&B project name",
    )
    wandb_entity: str | None = Field(
        default=None,
        description="W&B entity/team name",
    )

    # ===========================================
    # Training Configuration
    # ===========================================
    dataset: str = Field(
        default="JeanIbarz/irca_agent_dataset_v5-5acc",
        description="HuggingFace dataset ID for training",
    )
    version: str = Field(
        default="v5-6",
        description="Version suffix for model names",
    )
    push_to_hub: bool = Field(
        default=False,
        description="Whether to push trained model to HuggingFace Hub",
    )

    # LoRA Configuration
    lora_r: int = Field(
        default=128,
        description="LoRA rank (r parameter)",
        ge=1,
    )
    lora_alpha: int = Field(
        default=64,
        description="LoRA alpha parameter",
        ge=1,
    )
    lora_dropout: float = Field(
        default=0.05,
        description="LoRA dropout rate",
        ge=0.0,
        le=1.0,
    )

    # Training Parameters
    num_train_epochs: int = Field(
        default=5,
        description="Number of training epochs",
        ge=1,
    )
    learning_rate: float = Field(
        default=2e-4,
        description="Learning rate",
        gt=0,
    )
    max_seq_length: int = Field(
        default=4096,
        description="Maximum sequence length for training",
        ge=512,
    )

    # ===========================================
    # Model Presets
    # ===========================================
    default_model_type: Literal["mistral", "tinyllama", "qwen", "custom"] = Field(
        default="mistral",
        description="Default model type to use",
    )

    # ===========================================
    # Computed Properties
    # ===========================================
    @property
    def models_path(self) -> Path:
        """Full path to models directory."""
        return self.workspace_dir / self.models_dir

    @property
    def datasets_path(self) -> Path:
        """Full path to datasets directory."""
        return self.workspace_dir / self.datasets_dir

    @property
    def finetuned_models_path(self) -> Path:
        """Full path to finetuned models directory."""
        return self.models_path / self.finetuned_models_dir

    @property
    def dotenv_path(self) -> Path:
        """Path to .env file."""
        return self.workspace_dir / ".env"

    # ===========================================
    # Model Configurations
    # ===========================================
    def get_model_config(self, model_type: str | None = None) -> dict[str, str]:
        """
        Get configuration for a specific model type.

        Args:
            model_type: One of 'mistral', 'tinyllama', 'qwen', etc.
                       Uses default_model_type if None.

        Returns:
            Dictionary with 'base_model' and 'model_name' keys.
        """
        model_type = model_type or self.default_model_type

        # Model presets - can be extended or overridden via environment
        presets = {
            "mistral": {
                "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "model_name": f"Mistral-7B-Instruct-v0.2_irca_agent_{self.version}",
            },
            "mistral-v3": {
                "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
                "model_name": f"Mistral-7B-Instruct-v0.3_irca_agent_{self.version}",
            },
            "tinyllama": {
                "base_model": "Doctor-Shotgun/TinyLlama-1.1B-32k",
                "model_name": f"TinyLlama-1.1B_irca_agent_{self.version}",
            },
            "qwen-4b": {
                "base_model": "Qwen/Qwen1.5-4B-Chat",
                "model_name": f"Qwen1.5-4B_irca_agent_{self.version}",
            },
            "qwen-14b": {
                "base_model": "Qwen/Qwen1.5-14B-Chat",
                "model_name": f"Qwen1.5-14B_irca_agent_{self.version}",
            },
        }

        if model_type not in presets:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(presets.keys())}")

        return presets[model_type]

    def get_training_config(self, model_type: str | None = None) -> dict[str, Any]:
        """
        Get complete training configuration.

        Args:
            model_type: Model type to use. Uses default if None.

        Returns:
            Complete configuration dictionary for training.
        """
        model_config = self.get_model_config(model_type)

        return {
            # Dataset
            "dataset": self.dataset,
            "push_to_hub": self.push_to_hub,
            # Model
            "base_model": model_config["base_model"],
            "model_name": model_config["model_name"],
            # LoRA
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            # Training
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "max_seq_length": self.max_seq_length,
            # Paths
            "output_dir": str(self.finetuned_models_path / model_config["model_name"]),
        }

    @field_validator("workspace_dir", "models_dir", "datasets_dir", "finetuned_models_dir", mode="before")
    @classmethod
    def validate_path(cls, v: Any) -> Any:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function is cached, so it only loads settings once.
    Call `get_settings.cache_clear()` to reload settings.

    Returns:
        Settings instance with all configuration.
    """
    return Settings()
