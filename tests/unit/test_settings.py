"""
Tests for Settings Configuration

Tests the Settings class and configuration loading from core.config
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from config import Settings, get_settings, clear_settings_cache


class TestSettings:
    """Tests for Settings class."""

    def test_settings_default_values(self):
        """Settings should have sensible defaults."""
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings()

            # workspace_dir defaults to cwd() via default_factory
            assert settings.workspace_dir == Path.cwd()
            assert settings.models_dir == Path("models")
            assert settings.datasets_dir == Path("datasets")
            # Current defaults optimized for Unsloth on 24GB GPU
            assert settings.lora_r == 16
            assert settings.lora_alpha == 16
            assert settings.num_train_epochs == 3

    def test_settings_from_environment(self):
        """Settings can be loaded from environment variables."""
        env_vars = {
            "WORKSPACE_DIR": "/custom/workspace",
            "HUGGINGFACE_TOKEN": "test_token",
            "LORA_R": "64",
            "NUM_TRAIN_EPOCHS": "10",
        }
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()

            assert settings.workspace_dir == Path("/custom/workspace")
            assert settings.huggingface_token == "test_token"
            assert settings.lora_r == 64
            assert settings.num_train_epochs == 10

    def test_settings_computed_paths(self):
        """Computed path properties work correctly."""
        settings = Settings(
            workspace_dir=Path("/workspace"),
            models_dir=Path("models"),
            datasets_dir=Path("datasets"),
            finetuned_models_dir=Path("finetuned"),
        )

        assert settings.models_path == Path("/workspace/models")
        assert settings.datasets_path == Path("/workspace/datasets")
        assert settings.finetuned_models_path == Path("/workspace/models/finetuned")


class TestGetModelConfig:
    """Tests for Settings.get_model_config method."""

    def test_get_mistral_config(self):
        """Get configuration for Mistral model."""
        settings = Settings()
        config = settings.get_model_config("mistral")

        assert "base_model" in config
        assert "model_name" in config
        assert "mistral" in config["base_model"].lower()

    def test_get_tinyllama_config(self):
        """Get configuration for TinyLlama model."""
        settings = Settings()
        config = settings.get_model_config("tinyllama")

        assert "tinyllama" in config["base_model"].lower() or "TinyLlama" in config["base_model"]

    def test_get_unknown_model_raises(self):
        """Unknown model type raises ValueError."""
        settings = Settings()

        with pytest.raises(ValueError, match="Unknown model type"):
            settings.get_model_config("unknown_model")

    def test_get_model_config_default(self):
        """get_model_config uses default_model_type when None."""
        settings = Settings(default_model_type="mistral")
        config = settings.get_model_config(None)

        assert "mistral" in config["base_model"].lower()


class TestGetTrainingConfig:
    """Tests for Settings.get_training_config method."""

    def test_get_training_config_complete(self):
        """get_training_config returns all required fields."""
        settings = Settings()
        config = settings.get_training_config("mistral")

        required_fields = [
            "dataset",
            "base_model",
            "model_name",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "num_train_epochs",
            "learning_rate",
            "output_dir",
        ]

        for field in required_fields:
            assert field in config, f"Missing field: {field}"

    def test_get_training_config_uses_settings_values(self):
        """Training config uses values from settings."""
        settings = Settings(
            lora_r=256,
            learning_rate=2e-4,
            num_train_epochs=10,
        )
        config = settings.get_training_config("mistral")

        assert config["lora_r"] == 256
        assert config["learning_rate"] == 2e-4
        assert config["num_train_epochs"] == 10

    def test_get_training_config_output_dir(self):
        """Training config includes correct output directory."""
        settings = Settings(
            workspace_dir=Path("/test/workspace"),
            models_dir=Path("models"),
            finetuned_models_dir=Path("finetuned"),
        )
        config = settings.get_training_config("mistral")

        assert "/test/workspace/models/finetuned" in config["output_dir"]


class TestGetSettings:
    """Tests for get_settings cached function."""

    def test_get_settings_returns_settings(self):
        """get_settings returns a Settings instance."""
        # Clear cache first
        clear_settings_cache()

        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self):
        """get_settings returns the same instance."""
        clear_settings_cache()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_cache_can_be_cleared(self):
        """Cache can be cleared to reload settings."""
        clear_settings_cache()
        settings1 = get_settings()

        clear_settings_cache()
        settings2 = get_settings()

        # Different instances after cache clear
        # (may be equal but not identical)
        assert settings1 is not settings2
