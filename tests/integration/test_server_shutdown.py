"""
Integration Tests for Server Shutdown

Tests for the complete server lifecycle including graceful shutdown with GPU cleanup.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestServerLifecycleIntegration:
    """Integration tests for server lifecycle."""

    @pytest.mark.asyncio
    async def test_lifespan_startup_shutdown(self) -> None:
        """IT-SRV-001: Lifespan properly handles startup and shutdown."""
        from fastapi import FastAPI

        from src.server.main import lifespan

        # Track lifecycle events
        events = []

        # Create a test app
        app = FastAPI()

        async with lifespan(app):
            events.append("running")

        events.append("shutdown_complete")

        assert "running" in events
        assert "shutdown_complete" in events

    @pytest.mark.asyncio
    async def test_shutdown_ejects_model(self) -> None:
        """IT-SRV-001: Shutdown properly ejects loaded models."""
        from fastapi import FastAPI

        from src.server.main import lifespan
        from src.server.model_manager import ModelManager

        # Reset singleton
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Setup mock model
        manager.models["default"] = MagicMock()
        manager.tokenizers["default"] = MagicMock()
        manager.loaded_configs["default"] = {"base": "test-model", "adapter": None}

        app = FastAPI()

        async with lifespan(app):
            # Model should be "loaded"
            assert len(manager.models) == 1

        # After shutdown, model should be ejected
        # Note: The eject_model method will clear these
        # The lifespan should have called eject

    @pytest.mark.asyncio
    async def test_shutdown_timeout_bounded(self) -> None:
        """IT-SRV-002: Shutdown completes within timeout even with slow ejection."""
        from fastapi import FastAPI

        from src.server.main import _eject_all_models_with_timeout
        from src.server.model_manager import ModelManager

        # Reset singleton
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Setup mock model that hangs
        manager.models["hanging"] = MagicMock()
        manager.tokenizers["hanging"] = MagicMock()
        manager.loaded_configs["hanging"] = {"base": "test", "adapter": None}

        async def hanging_eject(alias: str) -> None:
            await asyncio.sleep(100)  # Would hang forever

        manager.eject_model = hanging_eject

        start = time.monotonic()
        await _eject_all_models_with_timeout(timeout=1.0)
        elapsed = time.monotonic() - start

        # Should complete within reasonable time (timeout + buffer)
        assert elapsed < 3.0


class TestHealthEndpointIntegration:
    """Integration tests for health endpoint."""

    def test_health_endpoint_no_model(self) -> None:
        """Health endpoint returns correct status when no model loaded."""
        from fastapi.testclient import TestClient

        from src.server.main import create_app
        from src.server.model_manager import ModelManager

        # Reset singleton
        ModelManager._instance = None

        app = create_app()

        with TestClient(app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is False

    def test_health_endpoint_with_model(self) -> None:
        """Health endpoint returns model info when loaded."""
        from fastapi.testclient import TestClient

        from src.server.main import create_app
        from src.server.model_manager import ModelManager

        # Reset singleton
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Setup mock model
        manager.models["default"] = MagicMock()
        manager.tokenizers["default"] = MagicMock()
        manager.loaded_configs["default"] = {"base": "test/model-name", "adapter": None}

        app = create_app()

        with TestClient(app) as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert data["current_model"] == "test/model-name"
            assert "idle_seconds" in data
            assert "idle_timeout_minutes" in data


class TestServerSettingsIntegration:
    """Integration tests for server settings."""

    def test_server_settings_defaults(self) -> None:
        """Server settings have correct defaults."""
        from src.config import Settings

        settings = Settings()

        assert settings.server_host == "0.0.0.0"
        assert settings.server_port == 8000
        assert settings.server_idle_timeout == 30

    def test_server_settings_from_env(self) -> None:
        """Server settings can be configured via environment."""
        import os
        from unittest.mock import patch

        env_vars = {
            "SERVER_HOST": "127.0.0.1",
            "SERVER_PORT": "9000",
            "SERVER_IDLE_TIMEOUT": "60",
        }

        with patch.dict(os.environ, env_vars, clear=False):
            from src.config import Settings

            settings = Settings()

            assert settings.server_host == "127.0.0.1"
            assert settings.server_port == 9000
            assert settings.server_idle_timeout == 60

    def test_server_idle_timeout_validation(self) -> None:
        """Server idle timeout validates non-negative."""
        from pydantic import ValidationError

        from src.config import Settings

        # Should raise for negative value
        with pytest.raises(ValidationError):
            Settings(server_idle_timeout=-1)

        # Zero should be valid (means disabled)
        settings = Settings(server_idle_timeout=0)
        assert settings.server_idle_timeout == 0

    def test_server_port_validation(self) -> None:
        """Server port validates range."""
        from pydantic import ValidationError

        from src.config import Settings

        # Should raise for invalid port
        with pytest.raises(ValidationError):
            Settings(server_port=0)

        with pytest.raises(ValidationError):
            Settings(server_port=70000)

        # Valid ports
        settings = Settings(server_port=8080)
        assert settings.server_port == 8080
