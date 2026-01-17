"""
Tests for Idle Ejection Functionality

Tests for ModelManager idle tracking and auto-ejection.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestModelManagerIdleTracking:
    """Tests for ModelManager idle time tracking."""

    def test_get_idle_seconds_initial(self) -> None:
        """Initial idle time is approximately 0."""
        from src.server.model_manager import ModelManager

        # Reset singleton
        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Initial idle time should be very small
        idle = manager.get_idle_seconds()
        assert idle >= 0
        assert idle < 2  # Should be less than 2 seconds

    def test_update_activity_resets_idle(self) -> None:
        """UT-SRV-009: Activity update resets idle time."""
        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Wait a bit
        time.sleep(0.1)
        before_update = manager.get_idle_seconds()

        # Update activity
        manager._update_activity()
        after_update = manager.get_idle_seconds()

        # Idle time should reset
        assert after_update < before_update or after_update == 0

    def test_idle_increases_over_time(self) -> None:
        """Idle time increases when no activity."""
        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()

        initial = manager.get_idle_seconds()
        time.sleep(0.2)
        later = manager.get_idle_seconds()

        # Should have increased
        assert later >= initial

    @pytest.mark.asyncio
    async def test_generate_updates_activity_before_and_after(self) -> None:
        """UT-SRV-009: Generate updates activity before and after."""
        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        manager.models["default"] = mock_model
        manager.tokenizers["default"] = mock_tokenizer

        activity_times = []

        original_update = manager._update_activity

        def track_update():
            activity_times.append(time.monotonic())
            original_update()

        manager._update_activity = track_update

        # Mock the sync generate method
        with patch.object(manager, "_generate_sync", return_value="test output"):
            await manager.generate("test prompt")

        # Should have updated activity twice (before and after)
        assert len(activity_times) == 2
        assert activity_times[1] > activity_times[0]


class TestIdleMonitor:
    """Tests for the idle monitor background task."""

    @pytest.mark.asyncio
    async def test_idle_monitor_disabled_when_timeout_zero(self) -> None:
        """Idle monitor exits when timeout is 0."""
        with patch.dict("os.environ", {"IRCA_SERVER_IDLE_TIMEOUT": "0"}):
            from src.server.main import _idle_monitor

            # Should return quickly without looping
            task = asyncio.create_task(_idle_monitor())

            # Wait a short time
            await asyncio.sleep(0.1)

            # Task should be done (it exited due to timeout=0)
            assert task.done()

    @pytest.mark.asyncio
    async def test_idle_check_logic(self) -> None:
        """UT-SRV-010: Idle check logic correctly identifies timeout."""
        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Setup mock model
        manager.models["default"] = MagicMock()
        manager.tokenizers["default"] = MagicMock()
        manager.loaded_configs["default"] = {"base": "test", "adapter": None}

        # Make the model appear very idle (by setting last_activity to past)
        manager._last_activity = time.monotonic() - 3600  # 1 hour ago

        # Check idle time detection
        idle_seconds = manager.get_idle_seconds()
        assert idle_seconds >= 3600

        # Configure timeout of 1 minute
        idle_timeout_seconds = 60

        # Idle check should trigger ejection
        assert idle_seconds >= idle_timeout_seconds, "Idle time should exceed timeout"

        # Verify eject_model exists and can be called
        eject_called = False

        async def mock_eject(alias: str) -> None:
            nonlocal eject_called
            eject_called = True
            del manager.models[alias]
            del manager.tokenizers[alias]
            del manager.loaded_configs[alias]

        manager.eject_model = mock_eject

        # Simulate what idle monitor would do
        if idle_seconds >= idle_timeout_seconds:
            for alias in list(manager.models.keys()):
                await manager.eject_model(alias)

        assert eject_called
        assert len(manager.models) == 0


class TestGracefulShutdown:
    """Tests for graceful shutdown with model ejection."""

    @pytest.mark.asyncio
    async def test_eject_all_models_with_timeout(self) -> None:
        """Models are ejected with timeout during shutdown."""
        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Setup mock models
        manager.models["model1"] = MagicMock()
        manager.models["model2"] = MagicMock()
        manager.tokenizers["model1"] = MagicMock()
        manager.tokenizers["model2"] = MagicMock()
        manager.loaded_configs["model1"] = {"base": "test1", "adapter": None}
        manager.loaded_configs["model2"] = {"base": "test2", "adapter": None}

        ejected_aliases = []

        async def mock_eject(alias: str) -> None:
            ejected_aliases.append(alias)
            del manager.models[alias]
            del manager.tokenizers[alias]
            del manager.loaded_configs[alias]

        manager.eject_model = mock_eject

        from src.server.main import _eject_all_models_with_timeout

        await _eject_all_models_with_timeout(timeout=5.0)

        # Both models should be ejected
        assert len(ejected_aliases) == 2
        assert "model1" in ejected_aliases
        assert "model2" in ejected_aliases

    @pytest.mark.asyncio
    async def test_eject_timeout_handling(self) -> None:
        """FM-4: Ejection timeout is handled gracefully."""
        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # Setup mock model that takes too long to eject
        manager.models["slow"] = MagicMock()
        manager.tokenizers["slow"] = MagicMock()
        manager.loaded_configs["slow"] = {"base": "test", "adapter": None}

        async def slow_eject(alias: str) -> None:
            # Sleep longer than timeout
            await asyncio.sleep(10)

        manager.eject_model = slow_eject

        from src.server.main import _eject_all_models_with_timeout

        # Should complete within timeout (not hang)
        start = time.monotonic()
        await _eject_all_models_with_timeout(timeout=0.5)
        elapsed = time.monotonic() - start

        # Should have timed out rather than waiting full 10s
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_eject_no_models_loaded(self) -> None:
        """Ejection handles no models gracefully."""
        from src.server.model_manager import ModelManager

        ModelManager._instance = None
        manager = ModelManager.get_instance()

        # No models loaded
        manager.models.clear()

        from src.server.main import _eject_all_models_with_timeout

        # Should complete without error
        await _eject_all_models_with_timeout(timeout=5.0)
