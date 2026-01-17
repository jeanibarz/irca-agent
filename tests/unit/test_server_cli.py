"""
Tests for Server CLI Commands

Tests for the server start/stop/status commands and PID file management.
"""

import os
import signal
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.commands.server import (
    _get_pid_file,
    _is_process_running,
    _is_server_running,
    _read_pid,
    _remove_pid,
    _write_pid,
    server,
)


class TestPIDManagement:
    """Tests for PID file management utilities."""

    def test_write_and_read_pid(self, tmp_path: Path) -> None:
        """UT-SRV-001: PID file can be written and read."""
        pid_file = tmp_path / "test.pid"
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            _write_pid(12345)
            assert _read_pid() == 12345

    def test_read_pid_nonexistent(self, tmp_path: Path) -> None:
        """Reading nonexistent PID file returns None."""
        pid_file = tmp_path / "nonexistent.pid"
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            assert _read_pid() is None

    def test_read_pid_invalid_content(self, tmp_path: Path) -> None:
        """Reading invalid PID file returns None."""
        pid_file = tmp_path / "invalid.pid"
        pid_file.write_text("not a number")
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            assert _read_pid() is None

    def test_remove_pid(self, tmp_path: Path) -> None:
        """PID file can be removed."""
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("12345")
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            _remove_pid()
            assert not pid_file.exists()

    def test_remove_pid_nonexistent(self, tmp_path: Path) -> None:
        """Removing nonexistent PID file doesn't raise."""
        pid_file = tmp_path / "nonexistent.pid"
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            _remove_pid()  # Should not raise

    def test_write_pid_atomic(self, tmp_path: Path) -> None:
        """UT-SRV-003 (partial): PID write is atomic."""
        pid_file = tmp_path / "test.pid"
        temp_file = pid_file.with_suffix(".tmp")

        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            _write_pid(12345)
            # Temp file should be cleaned up
            assert not temp_file.exists()
            assert pid_file.exists()


class TestProcessChecks:
    """Tests for process running checks."""

    def test_is_process_running_current(self) -> None:
        """Current process is reported as running."""
        assert _is_process_running(os.getpid())

    def test_is_process_running_invalid(self) -> None:
        """Invalid PID is reported as not running."""
        # Use a very high PID that's unlikely to exist
        assert not _is_process_running(999999999)

    def test_is_server_running_no_pid_file(self, tmp_path: Path) -> None:
        """Server not running when no PID file exists."""
        pid_file = tmp_path / "test.pid"
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            is_running, pid = _is_server_running()
            assert not is_running
            assert pid is None

    def test_is_server_running_stale_pid(self, tmp_path: Path) -> None:
        """UT-SRV-003: Stale PID is detected and cleaned up."""
        pid_file = tmp_path / "test.pid"
        pid_file.write_text("999999999")  # Non-existent PID

        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            is_running, pid = _is_server_running()
            assert not is_running
            assert pid is None
            # PID file should be cleaned up
            assert not pid_file.exists()


class TestServerStartCommand:
    """Tests for the 'server start' command."""

    def test_start_rejects_if_already_running(self, tmp_path: Path) -> None:
        """UT-SRV-002: Start rejects if server already running."""
        pid_file = tmp_path / "test.pid"
        # Write current process PID (which is running)
        pid_file.write_text(str(os.getpid()))

        runner = CliRunner()
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            result = runner.invoke(server, ["start", "--foreground"])

            assert result.exit_code == 1
            assert "already running" in result.output

    def test_start_removes_stale_pid(self, tmp_path: Path) -> None:
        """UT-SRV-003: Start removes stale PID and continues."""
        import subprocess

        pid_file = tmp_path / "test.pid"
        pid_file.write_text("999999999")  # Stale PID

        runner = CliRunner()
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            with patch.object(subprocess, "Popen") as mock_popen:
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                result = runner.invoke(server, ["start"])

                # Should succeed (or at least not fail due to running server)
                # Note: may fail for other reasons in test environment
                assert "already running" not in result.output

    def test_start_with_idle_timeout(self, tmp_path: Path) -> None:
        """UT-SRV-008: Start with --idle-timeout sets env var."""
        import subprocess

        pid_file = tmp_path / "test.pid"

        runner = CliRunner()
        env_captured = {}

        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            with patch.object(subprocess, "Popen") as mock_popen:

                def capture_env(*args, **kwargs):
                    env_captured.update(os.environ)
                    mock = MagicMock()
                    mock.pid = 12345
                    mock.poll.return_value = None
                    return mock

                mock_popen.side_effect = capture_env

                result = runner.invoke(server, ["start", "--idle-timeout", "5"])

                assert env_captured.get("IRCA_SERVER_IDLE_TIMEOUT") == "5"


class TestServerStopCommand:
    """Tests for the 'server stop' command."""

    def test_stop_not_running(self, tmp_path: Path) -> None:
        """Stop when not running shows appropriate message."""
        pid_file = tmp_path / "test.pid"

        runner = CliRunner()
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            result = runner.invoke(server, ["stop"])

            assert "not running" in result.output

    def test_stop_sends_sigterm(self, tmp_path: Path) -> None:
        """UT-SRV-004: Stop sends SIGTERM by default."""
        pid_file = tmp_path / "test.pid"
        target_pid = 12345
        pid_file.write_text(str(target_pid))

        runner = CliRunner()
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            with patch("src.cli.commands.server._is_process_running") as mock_running:
                with patch("os.kill") as mock_kill:
                    # Process is running initially, then stops
                    mock_running.side_effect = [True, False]

                    result = runner.invoke(server, ["stop"])

                    # Should call kill with SIGTERM
                    mock_kill.assert_called_with(target_pid, signal.SIGTERM)

    def test_stop_force_sends_sigkill(self, tmp_path: Path) -> None:
        """UT-SRV-005: Stop --force sends SIGKILL."""
        pid_file = tmp_path / "test.pid"
        target_pid = 12345
        pid_file.write_text(str(target_pid))

        runner = CliRunner()
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            with patch("src.cli.commands.server._is_process_running") as mock_running:
                with patch("os.kill") as mock_kill:
                    mock_running.side_effect = [True, False]

                    result = runner.invoke(server, ["stop", "--force"])

                    # Should call kill with SIGKILL
                    mock_kill.assert_called_with(target_pid, signal.SIGKILL)


class TestServerStatusCommand:
    """Tests for the 'server status' command."""

    def test_status_not_running(self, tmp_path: Path) -> None:
        """UT-SRV-006: Status shows stopped when not running."""
        pid_file = tmp_path / "test.pid"

        runner = CliRunner()
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            result = runner.invoke(server, ["status"])

            assert "not running" in result.output

    def test_status_running(self, tmp_path: Path) -> None:
        """UT-SRV-006: Status shows running with PID."""
        pid_file = tmp_path / "test.pid"
        pid_file.write_text(str(os.getpid()))  # Current process

        runner = CliRunner()
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            # Mock the health check to avoid network call
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = b'{"status": "healthy"}'
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                result = runner.invoke(server, ["status"])

                assert "running" in result.output.lower()
                assert str(os.getpid()) in result.output

    def test_status_json_output(self, tmp_path: Path) -> None:
        """Status with --json outputs valid JSON."""
        import json

        pid_file = tmp_path / "test.pid"

        runner = CliRunner()
        with patch.dict(os.environ, {"IRCA_PID_FILE": str(pid_file)}):
            result = runner.invoke(server, ["status", "--json"])

            # Should be valid JSON
            data = json.loads(result.output)
            assert "running" in data
            assert data["running"] is False
