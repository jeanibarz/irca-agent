"""
Tests for Playground CLI Commands

Tests for the playground start/stop/status commands that manage frontend and backend together.
"""

import os
import signal
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.commands.playground import (
    _get_frontend_pid_file,
    _get_ui_dir,
    _is_frontend_running,
    _read_frontend_pid,
    _remove_frontend_pid,
    _write_frontend_pid,
    playground,
)


class TestFrontendPIDManagement:
    """Tests for frontend PID file management."""

    def test_write_and_read_frontend_pid(self, tmp_path: Path) -> None:
        """Frontend PID file can be written and read."""
        pid_file = tmp_path / "frontend.pid"
        with patch.dict(os.environ, {"IRCA_FRONTEND_PID_FILE": str(pid_file)}):
            _write_frontend_pid(54321)
            assert _read_frontend_pid() == 54321

    def test_read_frontend_pid_nonexistent(self, tmp_path: Path) -> None:
        """Reading nonexistent frontend PID file returns None."""
        pid_file = tmp_path / "nonexistent.pid"
        with patch.dict(os.environ, {"IRCA_FRONTEND_PID_FILE": str(pid_file)}):
            assert _read_frontend_pid() is None

    def test_remove_frontend_pid(self, tmp_path: Path) -> None:
        """Frontend PID file can be removed."""
        pid_file = tmp_path / "frontend.pid"
        pid_file.write_text("54321")
        with patch.dict(os.environ, {"IRCA_FRONTEND_PID_FILE": str(pid_file)}):
            _remove_frontend_pid()
            assert not pid_file.exists()

    def test_is_frontend_running_stale_pid(self, tmp_path: Path) -> None:
        """Stale frontend PID is detected and cleaned up."""
        pid_file = tmp_path / "frontend.pid"
        pid_file.write_text("999999999")  # Non-existent PID

        with patch.dict(os.environ, {"IRCA_FRONTEND_PID_FILE": str(pid_file)}):
            is_running, pid = _is_frontend_running()
            assert not is_running
            assert pid is None
            assert not pid_file.exists()


class TestUIDirectoryDetection:
    """Tests for UI directory detection."""

    def test_get_ui_dir_from_cwd(self, tmp_path: Path) -> None:
        """UI directory found relative to current working directory."""
        ui_dir = tmp_path / "ui"
        ui_dir.mkdir()
        (ui_dir / "package.json").write_text("{}")

        with patch("pathlib.Path.cwd", return_value=tmp_path):
            result = _get_ui_dir()
            assert result == ui_dir


class TestPlaygroundStartCommand:
    """Tests for the 'playground start' command."""

    def test_start_checks_npm(self, tmp_path: Path) -> None:
        """Start checks for npm availability."""
        runner = CliRunner()

        with patch("src.cli.commands.playground._check_npm", return_value=False):
            result = runner.invoke(playground, ["start"])

            assert result.exit_code == 1
            assert "npm is not installed" in result.output

    def test_start_backend_only(self, tmp_path: Path) -> None:
        """Start --backend-only skips frontend."""
        import subprocess

        pid_file = tmp_path / "server.pid"
        frontend_pid_file = tmp_path / "frontend.pid"

        runner = CliRunner()

        with patch.dict(os.environ, {
            "IRCA_PID_FILE": str(pid_file),
            "IRCA_FRONTEND_PID_FILE": str(frontend_pid_file),
        }):
            with patch.object(subprocess, "Popen") as mock_popen:
                mock_process = MagicMock()
                mock_process.pid = 12345
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                # Mock the health check to return immediately
                with patch("src.cli.commands.playground._wait_for_backend_health", return_value=True):
                    result = runner.invoke(playground, ["start", "--backend-only", "--no-open"])

                    # Backend should be started
                    assert mock_popen.called
                    assert "Backend started" in result.output
                    # Frontend should not be mentioned as started
                    assert "Frontend started" not in result.output

    def test_start_frontend_only_checks_ui_dir(self, tmp_path: Path) -> None:
        """Start --frontend-only checks UI directory exists."""
        frontend_pid_file = tmp_path / "frontend.pid"

        runner = CliRunner()

        with patch.dict(os.environ, {"IRCA_FRONTEND_PID_FILE": str(frontend_pid_file)}):
            with patch("src.cli.commands.playground._check_npm", return_value=True):
                with patch("src.cli.commands.playground._get_ui_dir", return_value=tmp_path / "nonexistent"):
                    result = runner.invoke(playground, ["start", "--frontend-only"])

                    assert result.exit_code == 1
                    assert "UI directory not found" in result.output

    def test_start_rejects_conflicting_flags(self) -> None:
        """Start rejects --backend-only and --frontend-only together."""
        runner = CliRunner()
        result = runner.invoke(playground, ["start", "--backend-only", "--frontend-only"])

        assert result.exit_code == 1
        assert "Cannot use both" in result.output


class TestPlaygroundStopCommand:
    """Tests for the 'playground stop' command."""

    def test_stop_nothing_running(self, tmp_path: Path) -> None:
        """Stop when nothing running shows appropriate message."""
        pid_file = tmp_path / "server.pid"
        frontend_pid_file = tmp_path / "frontend.pid"

        runner = CliRunner()
        with patch.dict(os.environ, {
            "IRCA_PID_FILE": str(pid_file),
            "IRCA_FRONTEND_PID_FILE": str(frontend_pid_file),
        }):
            result = runner.invoke(playground, ["stop"])

            assert "Nothing was running" in result.output

    def test_stop_backend_only(self, tmp_path: Path) -> None:
        """Stop --backend-only only stops backend."""
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(str(os.getpid()))  # Use current PID
        frontend_pid_file = tmp_path / "frontend.pid"
        frontend_pid_file.write_text(str(os.getpid()))

        runner = CliRunner()
        with patch.dict(os.environ, {
            "IRCA_PID_FILE": str(pid_file),
            "IRCA_FRONTEND_PID_FILE": str(frontend_pid_file),
        }):
            with patch("os.kill") as mock_kill:
                with patch("src.cli.commands.playground._is_process_running", return_value=False):
                    result = runner.invoke(playground, ["stop", "--backend-only"])

                    # Should only try to stop backend
                    assert "Stopping backend" in result.output
                    assert "Stopping frontend" not in result.output

    def test_stop_force_uses_sigkill(self, tmp_path: Path) -> None:
        """Stop --force sends SIGKILL."""
        pid_file = tmp_path / "server.pid"
        pid_file.write_text("12345")
        frontend_pid_file = tmp_path / "frontend.pid"

        runner = CliRunner()
        with patch.dict(os.environ, {
            "IRCA_PID_FILE": str(pid_file),
            "IRCA_FRONTEND_PID_FILE": str(frontend_pid_file),
        }):
            with patch("src.cli.commands.server._is_process_running") as mock_running:
                mock_running.side_effect = [True, False]  # Running, then stopped
                with patch("os.kill") as mock_kill:
                    result = runner.invoke(playground, ["stop", "--force"])

                    # Should use SIGKILL
                    mock_kill.assert_called_with(12345, signal.SIGKILL)


class TestPlaygroundStatusCommand:
    """Tests for the 'playground status' command."""

    def test_status_nothing_running(self, tmp_path: Path) -> None:
        """Status when nothing running."""
        pid_file = tmp_path / "server.pid"
        frontend_pid_file = tmp_path / "frontend.pid"

        runner = CliRunner()
        with patch.dict(os.environ, {
            "IRCA_PID_FILE": str(pid_file),
            "IRCA_FRONTEND_PID_FILE": str(frontend_pid_file),
        }):
            result = runner.invoke(playground, ["status"])

            assert "Backend:  " in result.output
            assert "Stopped" in result.output
            assert "Frontend: " in result.output
            assert "not running" in result.output.lower()

    def test_status_json_output(self, tmp_path: Path) -> None:
        """Status --json outputs valid JSON."""
        import json

        pid_file = tmp_path / "server.pid"
        frontend_pid_file = tmp_path / "frontend.pid"

        runner = CliRunner()
        with patch.dict(os.environ, {
            "IRCA_PID_FILE": str(pid_file),
            "IRCA_FRONTEND_PID_FILE": str(frontend_pid_file),
        }):
            result = runner.invoke(playground, ["status", "--json"])

            data = json.loads(result.output)
            assert "backend" in data
            assert "frontend" in data
            assert data["backend"]["running"] is False
            assert data["frontend"]["running"] is False

    def test_status_both_running(self, tmp_path: Path) -> None:
        """Status when both services running."""
        pid_file = tmp_path / "server.pid"
        pid_file.write_text(str(os.getpid()))
        frontend_pid_file = tmp_path / "frontend.pid"
        frontend_pid_file.write_text(str(os.getpid()))

        runner = CliRunner()
        with patch.dict(os.environ, {
            "IRCA_PID_FILE": str(pid_file),
            "IRCA_FRONTEND_PID_FILE": str(frontend_pid_file),
        }):
            # Mock the health check
            with patch("urllib.request.urlopen") as mock_urlopen:
                mock_response = MagicMock()
                mock_response.read.return_value = b'{"status": "healthy"}'
                mock_response.__enter__ = MagicMock(return_value=mock_response)
                mock_response.__exit__ = MagicMock(return_value=False)
                mock_urlopen.return_value = mock_response

                result = runner.invoke(playground, ["status"])

                assert "Running" in result.output
                assert "fully operational" in result.output


class TestPlaygroundRestartCommand:
    """Tests for the 'playground restart' command."""

    def test_restart_invokes_stop_and_start(self, tmp_path: Path) -> None:
        """Restart invokes stop then start."""
        pid_file = tmp_path / "server.pid"
        frontend_pid_file = tmp_path / "frontend.pid"

        runner = CliRunner()
        with patch.dict(os.environ, {
            "IRCA_PID_FILE": str(pid_file),
            "IRCA_FRONTEND_PID_FILE": str(frontend_pid_file),
        }):
            with patch("src.cli.commands.playground._check_npm", return_value=False):
                # This will fail at npm check, but we can verify stop was called first
                result = runner.invoke(playground, ["restart"])

                # Stop should have been invoked (nothing to stop)
                assert "Stopping" in result.output or "Nothing was running" in result.output
