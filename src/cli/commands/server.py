"""
Server Command

Commands for managing the IRCA backend server lifecycle.
"""

import logging
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import click

logger = logging.getLogger(__name__)

# PID file location (in temp directory for system-level access)
DEFAULT_PID_FILE = Path(tempfile.gettempdir()) / "irca-server.pid"


def _get_pid_file() -> Path:
    """Get the PID file path from env or default."""
    return Path(os.environ.get("IRCA_PID_FILE", str(DEFAULT_PID_FILE)))


def _read_pid() -> int | None:
    """Read PID from file, returning None if not found or invalid."""
    pid_file = _get_pid_file()
    if not pid_file.exists():
        return None

    try:
        pid = int(pid_file.read_text().strip())
        return pid
    except (ValueError, OSError):
        return None


def _write_pid(pid: int) -> None:
    """Write PID to file atomically (FM-10 mitigation)."""
    pid_file = _get_pid_file()
    temp_file = pid_file.with_suffix(".tmp")

    try:
        temp_file.write_text(str(pid))
        temp_file.rename(pid_file)
    except OSError as e:
        # Clean up temp file if rename failed
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to write PID file: {e}") from e


def _remove_pid() -> None:
    """Remove PID file if it exists."""
    pid_file = _get_pid_file()
    if pid_file.exists():
        try:
            pid_file.unlink()
        except OSError:
            pass  # Best effort


def _is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running (FM-1 mitigation)."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _is_server_running() -> tuple[bool, int | None]:
    """
    Check if server is running.

    Returns (is_running, pid).
    Handles stale PID files (FM-1 mitigation).
    """
    pid = _read_pid()
    if pid is None:
        return False, None

    if _is_process_running(pid):
        return True, pid
    else:
        # Stale PID file - clean it up
        _remove_pid()
        return False, None


@click.group()
def server() -> None:
    """Manage the IRCA backend server."""
    pass


@server.command("start")
@click.option(
    "--host",
    "-h",
    type=str,
    default=None,
    help="Host to bind to (default: from settings or 0.0.0.0)",
)
@click.option(
    "--port",
    "-p",
    type=int,
    default=None,
    help="Port to bind to (default: from settings or 8000)",
)
@click.option(
    "--reload",
    is_flag=True,
    default=False,
    help="Enable auto-reload mode for development",
)
@click.option(
    "--idle-timeout",
    type=int,
    default=None,
    help="Auto-eject model after N minutes of inactivity (0=disabled, default: from settings or 30)",
)
@click.option(
    "--foreground",
    "-f",
    is_flag=True,
    default=False,
    help="Run in foreground (default: background)",
)
@click.pass_context
def start(
    ctx: click.Context,
    host: str | None,
    port: int | None,
    reload: bool,
    idle_timeout: int | None,
    foreground: bool,
) -> None:
    """
    Start the IRCA backend server.

    By default, starts the server in the background and returns.
    Use --foreground to run in the current process.

    Examples:

        # Start server in background
        irca server start

        # Start with auto-reload for development
        irca server start --reload --foreground

        # Start with custom host/port
        irca server start --host 127.0.0.1 --port 9000

        # Start with auto-idle ejection after 5 minutes
        irca server start --idle-timeout 5
    """
    from src.config import get_settings

    settings = get_settings()

    # Resolve settings with CLI overrides
    resolved_host = host or getattr(settings, "server_host", "0.0.0.0")
    resolved_port = port or getattr(settings, "server_port", 8000)
    resolved_idle_timeout = idle_timeout if idle_timeout is not None else getattr(settings, "server_idle_timeout", 30)

    # Check if already running
    is_running, existing_pid = _is_server_running()
    if is_running:
        click.secho(f"Server is already running (PID: {existing_pid})", fg="yellow")
        click.echo(f"Use 'irca server stop' to stop it first.")
        sys.exit(1)

    # Set environment variables for the server process
    os.environ["IRCA_SERVER_HOST"] = resolved_host
    os.environ["IRCA_SERVER_PORT"] = str(resolved_port)
    os.environ["IRCA_SERVER_IDLE_TIMEOUT"] = str(resolved_idle_timeout)

    click.echo(f"🚀 Starting IRCA server...")
    click.echo(f"   Host: {resolved_host}")
    click.echo(f"   Port: {resolved_port}")
    click.echo(f"   Auto-reload: {'enabled' if reload else 'disabled'}")
    click.echo(f"   Idle timeout: {resolved_idle_timeout} minutes" if resolved_idle_timeout > 0 else "   Idle timeout: disabled")

    if foreground or reload:
        # Run in foreground (reload requires foreground)
        if reload and not foreground:
            click.echo("   (--reload implies --foreground)")

        # Write PID for current process
        _write_pid(os.getpid())

        try:
            import uvicorn

            uvicorn.run(
                "src.server.main:app",
                host=resolved_host,
                port=resolved_port,
                reload=reload,
            )
        finally:
            _remove_pid()
    else:
        # Start in background using subprocess
        import subprocess

        # Build command
        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "src.server.main:app",
            "--host",
            resolved_host,
            "--port",
            str(resolved_port),
        ]

        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Write PID
        _write_pid(process.pid)

        # Wait briefly to check if process started
        time.sleep(0.5)
        if process.poll() is not None:
            # Process exited immediately
            _remove_pid()
            click.secho("Error: Server failed to start", fg="red")
            sys.exit(1)

        click.secho(f"✅ Server started (PID: {process.pid})", fg="green")
        click.echo(f"   URL: http://{resolved_host}:{resolved_port}")
        click.echo(f"   Health: http://{resolved_host}:{resolved_port}/health")
        click.echo(f"\nUse 'irca server status' to check status")
        click.echo(f"Use 'irca server stop' to stop the server")


@server.command("stop")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force stop with SIGKILL (skip graceful shutdown)",
)
@click.option(
    "--timeout",
    type=int,
    default=10,
    help="Timeout in seconds for graceful shutdown (default: 10)",
)
@click.pass_context
def stop(ctx: click.Context, force: bool, timeout: int) -> None:
    """
    Stop the IRCA backend server.

    Sends SIGTERM for graceful shutdown, allowing the server to:
    - Eject any loaded GPU models
    - Release CUDA memory
    - Clean up resources

    Use --force to send SIGKILL immediately (skips model ejection).

    Examples:

        # Graceful stop
        irca server stop

        # Force stop (immediate)
        irca server stop --force

        # Graceful stop with longer timeout
        irca server stop --timeout 30
    """
    is_running, pid = _is_server_running()

    if not is_running:
        click.secho("Server is not running", fg="yellow")
        _remove_pid()  # Clean up stale PID file if any
        return

    click.echo(f"🛑 Stopping IRCA server (PID: {pid})...")

    try:
        if force:
            click.echo("   Sending SIGKILL (force stop)...")
            os.kill(pid, signal.SIGKILL)
        else:
            click.echo("   Sending SIGTERM (graceful shutdown)...")
            os.kill(pid, signal.SIGTERM)

            # Wait for process to terminate
            start_time = time.time()
            while time.time() - start_time < timeout:
                if not _is_process_running(pid):
                    break
                time.sleep(0.1)
            else:
                # Timeout reached, force kill
                click.secho(f"   Timeout after {timeout}s, sending SIGKILL...", fg="yellow")
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)

        # Verify stopped
        if _is_process_running(pid):
            click.secho("Error: Failed to stop server", fg="red")
            sys.exit(1)

        _remove_pid()
        click.secho("✅ Server stopped", fg="green")

    except ProcessLookupError:
        # Process already gone
        _remove_pid()
        click.secho("✅ Server stopped", fg="green")
    except PermissionError:
        click.secho(f"Error: Permission denied to stop process {pid}", fg="red")
        sys.exit(1)


@server.command("status")
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output status as JSON",
)
@click.pass_context
def status(ctx: click.Context, output_json: bool) -> None:
    """
    Check IRCA server status.

    Shows whether the server is running, and if so, provides details
    about loaded models and idle time.

    Examples:

        # Check status
        irca server status

        # JSON output for scripting
        irca server status --json
    """
    import json

    is_running, pid = _is_server_running()

    status_info = {
        "running": is_running,
        "pid": pid,
    }

    if is_running:
        # Try to get health info from server
        try:
            import urllib.request
            import urllib.error

            settings_host = os.environ.get("IRCA_SERVER_HOST", "0.0.0.0")
            settings_port = os.environ.get("IRCA_SERVER_PORT", "8000")

            # Use localhost for health check since 0.0.0.0 binds to all interfaces
            check_host = "127.0.0.1" if settings_host == "0.0.0.0" else settings_host
            url = f"http://{check_host}:{settings_port}/health"

            with urllib.request.urlopen(url, timeout=5) as response:
                health_data = json.loads(response.read().decode())
                status_info.update(health_data)

        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            status_info["health_check"] = "failed"

    if output_json:
        click.echo(json.dumps(status_info, indent=2))
    else:
        if is_running:
            click.secho(f"✅ Server is running", fg="green")
            click.echo(f"   PID: {pid}")

            if "status" in status_info:
                click.echo(f"   Health: {status_info.get('status', 'unknown')}")

            if "model_loaded" in status_info:
                if status_info["model_loaded"]:
                    click.echo(f"   Model loaded: {status_info.get('current_model', 'unknown')}")
                    if "idle_seconds" in status_info:
                        idle_mins = status_info["idle_seconds"] // 60
                        idle_secs = status_info["idle_seconds"] % 60
                        click.echo(f"   Idle time: {idle_mins}m {idle_secs}s")
                else:
                    click.echo("   Model loaded: none")

        else:
            click.secho("⚫ Server is not running", fg="yellow")


@server.command("restart")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force restart (SIGKILL)",
)
@click.pass_context
def restart(ctx: click.Context, force: bool) -> None:
    """
    Restart the IRCA backend server.

    Stops the server (if running) and starts it again.

    Examples:

        # Graceful restart
        irca server restart

        # Force restart
        irca server restart --force
    """
    # Invoke stop
    is_running, _ = _is_server_running()
    if is_running:
        ctx.invoke(stop, force=force)
        time.sleep(1)  # Brief pause between stop and start

    # Invoke start
    ctx.invoke(start)
