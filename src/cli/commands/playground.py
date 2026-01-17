"""
Playground Command

Commands for managing the IRCA playground (frontend + backend together).
"""

import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import webbrowser
from pathlib import Path

import click

from .server import (
    _is_server_running,
    _read_pid,
    _remove_pid,
    _write_pid,
    _is_process_running,
)

logger = logging.getLogger(__name__)

# PID file for frontend
DEFAULT_FRONTEND_PID_FILE = Path(tempfile.gettempdir()) / "irca-frontend.pid"


def _get_frontend_pid_file() -> Path:
    """Get the frontend PID file path."""
    return Path(os.environ.get("IRCA_FRONTEND_PID_FILE", str(DEFAULT_FRONTEND_PID_FILE)))


def _read_frontend_pid() -> int | None:
    """Read frontend PID from file."""
    pid_file = _get_frontend_pid_file()
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except (ValueError, OSError):
        return None


def _write_frontend_pid(pid: int) -> None:
    """Write frontend PID to file atomically."""
    pid_file = _get_frontend_pid_file()
    temp_file = pid_file.with_suffix(".tmp")
    try:
        temp_file.write_text(str(pid))
        temp_file.rename(pid_file)
    except OSError as e:
        if temp_file.exists():
            temp_file.unlink()
        raise RuntimeError(f"Failed to write frontend PID file: {e}") from e


def _remove_frontend_pid() -> None:
    """Remove frontend PID file."""
    pid_file = _get_frontend_pid_file()
    if pid_file.exists():
        try:
            pid_file.unlink()
        except OSError:
            pass


def _is_frontend_running() -> tuple[bool, int | None]:
    """Check if frontend is running."""
    pid = _read_frontend_pid()
    if pid is None:
        return False, None
    if _is_process_running(pid):
        return True, pid
    else:
        _remove_frontend_pid()
        return False, None


def _get_ui_dir() -> Path:
    """Get the UI directory path."""
    # Try relative to current working directory first
    cwd_ui = Path.cwd() / "ui"
    if cwd_ui.exists() and (cwd_ui / "package.json").exists():
        return cwd_ui

    # Try relative to the module location
    module_dir = Path(__file__).parent.parent.parent.parent
    module_ui = module_dir / "ui"
    if module_ui.exists() and (module_ui / "package.json").exists():
        return module_ui

    return cwd_ui  # Return default even if not found (will error later)


def _check_npm() -> bool:
    """Check if npm is available."""
    return shutil.which("npm") is not None


def _check_node() -> bool:
    """Check if node is available."""
    return shutil.which("node") is not None


def _wait_for_backend_health(port: int, timeout: int = 10) -> bool:
    """
    Wait for backend to be healthy (FM-7 mitigation).

    Args:
        port: Backend port to check
        timeout: Maximum seconds to wait

    Returns:
        True if backend became healthy, False if timeout
    """
    import urllib.request
    import urllib.error

    url = f"http://127.0.0.1:{port}/health"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)

    return False


@click.group()
def playground() -> None:
    """Manage the IRCA playground (frontend + backend)."""
    pass


@playground.command("start")
@click.option(
    "--dev/--prod",
    default=True,
    help="Run frontend in dev mode (hot reload) or serve production build",
)
@click.option(
    "--backend-only",
    is_flag=True,
    default=False,
    help="Only start the backend server",
)
@click.option(
    "--frontend-only",
    is_flag=True,
    default=False,
    help="Only start the frontend server",
)
@click.option(
    "--open/--no-open",
    default=True,
    help="Open browser automatically (default: yes)",
)
@click.option(
    "--backend-port",
    type=int,
    default=8000,
    help="Backend server port (default: 8000)",
)
@click.option(
    "--frontend-port",
    type=int,
    default=3000,
    help="Frontend server port (default: 3000)",
)
@click.option(
    "--idle-timeout",
    type=int,
    default=30,
    help="Auto-eject model after N minutes of inactivity (default: 30)",
)
@click.pass_context
def start(
    ctx: click.Context,
    dev: bool,
    backend_only: bool,
    frontend_only: bool,
    open: bool,
    backend_port: int,
    frontend_port: int,
    idle_timeout: int,
) -> None:
    """
    Start the IRCA playground.

    Starts both the backend API server and the frontend UI.
    By default, opens the browser to the frontend URL.

    Examples:

        # Start everything (default)
        irca playground start

        # Start in production mode (serve built frontend)
        irca playground start --prod

        # Start without opening browser
        irca playground start --no-open

        # Start only backend
        irca playground start --backend-only

        # Custom ports
        irca playground start --backend-port 9000 --frontend-port 4000
    """
    if backend_only and frontend_only:
        click.secho("Error: Cannot use both --backend-only and --frontend-only", fg="red")
        sys.exit(1)

    start_backend = not frontend_only
    start_frontend = not backend_only

    # Check prerequisites
    if start_frontend:
        if not _check_npm():
            click.secho("Error: npm is not installed. Please install Node.js first.", fg="red")
            sys.exit(1)

        ui_dir = _get_ui_dir()
        if not ui_dir.exists():
            click.secho(f"Error: UI directory not found at {ui_dir}", fg="red")
            sys.exit(1)

        if not (ui_dir / "node_modules").exists():
            click.secho("Installing frontend dependencies...", fg="yellow")
            result = subprocess.run(
                ["npm", "install"],
                cwd=ui_dir,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                click.secho(f"Error installing dependencies: {result.stderr}", fg="red")
                sys.exit(1)
            click.secho("Dependencies installed.", fg="green")

    click.echo("🎮 Starting IRCA Playground...")

    # Start backend
    if start_backend:
        backend_running, backend_pid = _is_server_running()
        if backend_running:
            click.echo(f"   Backend already running (PID: {backend_pid})")
        else:
            click.echo(f"   Starting backend on port {backend_port}...")

            # Set environment variables
            os.environ["IRCA_SERVER_HOST"] = "0.0.0.0"
            os.environ["IRCA_SERVER_PORT"] = str(backend_port)
            os.environ["IRCA_SERVER_IDLE_TIMEOUT"] = str(idle_timeout)

            # Start backend
            cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "src.server.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(backend_port),
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Import here to avoid circular dependency
            from .server import _write_pid as write_server_pid
            write_server_pid(process.pid)

            # FM-7 mitigation: Wait for health check instead of just process existence
            click.echo("   Waiting for backend to be healthy...")
            if not _wait_for_backend_health(backend_port, timeout=15):
                # Check if process died
                if process.poll() is not None:
                    from .server import _remove_pid as remove_server_pid
                    remove_server_pid()
                    click.secho("   Error: Backend process exited", fg="red")
                    sys.exit(1)
                else:
                    click.secho("   Warning: Backend started but health check failed", fg="yellow")
            else:
                click.secho(f"   Backend started and healthy (PID: {process.pid})", fg="green")

    # Start frontend
    if start_frontend:
        frontend_running, frontend_pid = _is_frontend_running()
        if frontend_running:
            click.echo(f"   Frontend already running (PID: {frontend_pid})")
        else:
            ui_dir = _get_ui_dir()
            click.echo(f"   Starting frontend on port {frontend_port}...")

            if dev:
                # Dev mode: npm run dev
                cmd = [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(frontend_port),
                ]
            else:
                # Production mode: build and serve
                dist_dir = ui_dir / "dist"
                if not dist_dir.exists():
                    click.echo("   Building frontend for production...")
                    result = subprocess.run(
                        ["npm", "run", "build"],
                        cwd=ui_dir,
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode != 0:
                        click.secho(f"   Error building frontend: {result.stderr}", fg="red")
                        sys.exit(1)

                # Use npx serve to serve the dist directory
                cmd = [
                    "npx",
                    "serve",
                    "-s",
                    "dist",
                    "-l",
                    str(frontend_port),
                ]

            process = subprocess.Popen(
                cmd,
                cwd=ui_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            _write_frontend_pid(process.pid)

            # Wait and verify
            time.sleep(2)
            if process.poll() is not None:
                _remove_frontend_pid()
                click.secho("   Error: Frontend failed to start", fg="red")
                sys.exit(1)

            click.secho(f"   Frontend started (PID: {process.pid})", fg="green")

    # Summary
    click.echo("")
    click.secho("✅ Playground is running!", fg="green")
    click.echo("")

    if start_frontend:
        frontend_url = f"http://localhost:{frontend_port}"
        click.echo(f"   🌐 Frontend:  {frontend_url}")

    if start_backend:
        backend_url = f"http://localhost:{backend_port}"
        click.echo(f"   🔧 Backend:   {backend_url}")
        click.echo(f"   📚 API Docs:  {backend_url}/docs")

    click.echo("")
    click.echo("Use 'irca playground stop' to stop all services")

    # Open browser
    if open and start_frontend:
        time.sleep(1)  # Give frontend a moment to be ready
        frontend_url = f"http://localhost:{frontend_port}"
        click.echo(f"\nOpening browser to {frontend_url}...")
        webbrowser.open(frontend_url)


@playground.command("stop")
@click.option(
    "--backend-only",
    is_flag=True,
    default=False,
    help="Only stop the backend server",
)
@click.option(
    "--frontend-only",
    is_flag=True,
    default=False,
    help="Only stop the frontend server",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force stop with SIGKILL",
)
@click.pass_context
def stop(
    ctx: click.Context,
    backend_only: bool,
    frontend_only: bool,
    force: bool,
) -> None:
    """
    Stop the IRCA playground.

    Stops both the backend API server and the frontend UI.

    Examples:

        # Stop everything
        irca playground stop

        # Stop only frontend
        irca playground stop --frontend-only

        # Force stop
        irca playground stop --force
    """
    if backend_only and frontend_only:
        click.secho("Error: Cannot use both --backend-only and --frontend-only", fg="red")
        sys.exit(1)

    stop_backend = not frontend_only
    stop_frontend = not backend_only

    click.echo("🛑 Stopping IRCA Playground...")

    stopped_any = False

    # Stop frontend first (it depends on backend)
    if stop_frontend:
        frontend_running, frontend_pid = _is_frontend_running()
        if frontend_running:
            click.echo(f"   Stopping frontend (PID: {frontend_pid})...")
            try:
                sig = signal.SIGKILL if force else signal.SIGTERM
                # FM-5/FM-18 mitigation: Kill entire process group to clean up npm children
                # Since we use start_new_session=True, the PID is also the PGID
                try:
                    os.killpg(frontend_pid, sig)
                except ProcessLookupError:
                    # Process group gone, try individual process
                    os.kill(frontend_pid, sig)

                # Wait for termination
                for _ in range(50):  # 5 seconds max
                    if not _is_process_running(frontend_pid):
                        break
                    time.sleep(0.1)
                else:
                    # Force kill process group if still running
                    try:
                        os.killpg(frontend_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        os.kill(frontend_pid, signal.SIGKILL)
                    time.sleep(0.5)

                _remove_frontend_pid()
                click.secho("   Frontend stopped", fg="green")
                stopped_any = True
            except ProcessLookupError:
                _remove_frontend_pid()
                click.secho("   Frontend stopped", fg="green")
                stopped_any = True
            except PermissionError:
                click.secho("   Error: Permission denied to stop frontend", fg="red")
        else:
            click.echo("   Frontend not running")

    # Stop backend
    if stop_backend:
        backend_running, backend_pid = _is_server_running()
        if backend_running:
            click.echo(f"   Stopping backend (PID: {backend_pid})...")
            try:
                sig = signal.SIGKILL if force else signal.SIGTERM
                # FM-18 mitigation: Kill entire process group (uvicorn may spawn workers)
                try:
                    os.killpg(backend_pid, sig)
                except ProcessLookupError:
                    os.kill(backend_pid, sig)

                # Wait for termination
                for _ in range(100):  # 10 seconds max (backend needs time for cleanup)
                    if not _is_process_running(backend_pid):
                        break
                    time.sleep(0.1)
                else:
                    # Force kill process group if still running
                    try:
                        os.killpg(backend_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        os.kill(backend_pid, signal.SIGKILL)
                    time.sleep(0.5)

                from .server import _remove_pid as remove_server_pid
                remove_server_pid()
                click.secho("   Backend stopped", fg="green")
                stopped_any = True
            except ProcessLookupError:
                from .server import _remove_pid as remove_server_pid
                remove_server_pid()
                click.secho("   Backend stopped", fg="green")
                stopped_any = True
            except PermissionError:
                click.secho("   Error: Permission denied to stop backend", fg="red")
        else:
            click.echo("   Backend not running")

    if stopped_any:
        click.echo("")
        click.secho("✅ Playground stopped", fg="green")
    else:
        click.echo("")
        click.secho("⚫ Nothing was running", fg="yellow")


@playground.command("status")
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
    Check IRCA playground status.

    Shows the status of both backend and frontend services.

    Examples:

        # Check status
        irca playground status

        # JSON output
        irca playground status --json
    """
    import json
    import urllib.request
    import urllib.error

    backend_running, backend_pid = _is_server_running()
    frontend_running, frontend_pid = _is_frontend_running()

    status_info = {
        "backend": {
            "running": backend_running,
            "pid": backend_pid,
        },
        "frontend": {
            "running": frontend_running,
            "pid": frontend_pid,
        },
    }

    # Try to get backend health info
    if backend_running:
        try:
            backend_port = os.environ.get("IRCA_SERVER_PORT", "8000")
            url = f"http://127.0.0.1:{backend_port}/health"
            with urllib.request.urlopen(url, timeout=5) as response:
                health_data = json.loads(response.read().decode())
                status_info["backend"].update(health_data)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            status_info["backend"]["health_check"] = "failed"

    if output_json:
        click.echo(json.dumps(status_info, indent=2))
    else:
        click.echo("IRCA Playground Status")
        click.echo("=" * 40)

        # Backend status
        if backend_running:
            click.secho("Backend:  ✅ Running", fg="green")
            click.echo(f"  PID: {backend_pid}")
            if "status" in status_info["backend"]:
                click.echo(f"  Health: {status_info['backend'].get('status', 'unknown')}")
            if status_info["backend"].get("model_loaded"):
                click.echo(f"  Model: {status_info['backend'].get('current_model', 'unknown')}")
                idle = status_info["backend"].get("idle_seconds", 0)
                click.echo(f"  Idle: {idle // 60}m {idle % 60}s")
            else:
                click.echo("  Model: none loaded")
        else:
            click.secho("Backend:  ⚫ Stopped", fg="yellow")

        click.echo("")

        # Frontend status
        if frontend_running:
            click.secho("Frontend: ✅ Running", fg="green")
            click.echo(f"  PID: {frontend_pid}")
        else:
            click.secho("Frontend: ⚫ Stopped", fg="yellow")

        click.echo("")

        # Overall status
        if backend_running and frontend_running:
            click.secho("Playground is fully operational!", fg="green")
        elif backend_running or frontend_running:
            click.secho("Playground is partially running", fg="yellow")
        else:
            click.secho("Playground is not running", fg="yellow")
            click.echo("\nUse 'irca playground start' to start")


@playground.command("restart")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force restart",
)
@click.pass_context
def restart(ctx: click.Context, force: bool) -> None:
    """
    Restart the IRCA playground.

    Stops and starts both services.

    Examples:

        # Restart everything
        irca playground restart
    """
    ctx.invoke(stop, force=force)
    time.sleep(1)
    ctx.invoke(start)


@playground.command("logs")
@click.option(
    "--backend",
    "-b",
    is_flag=True,
    default=False,
    help="Show backend logs",
)
@click.option(
    "--frontend",
    "-f",
    is_flag=True,
    default=False,
    help="Show frontend logs",
)
@click.option(
    "--follow",
    "-F",
    is_flag=True,
    default=False,
    help="Follow log output",
)
@click.pass_context
def logs(ctx: click.Context, backend: bool, frontend: bool, follow: bool) -> None:
    """
    View playground logs.

    Note: Logs are only available when running in foreground mode.
    For background services, use system logs or run with --foreground.

    Examples:

        # Show info about logs
        irca playground logs
    """
    click.echo("Log viewing for background services is not yet implemented.")
    click.echo("")
    click.echo("For real-time logs, run services in foreground:")
    click.echo("  # Backend with logs:")
    click.echo("  poetry run uvicorn src.server.main:app --host 0.0.0.0 --port 8000")
    click.echo("")
    click.echo("  # Frontend with logs:")
    click.echo("  cd ui && npm run dev")
