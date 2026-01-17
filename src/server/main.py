import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from src.config import get_settings
from src.server.events import EventBroadcaster
from src.server.routers import conversations, generation, models, synthetic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("irca-server")


# Initialize settings
settings = get_settings()


# Shutdown state
_shutdown_event: asyncio.Event | None = None
_idle_monitor_task: asyncio.Task | None = None


async def _eject_all_models_with_timeout(timeout: float = 5.0) -> None:
    """
    Eject all loaded models with a bounded timeout (FM-4 mitigation).

    CUDA operations can hang in certain conditions; this ensures we don't
    block shutdown indefinitely.
    """
    from src.server.model_manager import ModelManager

    manager = ModelManager.get_instance()

    # Get all loaded model aliases
    aliases = list(manager.models.keys())
    if not aliases:
        logger.info("No models loaded, skipping ejection")
        return

    logger.info(f"Ejecting {len(aliases)} model(s) before shutdown...")

    for alias in aliases:
        try:
            await asyncio.wait_for(manager.eject_model(alias), timeout=timeout)
            logger.info(f"Model '{alias}' ejected successfully")
        except asyncio.TimeoutError:
            logger.warning(f"Model '{alias}' ejection timed out after {timeout}s")
        except Exception as e:
            logger.error(f"Error ejecting model '{alias}': {e}")

    # Final CUDA cleanup
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")
    except Exception as e:
        logger.warning(f"CUDA cleanup failed: {e}")


def _setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """
    Setup signal handlers for graceful shutdown (FM-8 mitigation).

    Uses asyncio.run_coroutine_threadsafe to safely schedule coroutines
    from signal handlers.

    Note: Signal handlers can only be registered in the main thread.
    This will fail silently in test environments.
    """
    global _shutdown_event

    def signal_handler(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")

        if _shutdown_event is not None:
            # Schedule the shutdown event to be set from the main loop
            loop.call_soon_threadsafe(_shutdown_event.set)

    try:
        # Register handlers for SIGTERM and SIGINT
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    except ValueError:
        # Signal handlers can only be set in main thread
        # This is expected in test environments
        logger.debug("Could not set signal handlers (not in main thread)")


async def _idle_monitor() -> None:
    """
    Background task to monitor idle time and auto-eject models.

    Runs every minute and checks if the model has been idle longer
    than the configured timeout.
    """
    from src.server.model_manager import ModelManager

    # Get idle timeout from environment (set by CLI or settings)
    idle_timeout_minutes = int(os.environ.get("IRCA_SERVER_IDLE_TIMEOUT", "30"))

    if idle_timeout_minutes <= 0:
        logger.info("Idle monitor disabled (timeout=0)")
        return

    idle_timeout_seconds = idle_timeout_minutes * 60
    logger.info(f"Idle monitor started (timeout={idle_timeout_minutes}m)")

    manager = ModelManager.get_instance()

    while True:
        try:
            await asyncio.sleep(60)  # Check every minute

            # Check if any models are loaded
            if not manager.models:
                continue

            # Get idle time
            idle_seconds = manager.get_idle_seconds()

            if idle_seconds >= idle_timeout_seconds:
                logger.info(f"Model idle for {idle_seconds}s (timeout={idle_timeout_seconds}s), auto-ejecting...")

                # Eject all models (FM-2 mitigation: activity is updated before lock in generate)
                for alias in list(manager.models.keys()):
                    try:
                        await manager.eject_model(alias)
                        logger.info(f"Auto-ejected model '{alias}' due to idle timeout")
                    except Exception as e:
                        logger.error(f"Failed to auto-eject model '{alias}': {e}")

        except asyncio.CancelledError:
            logger.info("Idle monitor cancelled")
            break
        except Exception as e:
            logger.error(f"Idle monitor error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    FastAPI lifespan context manager for graceful startup and shutdown.

    Handles:
    - Signal handler registration for SIGTERM/SIGINT
    - Idle monitor task startup
    - Model ejection on shutdown
    """
    global _shutdown_event, _idle_monitor_task

    logger.info("IRCA server starting up...")

    # Create shutdown event
    _shutdown_event = asyncio.Event()

    # Setup signal handlers
    loop = asyncio.get_running_loop()
    _setup_signal_handlers(loop)

    # Start idle monitor
    _idle_monitor_task = asyncio.create_task(_idle_monitor())

    try:
        yield
    finally:
        logger.info("IRCA server shutting down...")

        # Cancel idle monitor
        if _idle_monitor_task is not None:
            _idle_monitor_task.cancel()
            try:
                await _idle_monitor_task
            except asyncio.CancelledError:
                pass

        # Eject all models
        await _eject_all_models_with_timeout()

        logger.info("IRCA server shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="IRCA Agent Playground API",
        description="Backend API for IRCA Agent Playground",
        version="1.0.0",
        lifespan=lifespan,
    )

    # FM-02: CORS - use environment variable for allowed origins
    # Default to localhost for development; set IRCA_CORS_ORIGINS for production
    cors_origins_env = os.environ.get("IRCA_CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
    cors_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
    logger.info(f"CORS allowed origins: {cors_origins}")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(generation.router, prefix="/v1/chat", tags=["chat"])
    app.include_router(models.router, prefix="/v1", tags=["models"])
    app.include_router(synthetic.router, prefix="/v1/synthetic", tags=["synthetic"])
    app.include_router(conversations.router, prefix="/v1", tags=["conversations"])

    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """
        Health check endpoint with model status.

        Returns server health and information about loaded models,
        including idle time for resource monitoring.
        """
        from src.server.model_manager import ModelManager

        manager = ModelManager.get_instance()

        # Get idle timeout from environment
        idle_timeout_minutes = int(os.environ.get("IRCA_SERVER_IDLE_TIMEOUT", "30"))

        # Build response
        response: dict[str, Any] = {"status": "healthy"}

        # Backend info
        response["inference_backend"] = "unsloth" if manager._using_unsloth else "transformers"

        # Model status
        if manager.models:
            # Get first (or default) model info
            aliases = list(manager.models.keys())
            current_alias = "default" if "default" in aliases else aliases[0]

            response["model_loaded"] = True
            response["current_model"] = manager.loaded_configs.get(current_alias, {}).get("base", "unknown")
            response["loaded_aliases"] = aliases
            response["idle_seconds"] = manager.get_idle_seconds()
            response["idle_timeout_minutes"] = idle_timeout_minutes
        else:
            response["model_loaded"] = False
            response["current_model"] = None
            response["idle_timeout_minutes"] = idle_timeout_minutes

        return response

    @app.get("/v1/events")
    async def events_stream() -> StreamingResponse:
        """Stream server-side events (SSE)."""
        broadcaster = EventBroadcaster.get_instance()
        return StreamingResponse(
            broadcaster.subscribe(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("IRCA_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("IRCA_SERVER_PORT", "8000"))

    uvicorn.run("src.server.main:app", host=host, port=port, reload=True)
