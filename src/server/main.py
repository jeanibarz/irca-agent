import logging

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


def create_app() -> FastAPI:
    app = FastAPI(
        title="IRCA Agent Playground API", description="Backend API for IRCA Agent Playground", version="1.0.0"
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # For dev
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
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

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

    uvicorn.run("src.server.main:app", host="0.0.0.0", port=8000, reload=True)
