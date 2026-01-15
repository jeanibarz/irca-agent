import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routers import generation, models, synthetic

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("irca-server")


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
    app.include_router(generation.router, prefix="/v1", tags=["generation"])
    app.include_router(models.router, prefix="/v1", tags=["models"])
    app.include_router(synthetic.router, prefix="/v1", tags=["synthetic"])

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host="0.0.0.0", port=8000, reload=True)
