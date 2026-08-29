# fastapi app factory with lifespan db init
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database.session import db_manager
from app.api.routes import router

logger = logging.getLogger("api.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: init database pool
    settings = get_settings()
    logger.info("api server starting, connecting to database...")
    await db_manager.init(settings.dsn)
    yield
    # shutdown
    logger.info("api server shutting down, closing db pool")
    await db_manager.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="edge-ai anomaly detection api",
        version="0.1.0",
        lifespan=lifespan,
    )

    # allow grafana and local UIs to call the api
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    @app.get("/health")
    async def health_check():
        return {"status": "ok"}

    return app


app = create_app()
