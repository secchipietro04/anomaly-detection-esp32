# fastapi application factory with auto db init
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.database.session import init_database

@asynccontextmanager
async def lifespan(app: FastAPI):
    # init db on startup
    await init_database()
    yield

def create_app() -> FastAPI:
    app = FastAPI(title="edge-ai backend api", version="0.2.0", lifespan=lifespan)
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
