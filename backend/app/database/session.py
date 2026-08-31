# sqlalchemy async engine, sessionmaker and automatic timescaledb table init
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import get_settings
from app.database.models import Base

def _get_async_url(dsn: str) -> str:
    # convert postgresql:// to postgresql+asyncpg://
    if dsn.startswith("postgresql://"):
        return dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
    if not dsn.startswith("postgresql+asyncpg://"):
        return f"postgresql+asyncpg://{dsn}"
    return dsn

settings = get_settings()
engine = create_async_engine(
    _get_async_url(settings.dsn),
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

async def init_database() -> None:
    # creates tables, timescaledb extension and hypertables automatically
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
        await conn.run_sync(Base.metadata.create_all)
        # convert raw_telemetry and inference_results to hypertables
        try:
            await conn.execute(text("SELECT create_hypertable('raw_telemetry', 'timestamp', if_not_exists => TRUE);"))
        except Exception:
            pass
        try:
            await conn.execute(text("SELECT create_hypertable('inference_results', 'timestamp', if_not_exists => TRUE);"))
        except Exception:
            pass

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    # async context manager yielding database session
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

async def get_session_dependency() -> AsyncGenerator[AsyncSession, None]:
    # fastapi dependency provider
    async with get_db_session() as session:
        yield session
