# sqlalchemy async engine and sessionmaker
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import get_settings

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
