# sqlalchemy async engine, sessionmaker and robust automatic timescaledb table init
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.config import get_settings
from app.database.models import Base

logger = logging.getLogger("db_session")

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

async def init_database(max_retries: int = 10, retry_delay: float = 2.0) -> None:
    # robust auto-init creating tables, extensions and hypertables on startup
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Database auto-init attempt {attempt}/{max_retries}...")
            
            # 1. create timescaledb extension
            try:
                async with engine.begin() as conn:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            except Exception as e:
                logger.warning(f"Notice enabling timescaledb extension: {e}")

            # 2. create all relational tables
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Successfully created all SQLAlchemy relational tables")

            # 3. create hypertables
            for table_name in ["raw_telemetry", "inference_results"]:
                try:
                    async with engine.begin() as conn:
                        await conn.execute(text(f"SELECT create_hypertable('{table_name}', 'timestamp', if_not_exists => TRUE);"))
                except Exception as e:
                    logger.debug(f"Hypertable notice for {table_name}: {e}")

            logger.info("Database schema initialized and ready!")
            return
        except Exception as e:
            logger.warning(f"Database init attempt {attempt} failed: {e}. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)

    logger.error("Failed to initialize database after all retries!")

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
    # fastapi dependency yielding async session
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
