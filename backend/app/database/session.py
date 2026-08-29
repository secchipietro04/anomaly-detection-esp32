# asyncpg session manager
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional, List, Any
import asyncpg
from app.config import get_settings

class DatabaseSessionManager:
    def __init__(self):
        # connection pool handle
        self._pool: Optional[asyncpg.Pool] = None

    async def init(
        self,
        dsn: Optional[str] = None,
        min_size: int = 2,
        max_size: int = 10,
        **kwargs
    ) -> None:
        # initialize asyncpg connection pool
        if self._pool is not None:
            return
        
        target_dsn = dsn or get_settings().dsn
        # create pool
        self._pool = await asyncpg.create_pool(
            dsn=target_dsn,
            min_size=min_size,
            max_size=max_size,
            **kwargs
        )

    async def close(self) -> None:
        # close active pool
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        # acquire connection from pool
        if self._pool is None:
            await self.init()
        assert self._pool is not None, "database pool not initialized"
        
        async with self._pool.acquire() as conn:
            yield conn

    async def execute(self, query: str, *args) -> str:
        # execute raw sql query
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        # fetch multiple rows
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        # fetch single row
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        # fetch single scalar value
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def execute_many(self, query: str, args_list: List[tuple]) -> None:
        # execute query with parameter list
        async with self.acquire() as conn:
            await conn.executemany(query, args_list)

    async def init_db(self, schema_path: Optional[str] = None) -> None:
        # run init.sql DDL script
        import os
        if not schema_path:
            # check default locations
            candidates = [
                os.path.join(os.path.dirname(__file__), "../../../database/init.sql"),
                os.path.join(os.path.dirname(__file__), "../../database/init.sql"),
                "database/init.sql",
                "/app/database/init.sql",
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    schema_path = candidate
                    break
        
        if schema_path and os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                sql = f.read()
            async with self.acquire() as conn:
                await conn.execute(sql)

    def is_connected(self) -> bool:
        # Check pool status
        return self._pool is not None and not self._pool._closed

# global session manager instance
db_manager = DatabaseSessionManager()

async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    # fastAPI dependency helper
    async with db_manager.acquire() as conn:
        yield conn
