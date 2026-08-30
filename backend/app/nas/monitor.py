# in-memory volume tracker helper
from typing import Optional, Dict
from sqlalchemy.ext.asyncio import AsyncSession
from app.mqtt.handlers import _volume_accumulator, _get_or_init_volume

async def get_node_accumulated_bytes(session: AsyncSession, node_id: str) -> int:
    # return tracked volume in memory
    return await _get_or_init_volume(session, node_id)
