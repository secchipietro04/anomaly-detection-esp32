# telemetry data volume tracker and nas trigger monitor
import asyncio
import inspect
import logging
from typing import Optional, Dict, Tuple, Callable, Any
import asyncpg

from app.config import get_settings

logger = logging.getLogger("nas_monitor")

NAS_THRESHOLD_BYTES = 1048576  # 1MB = 1024 * 1024 bytes

async def get_accumulated_telemetry_bytes(
    conn: asyncpg.Connection,
    node_id: str,
    since_segment_id: Optional[int] = None
) -> int:
    # calculate accumulated raw telemetry bytes for sensor node
    if since_segment_id is not None:
        val = await conn.fetchval(
            """
            SELECT COALESCE(SUM(raw_bytes_count), 0)
            FROM raw_telemetry
            WHERE node_id = $1 AND segment_id > $2;
            """,
            node_id, since_segment_id
        )
    else:
        val = await conn.fetchval(
            """
            SELECT COALESCE(SUM(raw_bytes_count), 0)
            FROM raw_telemetry
            WHERE node_id = $1;
            """,
            node_id
        )
    return int(val) if val is not None else 0

async def get_latest_segment_id(
    conn: asyncpg.Connection,
    node_id: str
) -> int:
    # get max segment id seen so far for node
    val = await conn.fetchval(
        """
        SELECT COALESCE(MAX(segment_id), 0)
        FROM raw_telemetry
        WHERE node_id = $1;
        """,
        node_id
    )
    return int(val) if val is not None else 0

async def check_volume_threshold(
    conn: asyncpg.Connection,
    node_id: str,
    threshold_bytes: Optional[int] = None,
    since_segment_id: Optional[int] = None
) -> Tuple[bool, int]:
    # check if untrained raw data volume exceeds threshold (default 1MB)
    settings = get_settings()
    threshold = threshold_bytes if threshold_bytes is not None else settings.nas_data_threshold_bytes
    current_bytes = await get_accumulated_telemetry_bytes(conn, node_id, since_segment_id)
    should_trigger = current_bytes > threshold
    return should_trigger, current_bytes

class DataVolumeMonitor:
    # stateful monitor tracking data volumes and triggering nas
    def __init__(
        self,
        threshold_bytes: Optional[int] = None,
        on_trigger_callback: Optional[Callable[[str, int], Any]] = None
    ):
        settings = get_settings()
        self.threshold_bytes = threshold_bytes or settings.nas_data_threshold_bytes
        self.on_trigger_callback = on_trigger_callback
        self._last_trained_segments: Dict[str, int] = {}
        self._in_progress: Dict[str, bool] = {}

    def get_last_trained_segment(self, node_id: str) -> Optional[int]:
        # return last checkpoint segment for node
        return self._last_trained_segments.get(node_id)

    def mark_trained(self, node_id: str, segment_id: int) -> None:
        # mark training checkpoint for node
        self._last_trained_segments[node_id] = segment_id
        self._in_progress[node_id] = False
        logger.info(f"Node {node_id} marked as trained up to segment {segment_id}")

    async def check_node(
        self,
        conn: asyncpg.Connection,
        node_id: str
    ) -> Tuple[bool, int]:
        # check volume for specific node
        since_seg = self._last_trained_segments.get(node_id)
        return await check_volume_threshold(
            conn=conn,
            node_id=node_id,
            threshold_bytes=self.threshold_bytes,
            since_segment_id=since_seg
        )

    async def check_and_trigger(
        self,
        conn: asyncpg.Connection,
        node_id: str,
        trigger_callback: Optional[Callable[[str, int], Any]] = None
    ) -> bool:
        # check volume and trigger callback if threshold exceeded
        if self._in_progress.get(node_id, False):
            logger.debug(f"NAS already in progress for node {node_id}, skipping trigger")
            return False

        should_trigger, volume = await self.check_node(conn, node_id)
        if should_trigger:
            logger.info(f"Node {node_id} volume {volume} bytes > threshold {self.threshold_bytes}. Triggering NAS...")
            self._in_progress[node_id] = True
            
            cb = trigger_callback or self.on_trigger_callback
            if cb is not None:
                try:
                    if inspect.iscoroutinefunction(cb):
                        await cb(node_id, volume)
                    else:
                        cb(node_id, volume)
                except Exception as e:
                    logger.error(f"Error in NAS trigger callback for node {node_id}: {e}", exc_info=True)
                    self._in_progress[node_id] = False
                    return False
            return True
        return False
