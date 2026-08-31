# entrypoint for nas training & periodic optimization worker
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from app.config import get_settings
from app.database.session import get_db_session
from app.database.models import NodeModel, db_get_untrained_bytes_for_node, db_get_capabilities, db_get_latest_health
from app.mqtt.publisher import MQTTPublisher
from app.nas.dataset import build_training_dataset
from app.nas.search import NASSearchEngine
from app.nas.publisher import deploy_nas_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("worker_nas")

async def run_nas_for_node(node_id: str):
    # run full nas pipeline for single node
    logger.info(f"Worker NAS: Starting optimization for node {node_id}...")
    async with get_db_session() as session:
        dataset = await build_training_dataset(session, node_id)
        caps = await db_get_capabilities(session, node_id)
        enabled_ops = caps.enabled_ops if caps else None
        
        health = await db_get_latest_health(session, node_id)
        ram_free = health.ram_free if health else None
        
        engine = NASSearchEngine(node_id=node_id, enabled_ops=enabled_ops, ram_free=ram_free)
        result = engine.search(dataset)
        
        publisher = MQTTPublisher()
        await deploy_nas_result(session, publisher, node_id, result)
        logger.info(f"Worker NAS: Finished optimization and deployed ensemble for node {node_id}")

async def nas_scan_loop():
    # periodic scanner checking nodes with accumulated telemetry
    settings = get_settings()
    while True:
        try:
            async with get_db_session() as session:
                stmt = select(NodeModel)
                res = await session.execute(stmt)
                nodes = list(res.scalars().all())
                for node in nodes:
                    untrained_bytes = await db_get_untrained_bytes_for_node(session, node.node_id)
                    if untrained_bytes >= settings.nas_data_threshold_bytes:
                        logger.info(f"Node {node.node_id} has {untrained_bytes} untrained bytes >= threshold. Running NAS...")
                        await run_nas_for_node(node.node_id)
        except Exception as e:
            logger.error(f"Error in NAS scan loop: {e}", exc_info=True)
        await asyncio.sleep(60)

async def main():
    logger.info("Starting NAS Optimization Worker...")
    await nas_scan_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("NAS Worker shutting down...")
