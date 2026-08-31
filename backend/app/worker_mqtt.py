# entrypoint for mqtt ingestion microservice
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.session import init_database, get_db_session
from app.database.models import db_get_capabilities, db_get_latest_health
from app.mqtt.client import MQTTClientManager
from app.mqtt.publisher import MQTTPublisher
from app.mqtt.handlers import set_nas_trigger_callback
from app.nas.dataset import build_training_dataset
from app.nas.search import NASSearchEngine
from app.nas.publisher import deploy_nas_result

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("worker_mqtt")

async def trigger_nas_pipeline(node_id: str):
    logger.info(f"Launching NAS pipeline for node {node_id}...")
    try:
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
            logger.info(f"NAS pipeline completed successfully for node {node_id}")
    except Exception as e:
        logger.error(f"Failed to execute NAS pipeline for node {node_id}: {e}", exc_info=True)

async def on_nas_threshold_reached(node_id: str, volume: int):
    logger.info(f"Notification: Node {node_id} reached {volume} bytes. Spawning NAS pipeline task...")
    asyncio.create_task(trigger_nas_pipeline(node_id))

async def main():
    logger.info("Initializing database...")
    await init_database()
    logger.info("Starting MQTT Ingestion Worker...")
    set_nas_trigger_callback(on_nas_threshold_reached)
    client = MQTTClientManager()
    await client.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("MQTT Worker shutting down...")
