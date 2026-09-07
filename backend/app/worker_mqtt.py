# entrypoint for mqtt ingestion and event-driven nas microservice
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

_active_nas_nodes: set = set()

async def trigger_nas_pipeline(node_id: str):
    if node_id in _active_nas_nodes:
        logger.info(f"NAS pipeline already running for node {node_id}. Skipping duplicate trigger.")
        return
    
    _active_nas_nodes.add(node_id)
    try:
        logger.info(f"Event-driven NAS: Starting optimization for node {node_id} (1MB threshold reached)...")
        # 1. fetch dataset and capabilities in short-lived session
        async with get_db_session() as session:
            train_data, val_data = await build_training_dataset(session, node_id)
            caps = await db_get_capabilities(session, node_id)
            enabled_ops = caps.enabled_ops if caps else None
            
            health = await db_get_latest_health(session, node_id)
            ram_free = health.ram_free if health else None
            
        # 2. run optuna NAS search with train and holdout validation sets
        engine = NASSearchEngine(node_id=node_id, enabled_ops=enabled_ops, ram_free=ram_free)
        result = engine.search(train_dataset=train_data, val_dataset=val_data)
        
        # 3. deploy models and ensemble with fresh session
        async with get_db_session() as session:
            publisher = MQTTPublisher()
            await deploy_nas_result(session, publisher, node_id, result)
            logger.info(f"Event-driven NAS: Successfully deployed models and ensemble for node {node_id}")
    except Exception as e:
        logger.error(f"Failed to execute NAS pipeline for node {node_id}: {e}", exc_info=True)
    finally:
        _active_nas_nodes.discard(node_id)

async def on_nas_threshold_reached(node_id: str, volume: int):
    logger.info(f"RAM Event: Node {node_id} reached {volume} bytes. Spawning event-driven NAS pipeline...")
    asyncio.create_task(trigger_nas_pipeline(node_id))

async def main():
    logger.info("Initializing database...")
    await init_database()
    logger.info("Starting MQTT Ingestion and Event-Driven NAS Service...")
    set_nas_trigger_callback(on_nas_threshold_reached)
    client = MQTTClientManager()
    await client.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("MQTT Service shutting down...")
