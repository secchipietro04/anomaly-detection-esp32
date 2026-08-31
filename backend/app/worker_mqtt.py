# entrypoint for mqtt ingestion microservice
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.session import init_database
from app.mqtt.client import MQTTClientManager
from app.mqtt.handlers import set_nas_trigger_callback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("worker_mqtt")

async def on_nas_threshold_reached(node_id: str, volume: int):
    logger.info(f"Notification: Node {node_id} reached {volume} bytes. Ready for NAS!")

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
