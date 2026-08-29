# Main entrypoint for MQTT ingestion service
import asyncio
import logging
import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import get_settings
from app.database.session import db_manager
from app.mqtt.consumer import MQTTConsumer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

async def main_async() -> None:
    # initialize database connection pool and start consumer
    settings = get_settings()
    logger.info("Initializing backend MQTT ingestion service...")
    
    # init db pool
    await db_manager.init(settings.dsn)
    
    consumer = MQTTConsumer(
        broker=settings.mqtt_broker,
        port=settings.mqtt_port,
        topic_prefix=settings.mqtt_topic_prefix,
        session_manager=db_manager
    )
    
    try:
        await consumer.listen()
    except Exception as e:
        logger.error(f"Consumer encountered error: {e}")
    finally:
        await db_manager.close()

def main() -> None:
    # run async loop
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Service shutting down...")

if __name__ == "__main__":
    main()