# entrypoint for nas training & garbage collection worker
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.session import get_db_session
from app.nas.garbage_collector import run_garbage_collection
from app.mqtt.publisher import MQTTPublisher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("worker_nas")

async def gc_loop():
    # periodic garbage collection every 10 minutes
    publisher = MQTTPublisher()
    while True:
        try:
            async with get_db_session() as session:
                await run_garbage_collection(session, publisher, grace_period_hours=1)
        except Exception as e:
            logger.error(f"Error in GC loop: {e}")
        await asyncio.sleep(600)

async def main():
    logger.info("Starting NAS & Garbage Collection Worker...")
    await gc_loop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("NAS Worker shutting down...")
