# Background ingestion worker entrypoint
import asyncio
import logging
import signal
import sys
from app.config import get_settings
from app.database.session import db_manager
from app.mqtt.consumer import MQTTConsumer

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("worker")

async def run_worker() -> None:
    # initialize database pool and mqtt consumer loop
    settings = get_settings()
    logger.info("Starting edge-AI telemetry worker...")
    
    # initialize database connection pool
    try:
        await db_manager.init(settings.dsn)
        logger.info("Database pool initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        sys.exit(1)

    consumer = MQTTConsumer(
        broker=settings.mqtt_broker,
        port=settings.mqtt_port,
        topic_prefix=settings.mqtt_topic_prefix,
        session_manager=db_manager
    )

    loop = asyncio.get_running_loop()
    
    def shutdown_signal():
        logger.info("Received termination signal, stopping worker...")
        consumer.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown_signal)
        except NotImplementedError:
            # Signal handlers not implemented on some platforms (e.g. Windows)
            pass

    try:
        await consumer.listen()
    except Exception as e:
        logger.error(f"Worker runtime error: {e}", exc_info=True)
    finally:
        logger.info("Closing database pool...")
        await db_manager.close()
        logger.info("Worker shutdown complete")

def main():
    # sync entrypoint
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.info("Worker interrupted by user")

if __name__ == "__main__":
    main()
