# idle worker - NAS is driven on-demand in memory by worker_mqtt
import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("worker_nas")

async def main():
    logger.info("NAS Worker initialized in event-driven mode. Standby...")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
