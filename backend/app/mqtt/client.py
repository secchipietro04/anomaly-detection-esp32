# async mqtt client and event dispatcher
import asyncio
import logging
from typing import Optional
import aiomqtt

from app.config import get_settings
from app.database.session import get_db_session
from app.mqtt.router import MQTTRouter
from app.mqtt.publisher import MQTTPublisher
from app.mqtt.handlers import router as default_router

logger = logging.getLogger("mqtt_client")

class MQTTClientManager:
    # manages async connection and dispatches messages to router handlers
    def __init__(
        self,
        router: Optional[MQTTRouter] = None,
        broker: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[str] = None,
    ):
        settings = get_settings()
        self.router = router or default_router
        self.broker = broker or settings.mqtt_broker
        self.port = port or settings.mqtt_port
        self.client_id = client_id or "backend_mqtt_worker"
        self.publisher = MQTTPublisher(broker=self.broker, port=self.port)
        self._running = False

    async def start(self) -> None:
        # connect, subscribe and listen loop
        self._running = True
        subscription_patterns = self.router.get_subscription_topics()

        while self._running:
            try:
                logger.info(f"Connecting to MQTT broker at {self.broker}:{self.port}...")
                async with aiomqtt.Client(hostname=self.broker, port=self.port, identifier=self.client_id) as client:
                    logger.info("Connected to MQTT broker. Subscribing to topics...")
                    for pattern in subscription_patterns:
                        await client.subscribe(pattern, qos=1)
                        logger.info(f"Subscribed: {pattern}")

                    async for message in client.messages:
                        if not self._running:
                            break
                        await self._dispatch_message(str(message.topic), message.payload)
            except aiomqtt.MqttError as e:
                logger.warning(f"MQTT connection error: {e}. Reconnecting in 3s...")
                await asyncio.sleep(3)
            except Exception as e:
                logger.error(f"Unexpected MQTT listener error: {e}", exc_info=True)
                await asyncio.sleep(3)

    async def _dispatch_message(self, topic: str, payload: bytes) -> None:
        # resolve route and invoke matching handler with injected session and publisher
        res = self.router.resolve(topic)
        if res is None:
            return
        handler, kwargs = res
        try:
            async with get_db_session() as session:
                kwargs["session"] = session
                kwargs["publisher"] = self.publisher
                kwargs["payload"] = payload
                await handler(**kwargs)
        except Exception as e:
            logger.error(f"Error handling message on {topic}: {e}", exc_info=True)

    def stop(self) -> None:
        # stop listener loop
        self._running = False
