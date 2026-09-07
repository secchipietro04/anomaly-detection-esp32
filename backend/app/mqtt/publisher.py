# async mqtt publisher for downlinks, models and retained state cleanup
import asyncio
import logging
from typing import Optional, Dict, Any
import paho.mqtt.client as mqtt
from app.config import get_settings

logger = logging.getLogger("mqtt_publisher")

class MQTTPublisher:
    # publishes downlinks and manages model endpoints
    def __init__(
        self,
        broker: Optional[str] = None,
        port: Optional[int] = None,
        topic_prefix: Optional[str] = None,
        client_id: Optional[str] = None,
    ):
        settings = get_settings()
        self.broker = broker or settings.mqtt_broker
        self.port = port or settings.mqtt_port
        self.topic_prefix = topic_prefix or settings.mqtt_topic_prefix
        self.client_id = client_id

    async def connect(self):
        return self

    async def disconnect(self):
        pass

    async def publish(
        self,
        topic: str,
        payload: bytes,
        qos: int = 1,
        retain: bool = False
    ) -> None:
        # publish payload to topic using executor
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
        
        def _sync_pub():
            client = mqtt.Client()
            client.connect(self.broker, self.port, keepalive=30)
            client.loop_start()
            try:
                res = client.publish(topic, payload=bytes(payload), qos=qos, retain=retain)
                res.wait_for_publish(timeout=10.0)
            finally:
                client.loop_stop()
                client.disconnect()

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_pub)

    async def publish_config(self, node_id: str, payload: bytes, qos: int = 1, retain: bool = False) -> None:
        # publish runtime config downlink to sensor
        topic = f"{self.topic_prefix}/{node_id}/config"
        await self.publish(topic, payload=payload, qos=qos, retain=retain)

    async def publish_ensemble(self, node_id: str, payload: bytes, qos: int = 1, retain: bool = False) -> None:
        # publish lightweight ensemble routing table
        topic = f"{self.topic_prefix}/{node_id}/ensemble"
        await self.publish(topic, payload=payload, qos=qos, retain=retain)

    async def publish_model(
        self,
        node_id: str,
        model_type: str,
        model_id: int,
        payload: bytes,
        qos: int = 1,
        retain: bool = False
    ) -> None:
        # publish dedicated model binary on separate retained endpoint
        # model_type: router, memory, or submodel
        topic = f"{self.topic_prefix}/{node_id}/models/{model_type}/{model_id}"
        await self.publish(topic, payload=payload, qos=qos, retain=retain)

    async def clear_retained_model(
        self,
        node_id: str,
        model_type: str,
        model_id: int
    ) -> None:
        # clear emqx retained model topic with empty payload
        topic = f"{self.topic_prefix}/{node_id}/models/{model_type}/{model_id}"
        await self.publish(topic, payload=b"", qos=1, retain=True)
        logger.info(f"Cleared retained model on EMQX: {topic}")

    async def publish_command(
        self,
        node_id: str,
        command: str,
        payload: bytes = b"",
        qos: int = 1
    ) -> None:
        # publish control command downlink to sensor: reboot, dump, dump_i, updt_status
        topic = f"{self.topic_prefix}/{node_id}/cmd/{command}"
        await self.publish(topic, payload=payload, qos=qos, retain=False)
        logger.info(f"Published command '{command}' to sensor {node_id}")

