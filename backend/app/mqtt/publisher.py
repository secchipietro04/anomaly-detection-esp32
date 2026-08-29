# MQTT publisher service for downlink commands and configs
import asyncio
import logging
from typing import Optional, Dict, Any
import aiomqtt

from app.config import get_settings

logger = logging.getLogger("mqtt_publisher")

class MQTTPublisher:
    # async MQTT publisher for downlink topics
    def __init__(
        self,
        broker: Optional[str] = None,
        port: Optional[int] = None,
        topic_prefix: Optional[str] = None,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        settings = get_settings()
        self.broker = broker or settings.mqtt_broker
        self.port = port or settings.mqtt_port
        self.topic_prefix = topic_prefix or settings.mqtt_topic_prefix
        self.client_id = client_id
        self.username = username
        self.password = password
        self._client: Optional[aiomqtt.Client] = None

    async def connect(self) -> aiomqtt.Client:
        # connect to mqtt broker
        if self._client is not None:
            return self._client
        client_kwargs: Dict[str, Any] = {
            "hostname": self.broker,
            "port": self.port,
        }
        if self.username:
            client_kwargs["username"] = self.username
        if self.password:
            client_kwargs["password"] = self.password
        if self.client_id:
            client_kwargs["identifier"] = self.client_id
            
        self._client = aiomqtt.Client(**client_kwargs)
        await self._client.__aenter__()
        return self._client

    async def disconnect(self) -> None:
        # disconnect active client
        if self._client is not None:
            try:
                await self._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

    async def publish(
        self,
        topic: str,
        payload: bytes,
        qos: int = 1,
        retain: bool = False
    ) -> None:
        # publish binary cbor payload
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("payload must be bytes-like")
            
        if self._client is not None:
            await self._client.publish(topic, payload=bytes(payload), qos=qos, retain=retain)
        else:
            client_kwargs: Dict[str, Any] = {
                "hostname": self.broker,
                "port": self.port,
            }
            if self.username:
                client_kwargs["username"] = self.username
            if self.password:
                client_kwargs["password"] = self.password
            if self.client_id:
                client_kwargs["identifier"] = self.client_id
            async with aiomqtt.Client(**client_kwargs) as client:
                await client.publish(topic, payload=bytes(payload), qos=qos, retain=retain)

    async def publish_config(
        self,
        node_id: str,
        payload: bytes,
        qos: int = 1
    ) -> None:
        # publish runtime config downlink to sensor
        topic = f"{self.topic_prefix}/{node_id}/config"
        await self.publish(topic, payload=payload, qos=qos)

    async def publish_ensemble(
        self,
        node_id: str,
        payload: bytes,
        qos: int = 1
    ) -> None:
        # publish ensemble or model package downlink
        topic = f"{self.topic_prefix}/{node_id}/ensemble"
        await self.publish(topic, payload=payload, qos=qos)
