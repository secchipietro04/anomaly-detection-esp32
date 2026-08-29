# MQTT consumer and ingestion service
import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Callable
import aiomqtt

from app.config import get_settings
from app.database.session import DatabaseSessionManager, db_manager
from app.database.models import (
    Node,
    NodeCapabilities,
    NodeHealth,
    RawTelemetry,
    InferenceResult,
    register_node,
    update_node_last_seen,
    upsert_node_capabilities,
    insert_node_health,
    insert_raw_telemetry,
    insert_inference_result,
)
from app.cbor.codec import (
    CBORError,
    decode_segment,
    decode_inference_packet,
    decode_node_capabilities,
    decode_node_health,
    decode_node_alert,
    encode_runtime_config,
    encode_ensemble_config,
    encode_model_package,
)
from app.cbor.schemas import (
    Segment,
    InferencePacket,
    NodeCapabilities as CBORNodeCapabilities,
    NodeHealthInfo,
    RuntimeConfig,
    EnsembleConfig,
    ModelPackageUnion,
)

logger = logging.getLogger("mqtt_consumer")

class MQTTConsumer:
    # async MQTT subscriber for telemetry and sensor management
    def __init__(
        self,
        broker: Optional[str] = None,
        port: Optional[int] = None,
        topic_prefix: Optional[str] = None,
        session_manager: Optional[DatabaseSessionManager] = None,
        client_id: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        settings = get_settings()
        self.broker = broker or settings.mqtt_broker
        self.port = port or settings.mqtt_port
        self.topic_prefix = topic_prefix or settings.mqtt_topic_prefix
        self.db = session_manager or db_manager
        self.client_id = client_id
        self.username = username
        self.password = password
        
        self._running = False
        self._client: Optional[aiomqtt.Client] = None
        self._stop_event = asyncio.Event()
        
        # topic patterns
        prefix = re.escape(self.topic_prefix)
        self._patterns = [
            (re.compile(rf"^{prefix}/(?P<node_id>[^/]+)/info/caps$"), self._on_caps),
            (re.compile(rf"^{prefix}/(?P<node_id>[^/]+)/data/(?P<sensor_type>[^/]+)$"), self._on_telemetry),
            (re.compile(rf"^{prefix}/(?P<node_id>[^/]+)/inference$"), self._on_inference),
            (re.compile(rf"^{prefix}/(?P<node_id>[^/]+)/info/health$"), self._on_health),
            (re.compile(rf"^{prefix}/(?P<node_id>[^/]+)/alert/(?P<severity>[^/]+)$"), self._on_alert),
        ]

    @property
    def subscription_topics(self) -> List[str]:
        # list of mqtt wildcard topics
        return [
            f"{self.topic_prefix}/+/info/caps",
            f"{self.topic_prefix}/+/data/sensor",
            f"{self.topic_prefix}/+/data/+",
            f"{self.topic_prefix}/+/inference",
            f"{self.topic_prefix}/+/info/health",
            f"{self.topic_prefix}/+/alert/+",
        ]

    async def handle_message(self, topic: str, payload: bytes) -> bool:
        # route incoming binary message to registered handler
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            logger.error("Rejecting non-binary payload")
            return False

        for pattern, handler in self._patterns:
            match = pattern.match(topic)
            if match:
                groups = match.groupdict()
                try:
                    await handler(topic=topic, payload=bytes(payload), **groups)
                    return True
                except CBORError as e:
                    logger.warning(f"CBOR decoding failure on topic {topic}: {e}")
                    return False
                except Exception as e:
                    logger.error(f"Error handling message on {topic}: {e}", exc_info=True)
                    return False

        logger.debug(f"Unhandled topic: {topic}")
        return False

    async def _on_caps(self, topic: str, payload: bytes, node_id: str, **kwargs) -> None:
        # handle node capabilities registration
        caps = decode_node_capabilities(payload)
        logger.info(f"Node caps received for {node_id}: {len(caps.enabled_ops)} ops")
        
        async with self.db.acquire() as conn:
            # strictly register sensor node in nodes table
            await register_node(conn, node_id=node_id, status="registered")
            # store capabilities
            await upsert_node_capabilities(
                conn,
                node_id=node_id,
                accel_freqs=caps.accel_freqs,
                gyro_freqs=caps.gyro_freqs,
                enabled_ops=caps.enabled_ops
            )

    async def _on_telemetry(self, topic: str, payload: bytes, node_id: str, sensor_type: str = "sensor", **kwargs) -> None:
        # handle raw vibration telemetry segment
        segment = decode_segment(payload)
        now = datetime.now(timezone.utc)
        
        telem = RawTelemetry(
            node_id=node_id,
            timestamp=now,
            segment_id=segment.id,
            chunk_id=segment.chunk,
            sample_rate=segment.rate,
            emit_reason=int(segment.reason),
            accel_x=segment.data.accel.x,
            accel_y=segment.data.accel.y,
            accel_z=segment.data.accel.z,
            gyro_x=segment.data.gyro.x,
            gyro_y=segment.data.gyro.y,
            gyro_z=segment.data.gyro.z,
            raw_bytes_count=len(payload),
            created_at=now
        )
        
        async with self.db.acquire() as conn:
            # save raw telemetry hypertable record
            await insert_raw_telemetry(conn, telem)
            await update_node_last_seen(conn, node_id)

    async def _on_inference(self, topic: str, payload: bytes, node_id: str, **kwargs) -> None:
        # handle edge inference results
        packet = decode_inference_packet(payload)
        now = datetime.now(timezone.utc)
        
        result = InferenceResult(
            node_id=node_id,
            timestamp=now,
            segment_id=packet.id,
            emit_reason=int(packet.reason),
            router_model_id=packet.r_m_id,
            autoencoder_model_id=packet.ae_id,
            mse=packet.mse,
            anomaly=packet.anom,
            is_recalculated=False,
            created_at=now
        )
        
        async with self.db.acquire() as conn:
            # insert inference result record
            await insert_inference_result(conn, result)
            await update_node_last_seen(conn, node_id)

    async def _on_health(self, topic: str, payload: bytes, node_id: str, **kwargs) -> None:
        # handle heartbeat health telemetry
        health = decode_node_health(payload)
        now = datetime.now(timezone.utc)
        
        health_record = NodeHealth(
            node_id=node_id,
            timestamp=now,
            ram_free=health.ram,
            sd_status=health.sd,
            cached_models=health.cache,
            last_segment_id=health.last,
            status=health.status or "idle",
            ips=health.ips or 0.0
        )
        
        async with self.db.acquire() as conn:
            # insert health record
            await insert_node_health(conn, health_record)
            
            # update capabilities if embedded in health packet
            if health.caps is not None:
                await upsert_node_capabilities(
                    conn,
                    node_id=node_id,
                    accel_freqs=health.caps.accel_freqs,
                    gyro_freqs=health.caps.gyro_freqs,
                    enabled_ops=health.caps.enabled_ops
                )
            await update_node_last_seen(conn, node_id)

    async def _on_alert(self, topic: str, payload: bytes, node_id: str, severity: str = "warning", **kwargs) -> None:
        # handle sensor alert
        alert = decode_node_alert(payload)
        logger.warning(f"Sensor alert from {node_id} ({severity}): code={alert.code} detail={alert.detail}")

    async def publish(self, topic: str, payload: bytes, qos: int = 1, retain: bool = False) -> None:
        # publish binary message over mqtt
        if self._client is None:
            raise RuntimeError("MQTT client is not connected")
        await self._client.publish(topic, payload, qos=qos, retain=retain)

    async def listen(self) -> None:
        # main async listen loop
        self._running = True
        self._stop_event.clear()
        
        while self._running:
            try:
                logger.info(f"Connecting to MQTT broker at {self.broker}:{self.port}...")
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
                    self._client = client
                    logger.info(f"Connected to MQTT broker {self.broker}:{self.port}")
                    
                    # subscribe to required topics
                    for topic in set(self.subscription_topics):
                        await client.subscribe(topic)
                        logger.info(f"Subscribed to topic pattern: {topic}")
                        
                    async for message in client.messages:
                        if not self._running:
                            break
                        # process raw binary payload without utf-8 decode
                        raw_payload = message.payload
                        topic_str = str(message.topic)
                        asyncio.create_task(self.handle_message(topic_str, raw_payload))
                        
            except (aiomqtt.MqttError, OSError) as e:
                logger.warning(f"MQTT connection error: {e}. Reconnecting in 3s...")
                self._client = None
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                logger.info("MQTT listener cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected MQTT listener error: {e}", exc_info=True)
                await asyncio.sleep(1.0)
                
        self._client = None
        self._running = False

    def stop(self) -> None:
        # signal consumer shutdown
        self._running = False
        self._stop_event.set()
