# pure business logic mqtt handlers
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.mqtt.router import MQTTRouter
from app.mqtt.publisher import MQTTPublisher
from app.cbor.codec import from_cbor, loads_cbor
from app.cbor.schemas import (
    Segment, InferencePacket, NodeCapabilities, NodeHealthInfo
)
from app.database.models import (
    NodeModel, NodeCapabilitiesModel, NodeHealthModel,
    RawTelemetryModel, InferenceResultModel, ModelPackageModel,
    db_register_node, db_upsert_capabilities, db_get_untrained_bytes_for_node
)

logger = logging.getLogger("mqtt_handlers")

# global router instance
router = MQTTRouter(prefix="v1")

# in-memory volume accumulator: node_id -> accumulated bytes count
_volume_accumulator: Dict[str, int] = {}
_initialized_nodes: set = set()
_on_nas_trigger: Optional[Callable[[str, int], Any]] = None

def set_nas_trigger_callback(callback: Callable[[str, int], Any]) -> None:
    # register callback fired when 1MB threshold reached
    global _on_nas_trigger
    _on_nas_trigger = callback

async def _get_or_init_volume(session: AsyncSession, node_id: str) -> int:
    # fast indexed startup lookup on first message per node
    if node_id not in _initialized_nodes:
        bytes_in_db = await db_get_untrained_bytes_for_node(session, node_id)
        _volume_accumulator[node_id] = bytes_in_db
        _initialized_nodes.add(node_id)
        logger.info(f"Initialized in-memory volume for {node_id}: {bytes_in_db} bytes from DB")
    return _volume_accumulator.get(node_id, 0)

def reset_node_volume(node_id: str) -> None:
    # reset in-memory accumulator after nas checkpoint
    _volume_accumulator[node_id] = 0

@router.subscribe("v1/{node_id}/data/sensor")
async def handle_sensor_data(node_id: str, payload: bytes, session: AsyncSession, publisher: MQTTPublisher):
    # decode telemetry chunk and persist to timescaledb
    try:
        seg = from_cbor(payload, Segment)
    except Exception as e:
        logger.warning(f"Failed to decode sensor segment from {node_id}: {e}")
        return

    # update node last seen
    node = await session.get(NodeModel, node_id)
    if node:
        node.last_seen = datetime.now(timezone.utc)

    # insert raw telemetry
    raw_bytes = len(payload)
    record = RawTelemetryModel(
        node_id=node_id,
        timestamp=datetime.now(timezone.utc),
        segment_id=seg.id,
        chunk_id=seg.chunk,
        sample_rate=seg.rate,
        emit_reason=seg.reason,
        accel_x=seg.ax,
        accel_y=seg.ay,
        accel_z=seg.az,
        gyro_x=seg.gx,
        gyro_y=seg.gy,
        gyro_z=seg.gz,
        raw_bytes_count=raw_bytes
    )
    session.add(record)
    await session.commit()

    # in-memory volume accumulator tracking
    current_vol = await _get_or_init_volume(session, node_id)
    current_vol += raw_bytes
    _volume_accumulator[node_id] = current_vol

    # trigger nas if threshold exceeded (1MB = 1048576 bytes)
    if current_vol >= 1048576 and _on_nas_trigger is not None:
        logger.info(f"Node {node_id} volume {current_vol} >= 1MB threshold. Triggering NAS training!")
        try:
            res = _on_nas_trigger(node_id, current_vol)
            if hasattr(res, "__await__"):
                await res
        except Exception as e:
            logger.error(f"Error in NAS trigger callback: {e}")

@router.subscribe("v1/{node_id}/info/caps")
async def handle_node_capabilities(node_id: str, payload: bytes, session: AsyncSession, publisher: MQTTPublisher):
    # strict registration on capabilities reception
    try:
        caps = from_cbor(payload, NodeCapabilities)
    except Exception as e:
        logger.warning(f"Failed to decode capabilities from {node_id}: {e}")
        return

    await db_register_node(session, node_id)
    await db_upsert_capabilities(
        session,
        node_id=node_id,
        accel_freqs=caps.accel_freqs,
        gyro_freqs=caps.gyro_freqs,
        enabled_ops=caps.enabled_ops
    )
    logger.info(f"Registered node {node_id} with {len(caps.enabled_ops)} enabled ops")

@router.subscribe("v1/{node_id}/info/health")
async def handle_node_health(node_id: str, payload: bytes, session: AsyncSession, publisher: MQTTPublisher):
    # save health heartbeat
    try:
        h = from_cbor(payload, NodeHealthInfo)
    except Exception as e:
        logger.warning(f"Failed to decode health info from {node_id}: {e}")
        return

    health = NodeHealthModel(
        node_id=node_id,
        timestamp=datetime.now(timezone.utc),
        ram_free=h.ram,
        sd_status=1 if h.sd_ok else 0,
        cached_models=h.cache,
        last_segment_id=h.seg_id,
        status=h.stat,
        ips=h.ips
    )
    session.add(health)
    
    # update node last seen
    node = await session.get(NodeModel, node_id)
    if node:
        node.last_seen = datetime.now(timezone.utc)
        
    await session.commit()

@router.subscribe("v1/{node_id}/inference")
async def handle_inference_packet(node_id: str, payload: bytes, session: AsyncSession, publisher: MQTTPublisher):
    # save edge inference packet
    try:
        inf = from_cbor(payload, InferencePacket)
    except Exception as e:
        logger.warning(f"Failed to decode inference packet from {node_id}: {e}")
        return

    res = InferenceResultModel(
        node_id=node_id,
        timestamp=datetime.now(timezone.utc),
        segment_id=inf.seg_id,
        emit_reason=inf.reason,
        router_model_id=inf.r_m_id,
        autoencoder_model_id=inf.ae_m_id,
        mse=inf.mse,
        anomaly=inf.anomaly,
        is_recalculated=False
    )
    session.add(res)
    await session.commit()

@router.subscribe("v1/{node_id}/models/fetch/{model_type}/{model_id}")
async def handle_model_fetch_request(node_id: str, model_type: str, model_id: str, session: AsyncSession, publisher: MQTTPublisher):
    # on-demand model download request from sensor (e.g. no-SD mode)
    try:
        m_id = int(model_id)
    except ValueError:
        return

    model = await session.get(ModelPackageModel, m_id)
    if not model:
        logger.warning(f"Sensor {node_id} requested non-existent model {m_id}")
        return

    # publish requested model binary back to sensor
    await publisher.publish_model(
        node_id=node_id,
        model_type=model_type,
        model_id=m_id,
        payload=model.tflite_binary,
        qos=1,
        retain=False
    )
    logger.info(f"Served on-demand model {m_id} to sensor {node_id}")
