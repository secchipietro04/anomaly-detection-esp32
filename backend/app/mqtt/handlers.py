# pure business logic mqtt handlers
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Callable, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import get_settings
from app.mqtt.router import MQTTRouter
from app.mqtt.publisher import MQTTPublisher
from app.cbor.codec import from_cbor, to_cbor, loads_cbor
from app.cbor.schemas import (
    Segment, InferencePacket, NodeCapabilities, NodeHealthInfo,
    RuntimeConfig, EnsembleConfig, RouteEntry,
    AutoencoderModelPackage, RouterModelPackage, MemoryModelPackage, ModelType
)
from app.database.models import (
    NodeModel, NodeCapabilitiesModel, NodeHealthModel,
    RawTelemetryModel, InferenceResultModel, ModelPackageModel, EnsembleConfigModel,
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
    # register callback fired when threshold reached
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
@router.subscribe("v1/{node_id}/telemetry")
async def handle_sensor_data(node_id: str, payload: bytes, session: AsyncSession, publisher: MQTTPublisher):
    # decode telemetry chunk and persist to timescaledb
    try:
        raw_dict = loads_cbor(payload)
        if isinstance(raw_dict, dict):
            seg_id = raw_dict.get("id", 0)
            rate = float(raw_dict.get("rate", 3840.0))
            reason = int(raw_dict.get("reason", 1))
            chunk = int(raw_dict.get("chunk", 0))
            raw_data = raw_dict.get("data", {})
        else:
            seg = from_cbor(payload, Segment)
            seg_id = seg.id
            rate = seg.rate
            reason = int(seg.reason)
            chunk = seg.chunk
            raw_data = seg.data
    except Exception as e:
        logger.warning(f"Failed to decode sensor segment from {node_id}: {str(e)[:100]}")
        return

    # update node last seen
    node = await session.get(NodeModel, node_id)
    if node:
        node.last_seen = datetime.now(timezone.utc)

    # extract accel and gyro lists robustly
    ax, ay, az = [], [], []
    gx, gy, gz = [], [], []

    if isinstance(raw_data, dict):
        if "accel" in raw_data and isinstance(raw_data["accel"], dict):
            ax = list(raw_data["accel"].get("x", []))
            ay = list(raw_data["accel"].get("y", []))
            az = list(raw_data["accel"].get("z", []))
        if "gyro" in raw_data and isinstance(raw_data["gyro"], dict):
            gx = list(raw_data["gyro"].get("x", []))
            gy = list(raw_data["gyro"].get("y", []))
            gz = list(raw_data["gyro"].get("z", []))

        if not ax and not gx:
            for k, v in raw_data.items():
                if isinstance(v, (list, tuple)) and len(v) > 0:
                    if len(ax) == 0: ax = list(v)
                    elif len(ay) == 0: ay = list(v)
                    elif len(az) == 0: az = list(v)
                    elif len(gx) == 0: gx = list(v)
                    elif len(gy) == 0: gy = list(v)
                    elif len(gz) == 0: gz = list(v)
                if isinstance(k, (list, tuple)) and len(k) > 0:
                    if len(ax) == 0: ax = list(k)
                    elif len(ay) == 0: ay = list(k)
                    elif len(az) == 0: az = list(k)
                    elif len(gx) == 0: gx = list(k)
                    elif len(gy) == 0: gy = list(k)
                    elif len(gz) == 0: gz = list(k)

    raw_bytes = len(payload)
    record = RawTelemetryModel(
        node_id=node_id,
        timestamp=datetime.now(timezone.utc),
        segment_id=seg_id,
        chunk_id=chunk,
        sample_rate=rate,
        emit_reason=reason,
        accel_x=ax,
        accel_y=ay,
        accel_z=az,
        gyro_x=gx,
        gyro_y=gy,
        gyro_z=gz,
        raw_bytes_count=raw_bytes
    )
    session.add(record)
    await session.commit()
    logger.info(f"Persisted segment {seg_id} chunk {chunk} from {node_id} ({raw_bytes} bytes, samples: {len(ax)})")

    # in-memory volume accumulator tracking
    current_vol = await _get_or_init_volume(session, node_id)
    current_vol += raw_bytes
    _volume_accumulator[node_id] = current_vol

    # trigger nas if threshold exceeded
    settings = get_settings()
    if current_vol >= settings.nas_data_threshold_bytes and _on_nas_trigger is not None:
        logger.info(f"Node {node_id} volume {current_vol} >= {settings.nas_data_threshold_bytes} bytes threshold. Triggering NAS training!")
        try:
            res = _on_nas_trigger(node_id, current_vol)
            if hasattr(res, "__await__"):
                await res
        except Exception as e:
            logger.error(f"Error in NAS trigger callback: {e}")

@router.subscribe("v1/{node_id}/info/caps")
async def handle_node_capabilities(node_id: str, payload: bytes, session: AsyncSession, publisher: MQTTPublisher):
    # strict registration on capabilities reception
    if not payload:
        return
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
    if not payload:
        return
    try:
        h = from_cbor(payload, NodeHealthInfo)
    except Exception as e:
        logger.warning(f"Failed to decode health info from {node_id}: {e}")
        return

    # ensure node is registered before adding child records
    await db_register_node(session, node_id)

    health = NodeHealthModel(
        node_id=node_id,
        timestamp=datetime.now(timezone.utc),
        ram_free=h.ram,
        sd_status=int(h.sd),
        cached_models=h.cache,
        last_segment_id=h.last,
        status=h.status or "idle",
        ips=h.ips or 0.0
    )
    session.add(health)
    await session.commit()

@router.subscribe("v1/{node_id}/inference")
async def handle_inference_packet(node_id: str, payload: bytes, session: AsyncSession, publisher: MQTTPublisher):
    # save edge inference packet
    try:
        inf = from_cbor(payload, InferencePacket)
    except Exception as e:
        logger.warning(f"Failed to decode inference packet from {node_id}: {e}")
        return

    # ensure node is registered before adding child records
    await db_register_node(session, node_id)

    res = InferenceResultModel(
        node_id=node_id,
        timestamp=datetime.now(timezone.utc),
        segment_id=inf.id,
        emit_reason=int(inf.reason),
        router_model_id=inf.r_m_id,
        autoencoder_model_id=inf.ae_id,
        mse=float(inf.mse),
        anomaly=bool(inf.anom),
        inference_time_ms=inf.time,
        is_recalculated=False
    )
    session.add(res)
    await session.commit()

@router.subscribe("v1/{node_id}/ensemble/fetch")
async def handle_ensemble_fetch_request(node_id: str, payload: bytes, session: AsyncSession, publisher: MQTTPublisher):
    # on-demand ensemble routing table request from sensor on reconnect
    stmt = select(EnsembleConfigModel).where(EnsembleConfigModel.node_id == node_id).order_by(EnsembleConfigModel.deployed_at.desc()).limit(1)
    res = await session.execute(stmt)
    ens = res.scalar_one_or_none()
    if not ens:
        logger.warning(f"Sensor {node_id} requested ensemble table but none found")
        return

    routes = [RouteEntry(out_ix=r["out_ix"], m_id=r["m_id"]) for r in (ens.routes or [])]
    cfg = EnsembleConfig(
        warmup=ens.warmup,
        mem_id=ens.memory_model_id,
        r_m_id=ens.router_model_id or 0,
        routes=routes
    )
    await publisher.publish_ensemble(node_id, to_cbor(cfg), retain=True)
    logger.info(f"Served on-demand ensemble table to sensor {node_id}")

    # Also serve active router model package to node
    if ens.router_model_id:
        r_model = await session.get(ModelPackageModel, ens.router_model_id)
        if r_model:
            r_cfg = dict(r_model.config or {})
            r_cfg["m_id"] = r_model.id
            r_cfg["data"] = r_model.tflite_binary
            r_cfg["type"] = r_model.model_type
            if r_model.tag is not None:
                r_cfg["tag"] = r_model.tag
            r_pkg = RouterModelPackage.model_validate(r_cfg)
            await publisher.publish_model(
                node_id=node_id,
                model_type="router",
                model_id=r_model.id,
                payload=to_cbor(r_pkg),
                qos=1,
                retain=True
            )
            logger.info(f"Served active router model {r_model.id} to sensor {node_id}")

@router.subscribe("v1/{node_id}/config/fetch")
async def handle_config_fetch_request(node_id: str, payload: bytes, session: AsyncSession, publisher: MQTTPublisher):
    # on-demand runtime config request from sensor on reconnect
    node = await session.get(NodeModel, node_id)
    if not node or not node.current_config:
        logger.warning(f"Sensor {node_id} requested config but none registered")
        return

    try:
        cfg = RuntimeConfig.model_validate(node.current_config)
        await publisher.publish_config(node_id, to_cbor(cfg))
        logger.info(f"Served on-demand runtime config to sensor {node_id}")
    except Exception as e:
        logger.error(f"Error serving config to {node_id}: {e}")

@router.subscribe("v1/{node_id}/models/fetch/{model_type}/{model_id}")
async def handle_model_fetch_request(node_id: str, model_type: str, model_id: str, session: AsyncSession, publisher: MQTTPublisher):
    # on-demand model download request from sensor (reconstructs typed CBOR model package)
    try:
        m_id = int(model_id)
    except ValueError:
        return

    model = await session.get(ModelPackageModel, m_id)
    if not model:
        logger.warning(f"Sensor {node_id} requested non-existent model {m_id}")
        return

    cfg = dict(model.config or {})
    cfg["m_id"] = model.id
    cfg["data"] = model.tflite_binary
    cfg["type"] = model.model_type
    if model.tag is not None:
        cfg["tag"] = model.tag

    try:
        if model.model_type == 1:
            pkg = AutoencoderModelPackage.model_validate(cfg)
        elif model.model_type == 2:
            if "class_count" in cfg and "class" not in cfg:
                cfg["class"] = cfg["class_count"]
            pkg = RouterModelPackage.model_validate(cfg)
        elif model.model_type == 3:
            pkg = MemoryModelPackage.model_validate(cfg)
        else:
            logger.warning(f"Unknown model type {model.model_type} for model {m_id}")
            return

        # publish requested CBOR model package back to sensor
        await publisher.publish_model(
            node_id=node_id,
            model_type=model_type,
            model_id=m_id,
            payload=to_cbor(pkg),
            qos=1,
            retain=True
        )
        logger.info(f"Served on-demand CBOR model {m_id} ({model_type}) to sensor {node_id}")
    except Exception as e:
        logger.error(f"Failed to encode and serve model {m_id}: {e}")
