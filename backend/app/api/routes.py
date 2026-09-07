# fastapi rest routes for grafana and control plane
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database.session import get_session_dependency
from app.database.models import (
    NodeModel, NodeHealthModel, NodeCapabilitiesModel,
    RawTelemetryModel, InferenceResultModel, ModelPackageModel, EnsembleConfigModel
)
from app.services.config_service import sync_sensor_config, ConfigValidationError, NodeNotFoundError
from app.mqtt.publisher import MQTTPublisher

logger = logging.getLogger("api.routes")
router = APIRouter()

class ConfigUpdateRequest(BaseModel):
    mode: int = Field(ge=1, le=4)
    rate: float = Field(gt=0)
    beat: int = Field(gt=0)
    batch: int = Field(gt=0)
    sd_en: bool = False
    cad: Optional[int] = Field(default=None, gt=0)

class RecalculateRequest(BaseModel):
    node_id: str
    segment_id: int
    model_id: Optional[int] = None
    anomaly_threshold: float = Field(default=0.1, gt=0)
    tsteps: int = Field(default=8, ge=1, le=64)

class RecalculateResponse(BaseModel):
    node_id: str
    segment_id: int
    mse: float
    anomaly: bool
    is_recalculated: bool = True

@router.get("/sensors")
async def list_sensors(session: AsyncSession = Depends(get_session_dependency)):
    # list all registered sensor nodes
    stmt = select(NodeModel).order_by(NodeModel.registered_at.desc())
    res = await session.execute(stmt)
    nodes = res.scalars().all()
    return [{"node_id": n.node_id, "name": n.name, "status": n.status, "last_seen": n.last_seen, "config": n.current_config} for n in nodes]

@router.get("/sensors/{node_id}")
async def get_sensor(node_id: str, session: AsyncSession = Depends(get_session_dependency)):
    # get single sensor info
    node = await session.get(NodeModel, node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return {"node_id": node.node_id, "name": node.name, "status": node.status, "last_seen": node.last_seen, "config": node.current_config}

@router.get("/sensors/{node_id}/health")
async def get_sensor_health(node_id: str, session: AsyncSession = Depends(get_session_dependency)):
    # get latest health record
    stmt = select(NodeHealthModel).where(NodeHealthModel.node_id == node_id).order_by(NodeHealthModel.timestamp.desc()).limit(1)
    res = await session.execute(stmt)
    health = res.scalar_one_or_none()
    if not health:
        raise HTTPException(status_code=404, detail="No health data found")
    return {
        "node_id": health.node_id,
        "timestamp": health.timestamp,
        "ram_free": health.ram_free,
        "sd_status": health.sd_status,
        "cached_models": health.cached_models,
        "last_segment_id": health.last_segment_id,
        "status": health.status,
        "ips": health.ips
    }

@router.get("/sensors/{node_id}/volume")
async def get_sensor_volume(node_id: str, session: AsyncSession = Depends(get_session_dependency)):
    # get accumulated raw telemetry bytes and NAS threshold progress
    from app.config import get_settings
    settings = get_settings()
    stmt = select(
        func.sum(RawTelemetryModel.raw_bytes_count).label("total_bytes"),
        func.count().label("chunks_count"),
        func.max(RawTelemetryModel.segment_id).label("last_segment_id")
    ).where(RawTelemetryModel.node_id == node_id)
    res = await session.execute(stmt)
    row = res.one_or_none()
    
    total_bytes = int(row.total_bytes or 0) if row else 0
    chunks = int(row.chunks_count or 0) if row else 0
    last_seg = int(row.last_segment_id or 0) if row else 0
    
    threshold = settings.nas_data_threshold_bytes
    progress_pct = round(min(100.0, (total_bytes / threshold) * 100.0), 2) if threshold > 0 else 100.0
    
    return {
        "node_id": node_id,
        "accumulated_bytes": total_bytes,
        "threshold_bytes": threshold,
        "progress_percent": progress_pct,
        "chunks_count": chunks,
        "last_segment_id": last_seg,
        "bytes_remaining": max(0, threshold - total_bytes),
        "ready_for_nas": total_bytes >= threshold
    }

@router.get("/sensors/{node_id}/models")
async def get_sensor_models(node_id: str, session: AsyncSession = Depends(get_session_dependency)):
    # get supported and deployed models for grafana
    caps = await session.get(NodeCapabilitiesModel, node_id)
    enabled_ops = set(caps.enabled_ops) if caps else set()

    # check supported arch tags
    from app.nas.architectures.autoencoders import ALL_AUTOENCODER_ARCHS
    from app.nas.architectures.routers import ALL_ROUTER_ARCHS
    from app.nas.architectures.memory import ALL_MEMORY_ARCHS

    supported_archs = []
    for a in ALL_AUTOENCODER_ARCHS + ALL_ROUTER_ARCHS + ALL_MEMORY_ARCHS:
        supported_archs.append({
            "name": a.__class__.__name__,
            "tag": int(a.tag),
            "supported": a.is_supported(enabled_ops),
            "required_ops": list(a.required_ops)
        })

    # active ensemble
    stmt_ens = select(EnsembleConfigModel).where(EnsembleConfigModel.node_id == node_id).order_by(EnsembleConfigModel.deployed_at.desc()).limit(1)
    res_ens = await session.execute(stmt_ens)
    active_ensemble = res_ens.scalar_one_or_none()

    return {
        "node_id": node_id,
        "enabled_ops": list(enabled_ops),
        "architectures": supported_archs,
        "active_ensemble": {
            "router_model_id": active_ensemble.router_model_id if active_ensemble else None,
            "memory_model_id": active_ensemble.memory_model_id if active_ensemble else None,
            "routes": active_ensemble.routes if active_ensemble else [],
            "total_size_bytes": active_ensemble.total_size_bytes if active_ensemble else 0
        } if active_ensemble else None
    }

@router.post("/sensors/{node_id}/config")
async def update_sensor_config(
    node_id: str,
    req: ConfigUpdateRequest,
    session: AsyncSession = Depends(get_session_dependency)
):
    # deploy runtime config via mqtt
    publisher = MQTTPublisher()
    try:
        cfg = await sync_sensor_config(
            session=session,
            publisher=publisher,
            node_id=node_id,
            config=req.model_dump(exclude_none=True)
        )
        return {"status": "deployed", "node_id": node_id, "config": cfg.model_dump(exclude_none=True)}
    except NodeNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ConfigValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from app.api.recalculator import (
    recalculate_segment,
    test_noise_resilience,
    test_cross_submodel_specificity
)

class CrossSpecificityTestRequest(BaseModel):
    num_samples_per_mode: int = Field(default=50, ge=5, le=200)
    tsteps: int = Field(default=8, ge=1, le=64)

class ResilienceTestRequest(BaseModel):
    num_sample_segments: int = Field(default=64, ge=8, le=256)
    num_noise_levels: int = Field(default=64, ge=8, le=128)
    min_power_2: float = Field(default=-10.0)
    max_power_2: float = Field(default=0.0)
    tsteps: int = Field(default=8, ge=1, le=64)

class RecalculateRangeRequest(BaseModel):
    node_id: str
    from_time_ms: Optional[int] = None
    to_time_ms: Optional[int] = None
    from_time_str: Optional[str] = None
    to_time_str: Optional[str] = None
    model_id: Optional[int] = None
    anomaly_threshold: float = Field(default=0.15, gt=0)
    tsteps: int = Field(default=8, ge=1, le=64)

@router.post("/recalculate", response_model=RecalculateResponse)
async def recalculate(
    req: RecalculateRequest,
    session: AsyncSession = Depends(get_session_dependency)
):
    # run local tflite inference on active ensemble or explicit model
    try:
        mse, anomaly = await recalculate_segment(
            session=session,
            node_id=req.node_id,
            segment_id=req.segment_id,
            autoencoder_model_id=req.model_id,
            anomaly_threshold=req.anomaly_threshold,
            tsteps=req.tsteps
        )
        return RecalculateResponse(
            node_id=req.node_id,
            segment_id=req.segment_id,
            mse=mse,
            anomaly=anomaly
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Recalculate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recalculate/range")
async def recalculate_range(
    req: RecalculateRangeRequest,
    session: AsyncSession = Depends(get_session_dependency)
):
    # run local tflite inference over entire specified time window
    from app.api.recalculator import recalculate_time_range
    try:
        return await recalculate_time_range(
            session=session,
            node_id=req.node_id,
            from_time_ms=req.from_time_ms,
            to_time_ms=req.to_time_ms,
            from_time_str=req.from_time_str,
            to_time_str=req.to_time_str,
            autoencoder_model_id=req.model_id,
            anomaly_threshold=req.anomaly_threshold,
            tsteps=req.tsteps
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Recalculate range failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class FlagRangeRequest(BaseModel):
    from_time_ms: Optional[int] = None
    to_time_ms: Optional[int] = None
    from_time_str: Optional[str] = None
    to_time_str: Optional[str] = None
    is_anomaly: bool = True
    exclude_from_training: bool = True

@router.post("/sensors/{node_id}/flag-range")
async def flag_time_range(
    node_id: str,
    req: FlagRangeRequest,
    session: AsyncSession = Depends(get_session_dependency)
):
    # manually tag segments in active time range as anomaly or nominal
    from app.api.recalculator import flag_segments_in_range
    try:
        return await flag_segments_in_range(
            session=session,
            node_id=node_id,
            from_time_ms=req.from_time_ms,
            to_time_ms=req.to_time_ms,
            from_time_str=req.from_time_str,
            to_time_str=req.to_time_str,
            is_anomaly=req.is_anomaly,
            exclude_from_training=req.exclude_from_training
        )
    except Exception as e:
        logger.exception(f"Flag range failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sensors/{node_id}/cmd/{command}")
async def send_sensor_command(
    node_id: str,
    command: str,
    session: AsyncSession = Depends(get_session_dependency)
):
    # send hardware control command to sensor: reboot, updt_status, dump, dump_i
    allowed = {"reboot", "updt_status", "dump", "dump_i"}
    if command not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown command '{command}'. Allowed: {list(allowed)}")

    publisher = MQTTPublisher()
    try:
        await publisher.publish_command(node_id, command)
        return {"status": "ok", "node_id": node_id, "command": command}
    except Exception as e:
        logger.exception(f"Failed to publish command {command} to {node_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sensors/{node_id}/resilience-test")
async def benchmark_noise_resilience(
    node_id: str,
    req: ResilienceTestRequest = ResilienceTestRequest(),
    session: AsyncSession = Depends(get_session_dependency)
):
    # executes base-2 noise resilience spectrogram benchmark (2^-10 to 2^0) across frequency bins
    try:
        report = await test_noise_resilience(
            session=session,
            node_id=node_id,
            num_sample_segments=req.num_sample_segments,
            num_noise_levels=req.num_noise_levels,
            min_power_2=req.min_power_2,
            max_power_2=req.max_power_2,
            tsteps=req.tsteps
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Resilience test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sensors/{node_id}/cross-specificity-test")
async def benchmark_cross_specificity(
    node_id: str,
    req: CrossSpecificityTestRequest = CrossSpecificityTestRequest(),
    session: AsyncSession = Depends(get_session_dependency)
):
    # evaluates cross-submodel specificity matrix by testing autoencoders on foreign cluster data
    try:
        report = await test_cross_submodel_specificity(
            session=session,
            node_id=node_id,
            num_samples_per_mode=req.num_samples_per_mode,
            tsteps=req.tsteps
        )
        return report
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Cross-specificity test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sensors/{node_id}/stft-spectrogram")
async def get_stft_spectrogram(
    node_id: str,
    segment_id: Optional[int] = None,
    from_time_ms: Optional[int] = None,
    to_time_ms: Optional[int] = None,
    session: AsyncSession = Depends(get_session_dependency)
):
    # returns real 2D STFT spectrogram across the requested time window or segment
    from app.api.recalculator import get_segment_stft_spectrogram
    try:
        return await get_segment_stft_spectrogram(
            session=session,
            node_id=node_id,
            segment_id=segment_id,
            from_time_ms=from_time_ms,
            to_time_ms=to_time_ms
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Spectrogram failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sensors/{node_id}/noise-spectrogram")
async def get_noise_spectrogram(
    node_id: str,
    session: AsyncSession = Depends(get_session_dependency)
):
    # returns 2D noise resilience sensitivity matrix across base-2 noise levels
    try:
        return await test_noise_resilience(session=session, node_id=node_id, num_sample_segments=32, num_noise_levels=32)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Noise spectrogram failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


