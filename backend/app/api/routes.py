# fastapi rest routes for grafana and control plane
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.session import get_session_dependency
from app.database.models import (
    NodeModel, NodeHealthModel, NodeCapabilitiesModel,
    InferenceResultModel, ModelPackageModel, EnsembleConfigModel
)
from app.services.config_service import sync_sensor_config, ConfigValidationError, NodeNotFoundError
from app.mqtt.publisher import MQTTPublisher
from app.api.recalculator import recalculate_segment

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
    model_id: int
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

@router.post("/recalculate", response_model=RecalculateResponse)
async def recalculate(
    req: RecalculateRequest,
    session: AsyncSession = Depends(get_session_dependency)
):
    # run local tflite inference
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
