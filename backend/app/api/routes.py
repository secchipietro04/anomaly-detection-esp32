# fastapi routes - sensors, config, health, recalculate
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.database.models import (
    get_all_nodes,
    get_node,
    get_latest_node_health,
    get_inference_results,
    Node,
    NodeHealth,
    InferenceResult,
)
from app.database.session import db_manager
from app.services.config_service import (
    ConfigService,
    ConfigValidationError,
    NodeNotFoundError,
    CapabilitiesNotFoundError,
    ConfigDeployError,
)
from app.api.recalculator import recalculate_segment

logger = logging.getLogger("api.routes")

router = APIRouter()


# --- request / response schemas ---

class ConfigUpdateRequest(BaseModel):
    # runtime config fields sent by grafana buttons or direct api call
    mode: int = Field(..., ge=1, le=4, description="StreamMode 1-4")
    rate: float = Field(..., gt=0)
    beat: int = Field(..., gt=0)
    batch: int = Field(..., gt=0)
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


# --- sensor endpoints ---

@router.get("/sensors", response_model=List[Dict[str, Any]])
async def list_sensors():
    # return all registered sensors
    async with db_manager.acquire() as conn:
        nodes = await get_all_nodes(conn)
    return [n.model_dump() for n in nodes]


@router.get("/sensors/{node_id}", response_model=Dict[str, Any])
async def get_sensor(node_id: str):
    # return single sensor info
    async with db_manager.acquire() as conn:
        node = await get_node(conn, node_id)
    if not node:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"node '{node_id}' not found")
    return node.model_dump()


@router.get("/sensors/{node_id}/health", response_model=Dict[str, Any])
async def get_sensor_health(node_id: str):
    # latest health report
    async with db_manager.acquire() as conn:
        health = await get_latest_node_health(conn, node_id)
    if not health:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no health data")
    return health.model_dump()


@router.get("/sensors/{node_id}/inference", response_model=List[Dict[str, Any]])
async def get_sensor_inference(node_id: str, limit: int = 100, offset: int = 0):
    # paginated inference history
    async with db_manager.acquire() as conn:
        results = await get_inference_results(conn, node_id, limit=limit, offset=offset)
    return [r.model_dump() for r in results]


# --- config endpoint ---

@router.post("/sensors/{node_id}/config", response_model=Dict[str, Any])
async def update_sensor_config(node_id: str, req: ConfigUpdateRequest):
    # validate and push RuntimeConfig to sensor via mqtt downlink
    svc = ConfigService()
    try:
        cfg = await svc.sync_config(
            node_id=node_id,
            config=req.model_dump(exclude_none=True),
            require_capabilities=False,
        )
        return {"status": "deployed", "node_id": node_id, "config": cfg.model_dump(exclude_none=True)}
    except NodeNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ConfigValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except (CapabilitiesNotFoundError, ConfigDeployError) as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# --- recalculate endpoint ---

@router.post("/recalculate", response_model=RecalculateResponse)
async def recalculate(req: RecalculateRequest):
    # run local tflite inference on stored raw segment and update DB
    try:
        mse, anomaly = await recalculate_segment(
            node_id=req.node_id,
            segment_id=req.segment_id,
            autoencoder_model_id=req.model_id,
            anomaly_threshold=req.anomaly_threshold,
            tsteps=req.tsteps,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.exception(f"recalculate failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return RecalculateResponse(
        node_id=req.node_id,
        segment_id=req.segment_id,
        mse=mse,
        anomaly=anomaly,
    )
