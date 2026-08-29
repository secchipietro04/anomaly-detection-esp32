# database models and repository helpers
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import asyncpg

# pydantic data models

class Node(BaseModel):
    node_id: str
    name: Optional[str] = None
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "registered"

class NodeCapabilities(BaseModel):
    node_id: str
    accel_freqs: List[float] = Field(default_factory=list)
    gyro_freqs: List[float] = Field(default_factory=list)
    enabled_ops: List[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class NodeHealth(BaseModel):
    id: Optional[int] = None
    node_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ram_free: int = 0
    sd_status: int = 0
    cached_models: List[int] = Field(default_factory=list)
    last_segment_id: int = 0
    status: str = "idle"
    ips: float = 0.0

class ModelPackage(BaseModel):
    id: int
    tag: Optional[int] = None
    model_type: int # 1=autoencoder, 2=router, 3=memory
    tflite_binary: bytes
    size_bytes: int
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class EnsembleConfig(BaseModel):
    id: int
    node_id: Optional[str] = None
    router_model_id: Optional[int] = None
    memory_model_id: Optional[int] = None
    warmup: int = 0
    routes: List[Dict[str, Any]] = Field(default_factory=list)
    total_size_bytes: int = 0
    deployed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class RawTelemetry(BaseModel):
    id: Optional[int] = None
    node_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    segment_id: int
    chunk_id: int = 1
    sample_rate: float
    emit_reason: int
    accel_x: List[float] = Field(default_factory=list)
    accel_y: List[float] = Field(default_factory=list)
    accel_z: List[float] = Field(default_factory=list)
    gyro_x: List[float] = Field(default_factory=list)
    gyro_y: List[float] = Field(default_factory=list)
    gyro_z: List[float] = Field(default_factory=list)
    raw_bytes_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class InferenceResult(BaseModel):
    id: Optional[int] = None
    node_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    segment_id: int
    emit_reason: int
    router_model_id: Optional[int] = None
    autoencoder_model_id: Optional[int] = None
    mse: float
    anomaly: bool
    is_recalculated: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

# database query and persistence helpers

async def register_node(
    conn: asyncpg.Connection,
    node_id: str,
    name: Optional[str] = None,
    status: str = "registered"
) -> Node:
    # insert or update node record on registration
    row = await conn.fetchrow(
        """
        INSERT INTO nodes (node_id, name, registered_at, last_seen, status)
        VALUES ($1, $2, NOW(), NOW(), $3)
        ON CONFLICT (node_id) DO UPDATE
        SET last_seen = NOW(),
            status = EXCLUDED.status,
            name = COALESCE(EXCLUDED.name, nodes.name)
        RETURNING node_id, name, registered_at, last_seen, status;
        """,
        node_id, name, status
    )
    return Node(
        node_id=row["node_id"],
        name=row["name"],
        registered_at=row["registered_at"],
        last_seen=row["last_seen"],
        status=row["status"]
    )

async def get_node(conn: asyncpg.Connection, node_id: str) -> Optional[Node]:
    # fetch node by id
    row = await conn.fetchrow(
        "SELECT node_id, name, registered_at, last_seen, status FROM nodes WHERE node_id = $1;",
        node_id
    )
    if not row:
        return None
    return Node(
        node_id=row["node_id"],
        name=row["name"],
        registered_at=row["registered_at"],
        last_seen=row["last_seen"],
        status=row["status"]
    )

async def get_all_nodes(conn: asyncpg.Connection) -> List[Node]:
    # List all nodes
    rows = await conn.fetch("SELECT node_id, name, registered_at, last_seen, status FROM nodes ORDER BY registered_at DESC;")
    return [
        Node(
            node_id=r["node_id"],
            name=r["name"],
            registered_at=r["registered_at"],
            last_seen=r["last_seen"],
            status=r["status"]
        )
        for r in rows
    ]

async def update_node_last_seen(conn: asyncpg.Connection, node_id: str) -> None:
    # update timestamp when node sends telemetry
    await conn.execute(
        "UPDATE nodes SET last_seen = NOW() WHERE node_id = $1;",
        node_id
    )

async def upsert_node_capabilities(
    conn: asyncpg.Connection,
    node_id: str,
    accel_freqs: List[float],
    gyro_freqs: List[float],
    enabled_ops: List[str]
) -> NodeCapabilities:
    # Store or update capabilities
    row = await conn.fetchrow(
        """
        INSERT INTO node_capabilities (node_id, accel_freqs, gyro_freqs, enabled_ops, updated_at)
        VALUES ($1, $2, $3, $4, NOW())
        ON CONFLICT (node_id) DO UPDATE
        SET accel_freqs = EXCLUDED.accel_freqs,
            gyro_freqs = EXCLUDED.gyro_freqs,
            enabled_ops = EXCLUDED.enabled_ops,
            updated_at = NOW()
        RETURNING node_id, accel_freqs, gyro_freqs, enabled_ops, updated_at;
        """,
        node_id, accel_freqs, gyro_freqs, enabled_ops
    )
    return NodeCapabilities(
        node_id=row["node_id"],
        accel_freqs=list(row["accel_freqs"]),
        gyro_freqs=list(row["gyro_freqs"]),
        enabled_ops=list(row["enabled_ops"]),
        updated_at=row["updated_at"]
    )

async def get_node_capabilities(conn: asyncpg.Connection, node_id: str) -> Optional[NodeCapabilities]:
    # get capabilities for node
    row = await conn.fetchrow(
        "SELECT node_id, accel_freqs, gyro_freqs, enabled_ops, updated_at FROM node_capabilities WHERE node_id = $1;",
        node_id
    )
    if not row:
        return None
    return NodeCapabilities(
        node_id=row["node_id"],
        accel_freqs=list(row["accel_freqs"]),
        gyro_freqs=list(row["gyro_freqs"]),
        enabled_ops=list(row["enabled_ops"]),
        updated_at=row["updated_at"]
    )

async def insert_node_health(conn: asyncpg.Connection, health: NodeHealth) -> int:
    # record health report
    val = await conn.fetchval(
        """
        INSERT INTO node_health (node_id, timestamp, ram_free, sd_status, cached_models, last_segment_id, status, ips)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING id;
        """,
        health.node_id,
        health.timestamp,
        health.ram_free,
        health.sd_status,
        health.cached_models,
        health.last_segment_id,
        health.status,
        health.ips
    )
    return val

async def get_latest_node_health(conn: asyncpg.Connection, node_id: str) -> Optional[NodeHealth]:
    # retrieve latest health
    row = await conn.fetchrow(
        """
        SELECT id, node_id, timestamp, ram_free, sd_status, cached_models, last_segment_id, status, ips
        FROM node_health
        WHERE node_id = $1
        ORDER BY timestamp DESC
        LIMIT 1;
        """,
        node_id
    )
    if not row:
        return None
    return NodeHealth(
        id=row["id"],
        node_id=row["node_id"],
        timestamp=row["timestamp"],
        ram_free=row["ram_free"],
        sd_status=row["sd_status"],
        cached_models=list(row["cached_models"]),
        last_segment_id=row["last_segment_id"],
        status=row["status"],
        ips=row["ips"]
    )

async def insert_raw_telemetry(conn: asyncpg.Connection, telemetry: RawTelemetry) -> None:
    # save raw telemetry chunk
    await conn.execute(
        """
        INSERT INTO raw_telemetry (
            node_id, timestamp, segment_id, chunk_id, sample_rate, emit_reason,
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, raw_bytes_count, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (node_id, timestamp, segment_id, chunk_id) DO NOTHING;
        """,
        telemetry.node_id,
        telemetry.timestamp,
        telemetry.segment_id,
        telemetry.chunk_id,
        telemetry.sample_rate,
        telemetry.emit_reason,
        telemetry.accel_x,
        telemetry.accel_y,
        telemetry.accel_z,
        telemetry.gyro_x,
        telemetry.gyro_y,
        telemetry.gyro_z,
        telemetry.raw_bytes_count,
        telemetry.created_at
    )

async def insert_raw_telemetry_batch(conn: asyncpg.Connection, items: List[RawTelemetry]) -> None:
    # batch insert telemetry
    if not items:
        return
    query = """
        INSERT INTO raw_telemetry (
            node_id, timestamp, segment_id, chunk_id, sample_rate, emit_reason,
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, raw_bytes_count, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        ON CONFLICT (node_id, timestamp, segment_id, chunk_id) DO NOTHING;
    """
    args = [
        (
            item.node_id,
            item.timestamp,
            item.segment_id,
            item.chunk_id,
            item.sample_rate,
            item.emit_reason,
            item.accel_x,
            item.accel_y,
            item.accel_z,
            item.gyro_x,
            item.gyro_y,
            item.gyro_z,
            item.raw_bytes_count,
            item.created_at
        )
        for item in items
    ]
    await conn.executemany(query, args)

async def get_raw_telemetry_for_segment(
    conn: asyncpg.Connection,
    node_id: str,
    segment_id: int
) -> List[RawTelemetry]:
    # get chunks for segment
    rows = await conn.fetch(
        """
        SELECT id, node_id, timestamp, segment_id, chunk_id, sample_rate, emit_reason,
               accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, raw_bytes_count, created_at
        FROM raw_telemetry
        WHERE node_id = $1 AND segment_id = $2
        ORDER BY chunk_id ASC;
        """,
        node_id, segment_id
    )
    return [
        RawTelemetry(
            id=r["id"],
            node_id=r["node_id"],
            timestamp=r["timestamp"],
            segment_id=r["segment_id"],
            chunk_id=r["chunk_id"],
            sample_rate=r["sample_rate"],
            emit_reason=r["emit_reason"],
            accel_x=list(r["accel_x"]),
            accel_y=list(r["accel_y"]),
            accel_z=list(r["accel_z"]),
            gyro_x=list(r["gyro_x"]),
            gyro_y=list(r["gyro_y"]),
            gyro_z=list(r["gyro_z"]),
            raw_bytes_count=r["raw_bytes_count"],
            created_at=r["created_at"]
        )
        for r in rows
    ]

async def get_raw_telemetry_volume_bytes(conn: asyncpg.Connection, node_id: str) -> int:
    # calculate total bytes ingested
    val = await conn.fetchval(
        """
        SELECT COALESCE(SUM(raw_bytes_count), 0)
        FROM raw_telemetry
        WHERE node_id = $1;
        """,
        node_id
    )
    return int(val) if val is not None else 0

async def insert_inference_result(conn: asyncpg.Connection, result: InferenceResult) -> None:
    # store inference packet
    await conn.execute(
        """
        INSERT INTO inference_results (
            node_id, timestamp, segment_id, emit_reason, router_model_id,
            autoencoder_model_id, mse, anomaly, is_recalculated, created_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (node_id, timestamp, segment_id) DO UPDATE
        SET mse = EXCLUDED.mse,
            anomaly = EXCLUDED.anomaly,
            is_recalculated = EXCLUDED.is_recalculated;
        """,
        result.node_id,
        result.timestamp,
        result.segment_id,
        result.emit_reason,
        result.router_model_id,
        result.autoencoder_model_id,
        result.mse,
        result.anomaly,
        result.is_recalculated,
        result.created_at
    )

async def get_inference_results(
    conn: asyncpg.Connection,
    node_id: str,
    limit: int = 100,
    offset: int = 0
) -> List[InferenceResult]:
    # get inference history
    rows = await conn.fetch(
        """
        SELECT id, node_id, timestamp, segment_id, emit_reason, router_model_id,
               autoencoder_model_id, mse, anomaly, is_recalculated, created_at
        FROM inference_results
        WHERE node_id = $1
        ORDER BY timestamp DESC
        LIMIT $2 OFFSET $3;
        """,
        node_id, limit, offset
    )
    return [
        InferenceResult(
            id=r["id"],
            node_id=r["node_id"],
            timestamp=r["timestamp"],
            segment_id=r["segment_id"],
            emit_reason=r["emit_reason"],
            router_model_id=r["router_model_id"],
            autoencoder_model_id=r["autoencoder_model_id"],
            mse=r["mse"],
            anomaly=r["anomaly"],
            is_recalculated=r["is_recalculated"],
            created_at=r["created_at"]
        )
        for r in rows
    ]

async def update_inference_recalculated(
    conn: asyncpg.Connection,
    node_id: str,
    segment_id: int,
    mse: float,
    anomaly: bool
) -> bool:
    # update recalculated inference score
    res = await conn.execute(
        """
        UPDATE inference_results
        SET mse = $1,
            anomaly = $2,
            is_recalculated = TRUE
        WHERE node_id = $3 AND segment_id = $4;
        """,
        mse, anomaly, node_id, segment_id
    )
    return "UPDATE 1" in res or "UPDATE" in res

async def insert_model(conn: asyncpg.Connection, model: ModelPackage) -> int:
    # save model binary artifact
    val = await conn.fetchval(
        """
        INSERT INTO models (id, tag, model_type, tflite_binary, size_bytes, config, created_at)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7)
        ON CONFLICT (id) DO UPDATE
        SET tag = EXCLUDED.tag,
            model_type = EXCLUDED.model_type,
            tflite_binary = EXCLUDED.tflite_binary,
            size_bytes = EXCLUDED.size_bytes,
            config = EXCLUDED.config
        RETURNING id;
        """,
        model.id,
        model.tag,
        model.model_type,
        model.tflite_binary,
        model.size_bytes,
        json.dumps(model.config),
        model.created_at
    )
    return val

async def get_model(conn: asyncpg.Connection, model_id: int) -> Optional[ModelPackage]:
    # fetch model by id
    row = await conn.fetchrow(
        "SELECT id, tag, model_type, tflite_binary, size_bytes, config, created_at FROM models WHERE id = $1;",
        model_id
    )
    if not row:
        return None
    cfg = row["config"]
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    return ModelPackage(
        id=row["id"],
        tag=row["tag"],
        model_type=row["model_type"],
        tflite_binary=bytes(row["tflite_binary"]),
        size_bytes=row["size_bytes"],
        config=cfg,
        created_at=row["created_at"]
    )

async def insert_ensemble(conn: asyncpg.Connection, ensemble: EnsembleConfig) -> int:
    # record ensemble deployment
    val = await conn.fetchval(
        """
        INSERT INTO ensembles (id, node_id, router_model_id, memory_model_id, warmup, routes, total_size_bytes, deployed_at)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8)
        ON CONFLICT (id) DO UPDATE
        SET node_id = EXCLUDED.node_id,
            router_model_id = EXCLUDED.router_model_id,
            memory_model_id = EXCLUDED.memory_model_id,
            warmup = EXCLUDED.warmup,
            routes = EXCLUDED.routes,
            total_size_bytes = EXCLUDED.total_size_bytes,
            deployed_at = EXCLUDED.deployed_at
        RETURNING id;
        """,
        ensemble.id,
        ensemble.node_id,
        ensemble.router_model_id,
        ensemble.memory_model_id,
        ensemble.warmup,
        json.dumps(ensemble.routes),
        ensemble.total_size_bytes,
        ensemble.deployed_at
    )
    return val

async def get_latest_ensemble(conn: asyncpg.Connection, node_id: str) -> Optional[EnsembleConfig]:
    # get active ensemble for node
    row = await conn.fetchrow(
        """
        SELECT id, node_id, router_model_id, memory_model_id, warmup, routes, total_size_bytes, deployed_at
        FROM ensembles
        WHERE node_id = $1
        ORDER BY deployed_at DESC
        LIMIT 1;
        """,
        node_id
    )
    if not row:
        return None
    rts = row["routes"]
    if isinstance(rts, str):
        rts = json.loads(rts)
    return EnsembleConfig(
        id=row["id"],
        node_id=row["node_id"],
        router_model_id=row["router_model_id"],
        memory_model_id=row["memory_model_id"],
        warmup=row["warmup"],
        routes=rts,
        total_size_bytes=row["total_size_bytes"],
        deployed_at=row["deployed_at"]
    )
