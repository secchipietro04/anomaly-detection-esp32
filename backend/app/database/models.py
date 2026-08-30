# sqlalchemy orm models for timescaledb
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy import (
    String, Integer, BigInteger, Float, Boolean,
    DateTime, ForeignKey, Index, select, update, func
)
from sqlalchemy.dialects.postgresql import ARRAY, BYTEA, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncSession

class Base(DeclarativeBase):
    pass

class NodeModel(Base):
    # sensor registry
    __tablename__ = "nodes"

    node_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    status: Mapped[str] = mapped_column(String(32), default="registered")
    current_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    last_trained_segment_id: Mapped[int] = mapped_column(BigInteger, default=0)

class NodeCapabilitiesModel(Base):
    # supported freqs and compiled tflite ops
    __tablename__ = "node_capabilities"

    node_id: Mapped[str] = mapped_column(String(64), ForeignKey("nodes.node_id", ondelete="CASCADE"), primary_key=True)
    accel_freqs: Mapped[List[float]] = mapped_column(ARRAY(Float), default=list)
    gyro_freqs: Mapped[List[float]] = mapped_column(ARRAY(Float), default=list)
    enabled_ops: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class NodeHealthModel(Base):
    # periodic health reports
    __tablename__ = "node_health"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String(64), ForeignKey("nodes.node_id", ondelete="CASCADE"), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ram_free: Mapped[int] = mapped_column(BigInteger, default=0)
    sd_status: Mapped[int] = mapped_column(Integer, default=0)
    cached_models: Mapped[List[int]] = mapped_column(ARRAY(Integer), default=list)
    last_segment_id: Mapped[int] = mapped_column(BigInteger, default=0)
    status: Mapped[str] = mapped_column(String(32), default="idle")
    ips: Mapped[float] = mapped_column(Float, default=0.0)

class RawTelemetryModel(Base):
    # timescaledb hypertable for raw vibration data
    __tablename__ = "raw_telemetry"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String(64), ForeignKey("nodes.node_id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    segment_id: Mapped[int] = mapped_column(BigInteger)
    chunk_id: Mapped[int] = mapped_column(Integer, default=1)
    sample_rate: Mapped[float] = mapped_column(Float, default=0.0)
    emit_reason: Mapped[int] = mapped_column(Integer, default=0)
    accel_x: Mapped[List[float]] = mapped_column(ARRAY(Float), default=list)
    accel_y: Mapped[List[float]] = mapped_column(ARRAY(Float), default=list)
    accel_z: Mapped[List[float]] = mapped_column(ARRAY(Float), default=list)
    gyro_x: Mapped[List[float]] = mapped_column(ARRAY(Float), default=list)
    gyro_y: Mapped[List[float]] = mapped_column(ARRAY(Float), default=list)
    gyro_z: Mapped[List[float]] = mapped_column(ARRAY(Float), default=list)
    raw_bytes_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_raw_telemetry_node_seg", "node_id", "segment_id"),
    )

class InferenceResultModel(Base):
    # timescaledb hypertable for edge and recalculated inference scores
    __tablename__ = "inference_results"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    node_id: Mapped[str] = mapped_column(String(64), ForeignKey("nodes.node_id", ondelete="CASCADE"))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    segment_id: Mapped[int] = mapped_column(BigInteger)
    emit_reason: Mapped[int] = mapped_column(Integer, default=0)
    router_model_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    autoencoder_model_id: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    mse: Mapped[float] = mapped_column(Float, default=0.0)
    anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    is_recalculated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (
        Index("idx_inference_node_seg", "node_id", "segment_id"),
    )

class ModelPackageModel(Base):
    # persistent tflite binaries and metadata
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    node_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    tag: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_type: Mapped[int] = mapped_column(Integer) # 1=ae, 2=router, 3=memory
    tflite_binary: Mapped[bytes] = mapped_column(BYTEA)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    config: Mapped[Dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class EnsembleConfigModel(Base):
    # routing table and active ensemble metadata
    __tablename__ = "ensembles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    node_id: Mapped[Optional[str]] = mapped_column(String(64), ForeignKey("nodes.node_id", ondelete="SET NULL"), nullable=True)
    router_model_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("models.id", ondelete="SET NULL"), nullable=True)
    memory_model_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("models.id", ondelete="SET NULL"), nullable=True)
    warmup: Mapped[int] = mapped_column(Integer, default=0)
    routes: Mapped[List[Dict[str, Any]]] = mapped_column(JSONB, default=list)
    total_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    deployed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# --- orm query helpers ---

async def db_register_node(session: AsyncSession, node_id: str, name: Optional[str] = None) -> NodeModel:
    # insert or update node
    node = await session.get(NodeModel, node_id)
    if not node:
        node = NodeModel(node_id=node_id, name=name, last_seen=datetime.now(timezone.utc))
        session.add(node)
    else:
        node.last_seen = datetime.now(timezone.utc)
        if name:
            node.name = name
    await session.commit()
    return node

async def db_upsert_capabilities(
    session: AsyncSession,
    node_id: str,
    accel_freqs: List[float],
    gyro_freqs: List[float],
    enabled_ops: List[str]
) -> NodeCapabilitiesModel:
    # save node capabilities
    caps = await session.get(NodeCapabilitiesModel, node_id)
    if not caps:
        caps = NodeCapabilitiesModel(
            node_id=node_id,
            accel_freqs=accel_freqs,
            gyro_freqs=gyro_freqs,
            enabled_ops=enabled_ops
        )
        session.add(caps)
    else:
        caps.accel_freqs = accel_freqs
        caps.gyro_freqs = gyro_freqs
        caps.enabled_ops = enabled_ops
        caps.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return caps

async def db_get_capabilities(session: AsyncSession, node_id: str) -> Optional[NodeCapabilitiesModel]:
    # get caps
    return await session.get(NodeCapabilitiesModel, node_id)

async def db_get_latest_health(session: AsyncSession, node_id: str) -> Optional[NodeHealthModel]:
    # get latest health record
    stmt = select(NodeHealthModel).where(NodeHealthModel.node_id == node_id).order_by(NodeHealthModel.timestamp.desc()).limit(1)
    res = await session.execute(stmt)
    return res.scalar_one_or_none()

async def db_get_raw_telemetry_for_segment(session: AsyncSession, node_id: str, segment_id: int) -> List[RawTelemetryModel]:
    # pull raw chunks for segment
    stmt = (
        select(RawTelemetryModel)
        .where(RawTelemetryModel.node_id == node_id, RawTelemetryModel.segment_id == segment_id)
        .order_by(RawTelemetryModel.chunk_id.asc())
    )
    res = await session.execute(stmt)
    return list(res.scalars().all())

async def db_get_untrained_bytes_for_node(session: AsyncSession, node_id: str) -> int:
    # fast indexed startup lookup for accumulated bytes since checkpoint
    node = await session.get(NodeModel, node_id)
    last_seg = node.last_trained_segment_id if node else 0
    stmt = (
        select(func.coalesce(func.sum(RawTelemetryModel.raw_bytes_count), 0))
        .where(RawTelemetryModel.node_id == node_id, RawTelemetryModel.segment_id > last_seg)
    )
    res = await session.execute(stmt)
    return int(res.scalar() or 0)
