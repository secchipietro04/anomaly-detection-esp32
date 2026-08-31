# persists trained models to db and publishes lightweight ensemble routing table
import logging
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete

from app.cbor.codec import to_cbor
from app.cbor.schemas import (
    RouterModelPackage, AutoencoderModelPackage, MemoryModelPackage, EnsembleConfig
)
from app.database.models import ModelPackageModel, EnsembleConfigModel, NodeModel
from app.mqtt.publisher import MQTTPublisher
from app.nas.search import NASSearchResult
from app.mqtt.handlers import reset_node_volume

logger = logging.getLogger("nas_publisher")

async def deploy_nas_result(
    session: AsyncSession,
    publisher: MQTTPublisher,
    node_id: str,
    result: NASSearchResult,
    latest_segment_id: int = 0
) -> None:
    # 0. in-place cleanup: remove previous obsolete models and ensembles for this node from db
    stmt_del_m = delete(ModelPackageModel).where(ModelPackageModel.node_id == node_id)
    await session.execute(stmt_del_m)
    stmt_del_e = delete(EnsembleConfigModel).where(EnsembleConfigModel.node_id == node_id)
    await session.execute(stmt_del_e)

    # 1. save router model to db
    r_pkg = result.router_model
    r_record = ModelPackageModel(
        id=r_pkg.m_id,
        node_id=node_id,
        tag=r_pkg.tag,
        model_type=int(r_pkg.type),
        tflite_binary=r_pkg.data,
        size_bytes=len(r_pkg.data),
        config=r_pkg.model_dump(exclude={"data"})
    )
    session.add(r_record)

    # 2. save memory model to db if present
    if result.memory_model:
        m_pkg = result.memory_model
        m_record = ModelPackageModel(
            id=m_pkg.m_id,
            node_id=node_id,
            tag=m_pkg.tag,
            model_type=int(m_pkg.type),
            tflite_binary=m_pkg.data,
            size_bytes=len(m_pkg.data),
            config=m_pkg.model_dump(exclude={"data"})
        )
        session.add(m_record)

    # 3. save each autoencoder submodel to db
    for ae_pkg in result.autoencoder_models:
        ae_record = ModelPackageModel(
            id=ae_pkg.m_id,
            node_id=node_id,
            tag=ae_pkg.tag,
            model_type=int(ae_pkg.type),
            tflite_binary=ae_pkg.data,
            size_bytes=len(ae_pkg.data),
            config=ae_pkg.model_dump(exclude={"data"})
        )
        session.add(ae_record)

    # 4. save and publish lightweight ensemble routing table (retain=False)
    ens_cfg = result.ensemble_config
    ens_record = EnsembleConfigModel(
        id=int(datetime.now(timezone.utc).timestamp()),
        node_id=node_id,
        router_model_id=r_pkg.m_id,
        memory_model_id=result.memory_model.m_id if result.memory_model else None,
        warmup=ens_cfg.warmup,
        routes=[r.model_dump() for r in ens_cfg.routes],
        total_size_bytes=result.total_size_bytes
    )
    session.add(ens_record)
    await publisher.publish_ensemble(node_id, to_cbor(ens_cfg))

    # 5. update node training checkpoint and reset in-memory volume accumulator
    node = await session.get(NodeModel, node_id)
    if node:
        if latest_segment_id > 0:
            node.last_trained_segment_id = latest_segment_id
    await session.commit()
    reset_node_volume(node_id)

    logger.info(f"Successfully deployed ensemble for node {node_id}: router={r_pkg.m_id}, submodels={[a.m_id for a in result.autoencoder_models]}")

