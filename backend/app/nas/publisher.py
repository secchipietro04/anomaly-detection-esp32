# persists trained models to db and publishes lightweight ensemble routing table
import logging
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete

from app.cbor.codec import to_cbor
from app.cbor.schemas import (
    RouterModelPackage, AutoencoderModelPackage, MemoryModelPackage, EnsembleConfig, RouteEntry
)
from app.database.models import ModelPackageModel, EnsembleConfigModel, NodeModel, db_register_node
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
    # 0. ensure node exists in database
    await db_register_node(session, node_id)

    # 1. in-place cleanup: remove previous obsolete ensembles and models for this node from db
    stmt_del_e = delete(EnsembleConfigModel).where(EnsembleConfigModel.node_id == node_id)
    await session.execute(stmt_del_e)
    stmt_del_m = delete(ModelPackageModel).where(ModelPackageModel.node_id == node_id)
    await session.execute(stmt_del_m)

    await publisher.connect()
    try:
        # 1. save router model to db (Postgres autoincrement assigns id)
        r_pkg = result.router_model
        r_record = ModelPackageModel(
            node_id=node_id,
            tag=r_pkg.tag,
            model_type=int(r_pkg.type),
            tflite_binary=r_pkg.data,
            size_bytes=len(r_pkg.data),
            config=r_pkg.model_dump(exclude={"data", "m_id"})
        )
        session.add(r_record)
        await session.flush()
        r_pkg.m_id = r_record.id
        await publisher.publish_model(node_id, "router", r_pkg.m_id, to_cbor(r_pkg), retain=True)
        logger.info(f"Published router model {r_pkg.m_id} to node {node_id}")

        # 2. save each autoencoder submodel to db (Postgres autoincrement assigns id)
        routes = []
        for ix, ae_pkg in enumerate(result.autoencoder_models):
            ae_record = ModelPackageModel(
                node_id=node_id,
                tag=ae_pkg.tag,
                model_type=int(ae_pkg.type),
                tflite_binary=ae_pkg.data,
                size_bytes=len(ae_pkg.data),
                config=ae_pkg.model_dump(exclude={"data", "m_id"})
            )
            session.add(ae_record)
            await session.flush()
            ae_pkg.m_id = ae_record.id
            routes.append(RouteEntry(out_ix=ix, m_id=ae_pkg.m_id))
            await publisher.publish_model(node_id, "submodel", ae_pkg.m_id, to_cbor(ae_pkg), retain=True)
            logger.info(f"Published submodel {ae_pkg.m_id} to node {node_id}")

        # 3. save and publish lightweight ensemble routing table (Postgres autoincrement assigns id)
        ens_cfg = EnsembleConfig(warmup=10, r_m_id=r_pkg.m_id, mem_id=None, routes=routes)
        ens_record = EnsembleConfigModel(
            node_id=node_id,
            router_model_id=r_pkg.m_id,
            memory_model_id=None,
            warmup=ens_cfg.warmup,
            routes=[r.model_dump() for r in routes],
            total_size_bytes=result.total_size_bytes
        )
        session.add(ens_record)
        await session.flush()
        await publisher.publish_ensemble(node_id, to_cbor(ens_cfg), retain=True)
        logger.info(f"Published ensemble config (id {ens_record.id}) to node {node_id}")

        # 4. update node training checkpoint and reset in-memory volume accumulator
        node = await session.get(NodeModel, node_id)
        if node:
            if latest_segment_id > 0:
                node.last_trained_segment_id = latest_segment_id
        await session.commit()
        reset_node_volume(node_id)
        logger.info(f"NAS models and ensemble successfully deployed for node {node_id}")
    finally:
        await publisher.disconnect()
