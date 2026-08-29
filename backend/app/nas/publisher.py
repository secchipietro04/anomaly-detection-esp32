# CBOR ensemble packaging, TimescaleDB persistence, and MQTT deployment
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple, Union
import asyncpg

from app.config import get_settings
from app.cbor.codec import (
    encode_ensemble_config,
    encode_model_package,
    encode_payload,
)
from app.cbor.schemas import (
    ModelType,
    LossMode,
    ArchitectureTag,
    RouterModelPackage,
    AutoencoderModelPackage,
    MemoryModelPackage,
    EnsembleConfig,
    RouteEntry,
    ModelPackageUnion,
    PayloadUnion,
)
from app.database.models import (
    ModelPackage as DBModelPackage,
    EnsembleConfig as DBEnsembleConfig,
    insert_model,
    insert_ensemble,
)
from app.database.session import DatabaseSessionManager, db_manager
from app.mqtt.publisher import MQTTPublisher
from app.nas.search import NASSearchResult

logger = logging.getLogger("nas_publisher")

def package_ensemble(
    search_result: NASSearchResult,
    node_id: str,
    warmup: int = 10
) -> Tuple[EnsembleConfig, List[ModelPackageUnion]]:
    # extract and finalize ensemble config and list of model packages
    models: List[ModelPackageUnion] = []
    
    if search_result.memory_model is not None:
        models.append(search_result.memory_model)
        
    models.append(search_result.router_model)
    models.extend(search_result.autoencoder_models)
    
    ens_config = search_result.ensemble_config
    ens_config.warmup = warmup
    
    return ens_config, models

async def persist_ensemble_and_models(
    conn: asyncpg.Connection,
    node_id: str,
    ensemble: EnsembleConfig,
    models: List[ModelPackageUnion],
    ensemble_id: Optional[int] = None
) -> Tuple[int, List[int]]:
    # insert all model artifacts and ensemble deployment record into database
    now = datetime.now(timezone.utc)
    persisted_model_ids: List[int] = []
    
    # 1. persist models
    for m in models:
        # build config json for model
        cfg_dict: Dict[str, Any] = {
            "m_id": m.m_id,
            "type": int(m.type),
            "tag": m.tag,
        }
        if isinstance(m, AutoencoderModelPackage):
            cfg_dict.update({
                "accel_bins": m.accel_bins,
                "gyro_bins": m.gyro_bins,
                "tsteps": m.tsteps,
                "limit": m.limit,
                "loss": m.loss,
                "skip": m.skip,
                "mem_d": m.mem_d
            })
        elif isinstance(m, RouterModelPackage):
            cfg_dict.update({
                "accel_bins": m.accel_bins,
                "gyro_bins": m.gyro_bins,
                "tsteps": m.tsteps,
                "class_count": m.class_count,
                "mem_d": m.mem_d
            })
        elif isinstance(m, MemoryModelPackage):
            cfg_dict.update({
                "accel_bins": m.accel_bins,
                "gyro_bins": m.gyro_bins,
                "outdim": m.outdim,
                "state": m.state
            })
            
        db_model = DBModelPackage(
            id=m.m_id,
            tag=m.tag,
            model_type=int(m.type),
            tflite_binary=bytes(m.data),
            size_bytes=len(m.data),
            config=cfg_dict,
            created_at=now
        )
        saved_id = await insert_model(conn, db_model)
        persisted_model_ids.append(saved_id)
        logger.debug(f"Persisted model ID {saved_id} (type {m.type})")

    # 2. persist ensemble
    ens_id = ensemble_id if ensemble_id is not None else int(datetime.now(timezone.utc).timestamp() * 1000) % 2000000000
    total_size = sum(len(m.data) for m in models)
    routes_json = [{"out_ix": int(r.out_ix), "m_id": int(r.m_id)} for r in ensemble.routes]
    
    db_ensemble = DBEnsembleConfig(
        id=ens_id,
        node_id=node_id,
        router_model_id=ensemble.r_m_id,
        memory_model_id=ensemble.mem_id,
        warmup=ensemble.warmup,
        routes=routes_json,
        total_size_bytes=total_size,
        deployed_at=now
    )
    await insert_ensemble(conn, db_ensemble)
    logger.info(f"Persisted ensemble ID {ens_id} for node {node_id} (routes count: {len(routes_json)})")
    
    return ens_id, persisted_model_ids

async def publish_ensemble_to_mqtt(
    publisher: MQTTPublisher,
    node_id: str,
    ensemble: EnsembleConfig,
    models: Optional[List[ModelPackageUnion]] = None,
    publish_models: bool = True,
    qos: int = 1
) -> Dict[str, Any]:
    # serialize ensemble config to CBOR and publish downlink to node
    # 1. encode and publish ensemble configuration
    ens_cbor = encode_ensemble_config(ensemble)
    topic_ens = f"{publisher.topic_prefix}/{node_id}/ensemble"
    await publisher.publish(topic=topic_ens, payload=ens_cbor, qos=qos)
    logger.info(f"Published CBOR EnsembleConfig ({len(ens_cbor)} bytes) to {topic_ens}")

    published_models_count = 0
    # 2. optionally publish model packages
    if publish_models and models:
        for m in models:
            m_cbor = encode_model_package(m)
            # publish to node model downlink topic
            topic_model = f"{publisher.topic_prefix}/{node_id}/model/{m.m_id}"
            await publisher.publish(topic=topic_model, payload=m_cbor, qos=qos)
            published_models_count += 1
            
    return {
        "node_id": node_id,
        "ensemble_published": True,
        "ensemble_topic": topic_ens,
        "ensemble_bytes": len(ens_cbor),
        "models_published_count": published_models_count
    }

async def deploy_nas_ensemble(
    node_id: str,
    search_result: NASSearchResult,
    session_manager: Optional[DatabaseSessionManager] = None,
    publisher: Optional[MQTTPublisher] = None,
    warmup: int = 10,
    publish_models: bool = True
) -> Dict[str, Any]:
    # full pipeline: package, persist in DB, serialize to CBOR, and publish to MQTT
    db = session_manager or db_manager
    ens_config, models = package_ensemble(search_result, node_id=node_id, warmup=warmup)
    
    # DB persistence
    async with db.acquire() as conn:
        ens_id, model_ids = await persist_ensemble_and_models(
            conn=conn,
            node_id=node_id,
            ensemble=ens_config,
            models=models
        )

    # MQTT publish
    pub_result = {}
    if publisher is not None:
        pub_result = await publish_ensemble_to_mqtt(
            publisher=publisher,
            node_id=node_id,
            ensemble=ens_config,
            models=models,
            publish_models=publish_models
        )

    return {
        "status": "deployed",
        "ensemble_id": ens_id,
        "model_ids": model_ids,
        "publish_result": pub_result,
        "total_size_bytes": search_result.total_size_bytes,
        "score": search_result.best_score
    }
