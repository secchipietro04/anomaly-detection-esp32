# background garbage collector clearing obsolete retained models on emqx and db
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.database.models import NodeModel, ModelPackageModel, EnsembleConfigModel
from app.mqtt.publisher import MQTTPublisher

logger = logging.getLogger("nas_gc")

async def run_garbage_collection(session: AsyncSession, publisher: MQTTPublisher, grace_period_hours: int = 1) -> int:
    # purge models and clear emqx retained topics for inactive nodes or evicted models
    cutoff = datetime.now(timezone.utc) - timedelta(hours=grace_period_hours)
    
    # 1. find nodes inactive for > 1 hour
    stmt_inactive = select(NodeModel).where(NodeModel.last_seen < cutoff)
    res_inactive = await session.execute(stmt_inactive)
    inactive_nodes = list(res_inactive.scalars().all())
    
    cleared_count = 0
    for node in inactive_nodes:
        # find models belonging to inactive node
        stmt_m = select(ModelPackageModel).where(ModelPackageModel.node_id == node.node_id)
        res_m = await session.execute(stmt_m)
        models = list(res_m.scalars().all())
        
        for m in models:
            m_type_str = "submodel" if m.model_type == 1 else ("router" if m.model_type == 2 else "memory")
            try:
                await publisher.clear_retained_model(node.node_id, m_type_str, m.id)
                cleared_count += 1
            except Exception as e:
                logger.warning(f"Failed to clear retained topic for node {node.node_id} model {m.id}: {e}")
                
    logger.info(f"Garbage collection completed. Cleared {cleared_count} obsolete model topics.")
    return cleared_count
