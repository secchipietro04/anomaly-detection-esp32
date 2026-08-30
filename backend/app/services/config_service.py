# sensor runtime configuration management and downlink sync
import logging
from typing import Optional, Dict, Any, Union
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import NodeModel, NodeCapabilitiesModel
from app.cbor.schemas import RuntimeConfig
from app.cbor.codec import to_cbor, from_cbor
from app.mqtt.publisher import MQTTPublisher

logger = logging.getLogger("config_service")

class ConfigValidationError(ValueError):
    pass

class NodeNotFoundError(LookupError):
    pass

def validate_runtime_config(
    config: Union[RuntimeConfig, Dict[str, Any]],
    capabilities: Optional[NodeCapabilitiesModel] = None
) -> RuntimeConfig:
    # validate runtime configuration against schema and capabilities
    if isinstance(config, dict):
        if "mode" in config and (config["mode"] < 1 or config["mode"] > 4):
            raise ConfigValidationError(f"Invalid mode: {config.get('mode')}")
        if "rate" in config and config["rate"] <= 0:
            raise ConfigValidationError(f"Invalid rate: {config.get('rate')}")
        if "beat" in config and config["beat"] <= 0:
            raise ConfigValidationError(f"Invalid beat: {config.get('beat')}")
        if "batch" in config and config["batch"] <= 0:
            raise ConfigValidationError(f"Invalid batch: {config.get('batch')}")
        if "sd_en" in config and not isinstance(config["sd_en"], bool):
            raise ConfigValidationError("sd_en must be boolean")
        try:
            cfg = RuntimeConfig.model_validate(config)
        except Exception as e:
            raise ConfigValidationError(f"Schema validation error: {e}") from e
    elif isinstance(config, RuntimeConfig):
        cfg = config
    else:
        raise ConfigValidationError("config must be RuntimeConfig or dict")

    if capabilities is not None and capabilities.accel_freqs:
        supported = any(abs(cfg.rate - float(f)) < 1e-3 for f in capabilities.accel_freqs)
        if not supported:
            raise ConfigValidationError(f"Sampling rate {cfg.rate} Hz not supported by capabilities {capabilities.accel_freqs}")

    return cfg

async def sync_sensor_config(
    session: AsyncSession,
    publisher: MQTTPublisher,
    node_id: str,
    config: Union[RuntimeConfig, Dict[str, Any]],
    require_capabilities: bool = False
) -> RuntimeConfig:
    # validate, deploy cbor downlink to sensor and save to db
    node = await session.get(NodeModel, node_id)
    if not node:
        raise NodeNotFoundError(f"Node '{node_id}' not found")

    caps = await session.get(NodeCapabilitiesModel, node_id)
    validated_cfg = validate_runtime_config(config, capabilities=caps if require_capabilities else None)

    # publish cbor to v1/{node_id}/config
    cbor_bytes = to_cbor(validated_cfg)
    await publisher.publish_config(node_id, cbor_bytes)

    # update db
    node.current_config = validated_cfg.model_dump(exclude_none=True)
    await session.commit()
    logger.info(f"Deployed configuration to node {node_id}")
    return validated_cfg
