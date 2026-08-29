# Sensor configuration management and downlink synchronization service
import asyncio
import logging
from typing import Optional, Dict, Any, Union, List
from pydantic import ValidationError

from app.config import get_settings
from app.database.session import DatabaseSessionManager, db_manager
from app.database.models import (
    Node,
    NodeCapabilities,
    get_node,
    get_node_capabilities,
    update_node_config,
    get_node_config,
)
from app.cbor.schemas import (
    RuntimeConfig,
    StreamMode,
    NodeCapabilities as CBORNodeCapabilities,
)
from app.cbor.codec import encode_runtime_config, decode_runtime_config, encode_cbor
from app.mqtt.publisher import MQTTPublisher

logger = logging.getLogger("config_service")

# exceptions
class ConfigError(Exception):
    # base config error
    pass

class ConfigValidationError(ConfigError, ValueError):
    # raised on config validation failure
    pass

class NodeNotFoundError(ConfigError, LookupError):
    # raised when node is not found in database
    pass

class CapabilitiesNotFoundError(ConfigError, LookupError):
    # raised when node capabilities are missing
    pass

class ConfigDeployError(ConfigError, RuntimeError):
    # raised when downlink deployment fails
    pass

def validate_runtime_config(
    config: Union[RuntimeConfig, Dict[str, Any]],
    capabilities: Optional[Union[NodeCapabilities, CBORNodeCapabilities]] = None
) -> RuntimeConfig:
    # validate runtime configuration parameters against CDDL spec and node capabilities
    if isinstance(config, dict):
        if "sd_en" in config and not isinstance(config["sd_en"], bool):
            raise ConfigValidationError(f"sd_en must be boolean, got {type(config['sd_en'])}")
        if "mode" in config and (isinstance(config["mode"], bool) or not isinstance(config["mode"], int)):
            raise ConfigValidationError(f"Invalid StreamMode: {config.get('mode')}. Must be in 1..4.")
        if "beat" in config and (isinstance(config["beat"], bool) or not isinstance(config["beat"], int)):
            raise ConfigValidationError(f"Heartbeat interval (beat) must be positive integer, got {config.get('beat')}")
        if "batch" in config and (isinstance(config["batch"], bool) or not isinstance(config["batch"], int)):
            raise ConfigValidationError(f"Batch size (batch) must be positive integer, got {config.get('batch')}")
        if "cad" in config and config["cad"] is not None and (isinstance(config["cad"], bool) or not isinstance(config["cad"], int)):
            raise ConfigValidationError(f"Cadence (cad) must be positive integer if specified, got {config.get('cad')}")
        if "rate" in config and (isinstance(config["rate"], bool) or not isinstance(config["rate"], (int, float))):
            raise ConfigValidationError(f"Sampling rate must be numeric, got {type(config.get('rate'))}")
        try:
            cfg = RuntimeConfig.model_validate(config)
        except ValidationError as e:
            raise ConfigValidationError(f"Invalid RuntimeConfig schema: {e}") from e
    elif isinstance(config, RuntimeConfig):
        cfg = config
    else:
        raise ConfigValidationError("config must be RuntimeConfig or dict")

    # validate stream mode
    if not isinstance(cfg.mode, int) or isinstance(cfg.mode, bool) or cfg.mode not in (1, 2, 3, 4):
        raise ConfigValidationError(f"Invalid StreamMode: {cfg.mode}. Must be in 1..4.")

    # check sampling rate positive
    if cfg.rate <= 0:
        raise ConfigValidationError(f"Sampling rate must be positive, got {cfg.rate}")

    # check heartbeat interval
    if not isinstance(cfg.beat, int) or isinstance(cfg.beat, bool) or cfg.beat <= 0:
        raise ConfigValidationError(f"Heartbeat interval (beat) must be positive integer, got {cfg.beat}")

    # check batch size
    if not isinstance(cfg.batch, int) or isinstance(cfg.batch, bool) or cfg.batch <= 0:
        raise ConfigValidationError(f"Batch size (batch) must be positive integer, got {cfg.batch}")

    # validate sd_en boolean
    if not isinstance(cfg.sd_en, bool):
        raise ConfigValidationError(f"sd_en must be boolean, got {type(cfg.sd_en)}")

    # validate optional cadence
    if cfg.cad is not None:
        if not isinstance(cfg.cad, int) or isinstance(cfg.cad, bool) or cfg.cad <= 0:
            raise ConfigValidationError(f"Cadence (cad) must be positive integer if specified, got {cfg.cad}")

    # validate sampling rate against node capabilities if available
    if capabilities is not None and capabilities.accel_freqs:
        # float tolerance comparison for supported sampling rates
        supported = any(abs(cfg.rate - float(f)) < 1e-3 for f in capabilities.accel_freqs)
        if not supported:
            raise ConfigValidationError(
                f"Sampling rate {cfg.rate} Hz is not supported by node capabilities: {capabilities.accel_freqs}"
            )

    return cfg

class ConfigService:
    # manages sensor runtime configurations and dynamic downlink sync
    def __init__(
        self,
        session_manager: Optional[DatabaseSessionManager] = None,
        publisher: Optional[Any] = None,
        topic_prefix: Optional[str] = None,
    ):
        settings = get_settings()
        self.db = session_manager or db_manager
        self.topic_prefix = topic_prefix or settings.mqtt_topic_prefix
        self._publisher = publisher

    @property
    def publisher(self) -> Any:
        # lazy initialize publisher
        if self._publisher is None:
            settings = get_settings()
            self._publisher = MQTTPublisher(
                broker=settings.mqtt_broker,
                port=settings.mqtt_port,
                topic_prefix=self.topic_prefix
            )
        return self._publisher

    async def get_node_capabilities(self, node_id: str) -> Optional[NodeCapabilities]:
        # fetch capabilities from database
        async with self.db.acquire() as conn:
            return await get_node_capabilities(conn, node_id)

    async def get_node_config(self, node_id: str) -> Optional[RuntimeConfig]:
        # fetch active persistent config from database
        async with self.db.acquire() as conn:
            cfg_dict = await get_node_config(conn, node_id)
            if not cfg_dict:
                return None
            return RuntimeConfig.model_validate(cfg_dict)

    async def validate_node_config(
        self,
        node_id: str,
        config: Union[RuntimeConfig, Dict[str, Any]],
        require_capabilities: bool = True
    ) -> RuntimeConfig:
        # validate config against stored node capabilities
        if not node_id or not isinstance(node_id, str):
            raise ConfigValidationError("node_id must be non-empty string")
            
        async with self.db.acquire() as conn:
            node = await get_node(conn, node_id)
            if not node:
                raise NodeNotFoundError(f"Node '{node_id}' not found in database")
            
            caps = await get_node_capabilities(conn, node_id)
            if require_capabilities and (caps is None or not caps.accel_freqs):
                raise CapabilitiesNotFoundError(
                    f"Node '{node_id}' has no registered capabilities or sampling frequencies"
                )
            
            return validate_runtime_config(config, capabilities=caps)

    async def deploy_config(
        self,
        node_id: str,
        config: Union[RuntimeConfig, Dict[str, Any]],
        qos: int = 1,
        require_capabilities: bool = True
    ) -> bytes:
        # validate, encode into binary CBOR, publish to MQTT and update database
        validated_cfg = await self.validate_node_config(
            node_id=node_id,
            config=config,
            require_capabilities=require_capabilities
        )

        # encode into binary cbor
        try:
            cbor_payload = encode_runtime_config(validated_cfg)
        except Exception as e:
            raise ConfigDeployError(f"CBOR encoding failed: {e}") from e

        # publish to mqtt downlink topic
        try:
            pub = self.publisher
            if hasattr(pub, "publish_config"):
                await pub.publish_config(node_id, cbor_payload, qos=qos)
            elif hasattr(pub, "publish"):
                topic = f"{self.topic_prefix}/{node_id}/config"
                await pub.publish(topic, cbor_payload, qos=qos)
            else:
                raise ConfigDeployError("Publisher client missing publish method")
        except Exception as e:
            logger.error(f"MQTT downlink dispatch failed for node {node_id}: {e}")
            raise ConfigDeployError(f"MQTT publish failed: {e}") from e

        # update database persistent state
        try:
            async with self.db.acquire() as conn:
                cfg_dict = validated_cfg.model_dump(exclude_none=True)
                await update_node_config(conn, node_id, cfg_dict)
        except Exception as e:
            logger.error(f"Database update failed for node {node_id}: {e}")
            raise ConfigDeployError(f"Database update failed: {e}") from e

        logger.info(f"Deployed RuntimeConfig to node {node_id} (QoS {qos})")
        return cbor_payload

    async def sync_config(
        self,
        node_id: str,
        config: Union[RuntimeConfig, Dict[str, Any]],
        qos: int = 1,
        require_capabilities: bool = True
    ) -> RuntimeConfig:
        # validate, deploy and return the active RuntimeConfig
        await self.deploy_config(
            node_id=node_id,
            config=config,
            qos=qos,
            require_capabilities=require_capabilities
        )
        if isinstance(config, RuntimeConfig):
            return config
        return RuntimeConfig.model_validate(config)

# helper functions for fastapi routes and backend workers
async def deploy_sensor_config(
    node_id: str,
    config: Union[RuntimeConfig, Dict[str, Any]],
    session_manager: Optional[DatabaseSessionManager] = None,
    publisher: Optional[Any] = None,
    qos: int = 1,
    require_capabilities: bool = True
) -> RuntimeConfig:
    # trigger config deployment to edge sensor
    service = ConfigService(session_manager=session_manager, publisher=publisher)
    return await service.sync_config(
        node_id=node_id,
        config=config,
        qos=qos,
        require_capabilities=require_capabilities
    )

async def get_sensor_config(
    node_id: str,
    session_manager: Optional[DatabaseSessionManager] = None
) -> Optional[RuntimeConfig]:
    # get active config for edge sensor
    service = ConfigService(session_manager=session_manager)
    return await service.get_node_config(node_id)
