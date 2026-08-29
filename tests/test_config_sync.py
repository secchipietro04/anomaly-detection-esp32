# Tests for sensor configuration management, validation, CBOR downlink encoding and sync
import os
import sys
import json
import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# ensure backend is in python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.cbor.schemas import RuntimeConfig, StreamMode, NodeCapabilities as CBORNodeCapabilities
from app.cbor.codec import (
    encode_runtime_config,
    decode_runtime_config,
    encode_cbor,
    decode_cbor,
    CBORValidationError,
)
from app.database.models import Node, NodeCapabilities, update_node_config, get_node_config
from app.mqtt.publisher import MQTTPublisher
from app.services.config_service import (
    ConfigError,
    ConfigValidationError,
    NodeNotFoundError,
    CapabilitiesNotFoundError,
    ConfigDeployError,
    validate_runtime_config,
    ConfigService,
    deploy_sensor_config,
    get_sensor_config,
)

# ==============================================================================
# 1. RUNTIME CONFIG VALIDATION TESTS
# ==============================================================================

def test_validate_runtime_config_valid_all_modes():
    # test validation for all valid stream modes
    caps = NodeCapabilities(
        node_id="sensor_01",
        accel_freqs=[12.5, 26.0, 52.0, 104.0, 208.0, 416.0, 833.0, 1666.0],
        gyro_freqs=[12.5, 26.0, 52.0, 104.0, 208.0, 416.0, 833.0, 1666.0],
        enabled_ops=["CONV_2D"]
    )

    for mode in (
        StreamMode.CONTINUOUS,
        StreamMode.ANOMALY_ONLY,
        StreamMode.ANOMALY_OR_PERIODIC,
        StreamMode.CONTINUOUS_SCORE_RAW_ANOMALY,
    ):
        cfg = RuntimeConfig(
            rate=833.0,
            mode=int(mode),
            batch=512,
            beat=10,
            sd_en=True,
            cad=5
        )
        validated = validate_runtime_config(cfg, capabilities=caps)
        assert validated.mode == int(mode)
        assert validated.rate == 833.0
        assert validated.batch == 512
        assert validated.beat == 10
        assert validated.sd_en is True
        assert validated.cad == 5

def test_validate_runtime_config_without_cadence():
    # test validation when cad is omitted
    cfg = RuntimeConfig(
        rate=104.0,
        mode=1,
        batch=256,
        beat=15,
        sd_en=False
    )
    validated = validate_runtime_config(cfg)
    assert validated.cad is None
    assert validated.sd_en is False

def test_validate_runtime_config_from_dict():
    # test validating dictionary inputs
    dict_cfg = {
        "rate": 416.0,
        "mode": 2,
        "batch": 128,
        "beat": 30,
        "sd_en": True,
        "cad": 2
    }
    validated = validate_runtime_config(dict_cfg)
    assert isinstance(validated, RuntimeConfig)
    assert validated.rate == 416.0
    assert validated.mode == 2
    assert validated.batch == 128

def test_validate_runtime_config_unsupported_sampling_rate():
    # test failure on unsupported sampling rate
    caps = NodeCapabilities(
        node_id="sensor_01",
        accel_freqs=[12.5, 26.0, 52.0, 104.0],
        gyro_freqs=[12.5, 26.0, 52.0, 104.0],
        enabled_ops=["CONV_2D"]
    )
    cfg = RuntimeConfig(
        rate=833.0, # Not in accel_freqs
        mode=1,
        batch=256,
        beat=10,
        sd_en=False
    )
    with pytest.raises(ConfigValidationError, match="not supported by node capabilities"):
        validate_runtime_config(cfg, capabilities=caps)

def test_validate_runtime_config_invalid_modes():
    # test mode boundary and invalid values
    for bad_mode in (0, -1, 5, 99, 100):
        with pytest.raises(ConfigValidationError, match="Invalid StreamMode"):
            validate_runtime_config({
                "rate": 100.0,
                "mode": bad_mode,
                "batch": 256,
                "beat": 10,
                "sd_en": True
            })

def test_validate_runtime_config_invalid_beat():
    # test heartbeat interval <= 0
    for bad_beat in (0, -1, -50):
        with pytest.raises(ConfigValidationError, match="Heartbeat interval"):
            validate_runtime_config({
                "rate": 100.0,
                "mode": 1,
                "batch": 256,
                "beat": bad_beat,
                "sd_en": True
            })

def test_validate_runtime_config_invalid_batch():
    # test batch size <= 0
    for bad_batch in (0, -1, -256):
        with pytest.raises(ConfigValidationError, match="Batch size"):
            validate_runtime_config({
                "rate": 100.0,
                "mode": 1,
                "batch": bad_batch,
                "beat": 10,
                "sd_en": True
            })

def test_validate_runtime_config_invalid_sd_en():
    # test non-boolean sd_en
    with pytest.raises(ConfigValidationError, match="sd_en must be boolean"):
        validate_runtime_config({
            "rate": 100.0,
            "mode": 1,
            "batch": 256,
            "beat": 10,
            "sd_en": "yes" # String instead of bool
        })

def test_validate_runtime_config_invalid_cadence():
    # test cadence <= 0 or invalid type
    for bad_cad in (0, -1, -10):
        with pytest.raises(ConfigValidationError, match="Cadence"):
            validate_runtime_config({
                "rate": 100.0,
                "mode": 1,
                "batch": 256,
                "beat": 10,
                "sd_en": True,
                "cad": bad_cad
            })

def test_validate_runtime_config_negative_rate():
    # test rate <= 0
    for bad_rate in (0.0, -10.0, -100.0):
        with pytest.raises(ConfigValidationError, match="Sampling rate must be positive"):
            validate_runtime_config({
                "rate": bad_rate,
                "mode": 1,
                "batch": 256,
                "beat": 10,
                "sd_en": True
            })

def test_validate_runtime_config_float_precision_tolerance():
    # test floating point rate matching with tolerance
    caps = NodeCapabilities(
        node_id="sensor_01",
        accel_freqs=[833.3333, 1666.6666],
        gyro_freqs=[833.3333, 1666.6666],
        enabled_ops=["CONV_2D"]
    )
    # 833.333 matches 833.3333 within 1e-3
    cfg = RuntimeConfig(
        rate=833.333,
        mode=1,
        batch=256,
        beat=10,
        sd_en=False
    )
    validated = validate_runtime_config(cfg, capabilities=caps)
    assert validated.rate == pytest.approx(833.333)

# ==============================================================================
# 2. CBOR SERIALIZATION & ROUNDTRIP TESTS
# ==============================================================================

def test_config_cbor_serialization_and_deserialization():
    # test CBOR encoding and decoding of RuntimeConfig
    cfg = RuntimeConfig(
        rate=1666.0,
        mode=StreamMode.CONTINUOUS_SCORE_RAW_ANOMALY,
        batch=1024,
        beat=60,
        sd_en=True,
        cad=16
    )
    
    encoded = encode_runtime_config(cfg)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0
    
    decoded = decode_runtime_config(encoded)
    assert decoded.rate == 1666.0
    assert decoded.mode == 4
    assert decoded.batch == 1024
    assert decoded.beat == 60
    assert decoded.sd_en is True
    assert decoded.cad == 16

def test_config_cbor_generic_alias_roundtrip():
    # test encode_cbor and decode_cbor aliases
    cfg = RuntimeConfig(
        rate=52.0,
        mode=StreamMode.ANOMALY_ONLY,
        batch=64,
        beat=5,
        sd_en=False
    )
    
    encoded = encode_cbor(cfg)
    decoded = decode_cbor(encoded, RuntimeConfig)
    assert decoded.rate == 52.0
    assert decoded.mode == 2
    assert decoded.batch == 64
    assert decoded.beat == 5
    assert decoded.sd_en is False
    assert decoded.cad is None

# ==============================================================================
# 3. CONFIG SERVICE & DATABASE INTEGRATION TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_config_service_validate_node_not_found():
    # test validate_node_config when node is missing from DB
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    service = ConfigService(session_manager=mock_db)
    
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=None)):
        with pytest.raises(NodeNotFoundError, match="Node 'unknown_sensor' not found"):
            await service.validate_node_config(
                node_id="unknown_sensor",
                config={"rate": 100.0, "mode": 1, "batch": 100, "beat": 10, "sd_en": True}
            )

@pytest.mark.asyncio
async def test_config_service_validate_capabilities_not_found():
    # test validate_node_config when node has no capabilities registered
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    now = datetime.now(timezone.utc)
    mock_node = Node(node_id="node_no_caps", registered_at=now, last_seen=now, status="registered")
    
    service = ConfigService(session_manager=mock_db)
    
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=mock_node)), \
         patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=None)):
        
        with pytest.raises(CapabilitiesNotFoundError, match="has no registered capabilities"):
            await service.validate_node_config(
                node_id="node_no_caps",
                config={"rate": 100.0, "mode": 1, "batch": 100, "beat": 10, "sd_en": True},
                require_capabilities=True
            )

@pytest.mark.asyncio
async def test_config_service_deploy_config_success():
    # test full deploy_config pipeline: validate -> encode CBOR -> publish MQTT -> update DB
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    mock_publisher = AsyncMock()
    service = ConfigService(session_manager=mock_db, publisher=mock_publisher, topic_prefix="v1")
    
    now = datetime.now(timezone.utc)
    mock_node = Node(node_id="node_edge_01", registered_at=now, last_seen=now, status="registered")
    mock_caps = NodeCapabilities(
        node_id="node_edge_01",
        accel_freqs=[50.0, 100.0, 200.0],
        gyro_freqs=[50.0, 100.0, 200.0],
        enabled_ops=["CONV_2D"]
    )
    
    cfg = RuntimeConfig(
        rate=100.0,
        mode=StreamMode.ANOMALY_OR_PERIODIC,
        batch=512,
        beat=20,
        sd_en=True,
        cad=8
    )
    
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=mock_node)), \
         patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=mock_caps)), \
         patch("app.services.config_service.update_node_config", new=AsyncMock(return_value=True)) as mock_update_db:
        
        cbor_bytes = await service.deploy_config("node_edge_01", cfg, qos=1)
        
        # verify CBOR binary output
        assert isinstance(cbor_bytes, bytes)
        decoded = decode_runtime_config(cbor_bytes)
        assert decoded.rate == 100.0
        assert decoded.mode == 3
        assert decoded.batch == 512
        assert decoded.beat == 20
        assert decoded.sd_en is True
        assert decoded.cad == 8
        
        # verify MQTT publish
        mock_publisher.publish_config.assert_awaited_once_with("node_edge_01", cbor_bytes, qos=1)
        
        # verify DB persistence
        mock_update_db.assert_awaited_once_with(
            mock_conn,
            "node_edge_01",
            cfg.model_dump(exclude_none=True)
        )

@pytest.mark.asyncio
async def test_config_service_deploy_config_mqtt_failure():
    # test handling of MQTT publisher error
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    mock_publisher = AsyncMock()
    mock_publisher.publish_config.side_effect = RuntimeError("Broker connection lost")
    
    service = ConfigService(session_manager=mock_db, publisher=mock_publisher)
    
    now = datetime.now(timezone.utc)
    mock_node = Node(node_id="node_01", registered_at=now, last_seen=now, status="registered")
    mock_caps = NodeCapabilities(node_id="node_01", accel_freqs=[100.0], enabled_ops=[])
    
    cfg = RuntimeConfig(rate=100.0, mode=1, batch=100, beat=10, sd_en=False)
    
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=mock_node)), \
         patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=mock_caps)):
        
        with pytest.raises(ConfigDeployError, match="MQTT publish failed"):
            await service.deploy_config("node_01", cfg)

@pytest.mark.asyncio
async def test_config_service_deploy_config_database_failure():
    # test handling of database persistence error
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    mock_publisher = AsyncMock()
    service = ConfigService(session_manager=mock_db, publisher=mock_publisher)
    
    now = datetime.now(timezone.utc)
    mock_node = Node(node_id="node_01", registered_at=now, last_seen=now, status="registered")
    mock_caps = NodeCapabilities(node_id="node_01", accel_freqs=[100.0], enabled_ops=[])
    
    cfg = RuntimeConfig(rate=100.0, mode=1, batch=100, beat=10, sd_en=False)
    
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=mock_node)), \
         patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=mock_caps)), \
         patch("app.services.config_service.update_node_config", new=AsyncMock(side_effect=Exception("DB dead"))):
        
        with pytest.raises(ConfigDeployError, match="Database update failed"):
            await service.deploy_config("node_01", cfg)

@pytest.mark.asyncio
async def test_config_service_sync_config():
    # test sync_config returning active RuntimeConfig
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    mock_publisher = AsyncMock()
    service = ConfigService(session_manager=mock_db, publisher=mock_publisher)
    
    now = datetime.now(timezone.utc)
    mock_node = Node(node_id="node_01", registered_at=now, last_seen=now, status="registered")
    mock_caps = NodeCapabilities(node_id="node_01", accel_freqs=[200.0], enabled_ops=[])
    
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=mock_node)), \
         patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=mock_caps)), \
         patch("app.services.config_service.update_node_config", new=AsyncMock(return_value=True)):
        
        res = await service.sync_config(
            "node_01",
            {"rate": 200.0, "mode": 1, "batch": 256, "beat": 10, "sd_en": True}
        )
        assert isinstance(res, RuntimeConfig)
        assert res.rate == 200.0
        assert res.mode == 1
        assert res.batch == 256

@pytest.mark.asyncio
async def test_config_service_get_node_config_roundtrip():
    # test get_node_config helper
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    service = ConfigService(session_manager=mock_db)
    
    saved_cfg = {
        "rate": 833.0,
        "mode": 3,
        "batch": 512,
        "beat": 15,
        "sd_en": True,
        "cad": 4
    }
    
    with patch("app.services.config_service.get_node_config", new=AsyncMock(return_value=saved_cfg)):
        cfg = await service.get_node_config("node_01")
        assert cfg is not None
        assert cfg.rate == 833.0
        assert cfg.mode == 3
        assert cfg.cad == 4

@pytest.mark.asyncio
async def test_config_service_get_node_config_none():
    # test get_node_config when node has no config
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    service = ConfigService(session_manager=mock_db)
    
    with patch("app.services.config_service.get_node_config", new=AsyncMock(return_value=None)):
        cfg = await service.get_node_config("node_01")
        assert cfg is None

# ==============================================================================
# 4. GLOBAL HELPER FUNCTIONS TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_deploy_sensor_config_helper():
    # test global deploy_sensor_config function
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    mock_publisher = AsyncMock()
    
    now = datetime.now(timezone.utc)
    mock_node = Node(node_id="sensor_helper_01", registered_at=now, last_seen=now, status="registered")
    mock_caps = NodeCapabilities(node_id="sensor_helper_01", accel_freqs=[50.0], enabled_ops=[])
    
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=mock_node)), \
         patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=mock_caps)), \
         patch("app.services.config_service.update_node_config", new=AsyncMock(return_value=True)):
        
        cfg = await deploy_sensor_config(
            node_id="sensor_helper_01",
            config={"rate": 50.0, "mode": 1, "batch": 100, "beat": 5, "sd_en": False},
            session_manager=mock_db,
            publisher=mock_publisher
        )
        assert cfg.rate == 50.0
        assert cfg.sd_en is False
        mock_publisher.publish_config.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_sensor_config_helper():
    # test global get_sensor_config function
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    saved_cfg = {"rate": 50.0, "mode": 1, "batch": 100, "beat": 5, "sd_en": False}
    with patch("app.services.config_service.get_node_config", new=AsyncMock(return_value=saved_cfg)):
        cfg = await get_sensor_config("sensor_helper_01", session_manager=mock_db)
        assert cfg is not None
        assert cfg.rate == 50.0

# ==============================================================================
# 5. MQTT PUBLISHER SERVICE TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_mqtt_publisher_publish_config_and_ensemble():
    # test MQTTPublisher methods with mocked aiomqtt.Client
    pub = MQTTPublisher(broker="localhost", port=1883, topic_prefix="v1")
    mock_client = AsyncMock()
    pub._client = mock_client
    
    cfg_payload = encode_runtime_config(
        RuntimeConfig(rate=100.0, mode=1, batch=100, beat=10, sd_en=False)
    )
    
    await pub.publish_config("node_test", cfg_payload, qos=1)
    mock_client.publish.assert_awaited_with(
        "v1/node_test/config",
        payload=cfg_payload,
        qos=1,
        retain=False
    )
    
    ensemble_payload = b"dummy_ensemble_cbor"
    await pub.publish_ensemble("node_test", ensemble_payload, qos=1)
    mock_client.publish.assert_awaited_with(
        "v1/node_test/ensemble",
        payload=ensemble_payload,
        qos=1,
        retain=False
    )

@pytest.mark.asyncio
async def test_mqtt_publisher_context_manager():
    # test MQTTPublisher async context manager
    pub = MQTTPublisher(broker="localhost", port=1883)
    mock_client = AsyncMock()
    
    with patch("aiomqtt.Client", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        
        async with pub as p:
            assert p._client == mock_client
        assert pub._client is None

# ==============================================================================
# 6. CONCURRENT DEPLOYMENTS & STRESS
# ==============================================================================

@pytest.mark.asyncio
async def test_concurrent_sensor_config_deployments():
    # test concurrent config deployment to multiple sensor nodes
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    mock_publisher = AsyncMock()
    service = ConfigService(session_manager=mock_db, publisher=mock_publisher)
    
    now = datetime.now(timezone.utc)
    nodes = {
        f"node_{i}": Node(node_id=f"node_{i}", registered_at=now, last_seen=now, status="registered")
        for i in range(10)
    }
    caps = {
        f"node_{i}": NodeCapabilities(node_id=f"node_{i}", accel_freqs=[100.0, 200.0], enabled_ops=[])
        for i in range(10)
    }
    
    async def fake_get_node(conn, node_id):
        return nodes.get(node_id)
        
    async def fake_get_caps(conn, node_id):
        return caps.get(node_id)
        
    with patch("app.services.config_service.get_node", side_effect=fake_get_node), \
         patch("app.services.config_service.get_node_capabilities", side_effect=fake_get_caps), \
         patch("app.services.config_service.update_node_config", new=AsyncMock(return_value=True)):
        
        tasks = [
            service.deploy_config(
                f"node_{i}",
                {"rate": 100.0 if i % 2 == 0 else 200.0, "mode": (i % 4) + 1, "batch": 100 * (i + 1), "beat": 5, "sd_en": bool(i % 2)}
            )
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 10
        assert mock_publisher.publish_config.await_count == 10
        
        for i, cbor_data in enumerate(results):
            cfg = decode_runtime_config(cbor_data)
            assert cfg.rate in (100.0, 200.0)
            assert cfg.batch == 100 * (i + 1)

# ==============================================================================
# 7. DATABASE REPOSITORY DIRECT QUERY TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_update_node_config_db_query():
    # test update_node_config SQL execution
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "UPDATE 1"
    
    cfg_data = {"rate": 400.0, "mode": 1, "batch": 512, "beat": 10, "sd_en": True}
    ok = await update_node_config(mock_conn, "sensor_db_01", cfg_data)
    assert ok is True
    mock_conn.execute.assert_awaited_once()
    sql_arg = mock_conn.execute.call_args[0][0]
    assert "UPDATE nodes" in sql_arg
    assert "current_config = $1::jsonb" in sql_arg

@pytest.mark.asyncio
async def test_get_node_config_db_query():
    # test get_node_config SQL execution
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = {
        "current_config": json.dumps({"rate": 400.0, "mode": 1, "batch": 512, "beat": 10, "sd_en": True})
    }
    
    cfg = await get_node_config(mock_conn, "sensor_db_01")
    assert cfg is not None
    assert cfg["rate"] == 400.0
    assert cfg["mode"] == 1
    mock_conn.fetchrow.assert_awaited_once_with(
        "SELECT current_config FROM nodes WHERE node_id = $1;",
        "sensor_db_01"
    )

# ==============================================================================
# 8. ADDITIONAL EDGE CASES & COVERAGE TESTS
# ==============================================================================

def test_validate_runtime_config_invalid_input_types():
    # test passing unsupported types to validate_runtime_config
    for bad_input in ("invalid_string", 12345, [1, 2, 3], None):
        with pytest.raises(ConfigValidationError, match="config must be RuntimeConfig or dict"):
            validate_runtime_config(bad_input)

def test_validate_runtime_config_with_cbor_node_capabilities_schema():
    # test validate_runtime_config with CBORNodeCapabilities object
    cbor_caps = CBORNodeCapabilities(
        accel_freqs=[26.0, 52.0, 104.0],
        gyro_freqs=[26.0, 52.0, 104.0],
        enabled_ops=["ABS", "ADD"]
    )
    cfg = RuntimeConfig(rate=52.0, mode=1, batch=100, beat=10, sd_en=False)
    validated = validate_runtime_config(cfg, capabilities=cbor_caps)
    assert validated.rate == 52.0

@pytest.mark.asyncio
async def test_validate_node_config_empty_node_id():
    # test validate_node_config with empty node_id
    service = ConfigService()
    with pytest.raises(ConfigValidationError, match="node_id must be non-empty string"):
        await service.validate_node_config("", {"rate": 100.0, "mode": 1, "batch": 100, "beat": 10, "sd_en": True})
        
    with pytest.raises(ConfigValidationError, match="node_id must be non-empty string"):
        await service.validate_node_config(None, {"rate": 100.0, "mode": 1, "batch": 100, "beat": 10, "sd_en": True})

@pytest.mark.asyncio
async def test_validate_node_config_without_requiring_capabilities():
    # test validate_node_config with require_capabilities=False
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    now = datetime.now(timezone.utc)
    mock_node = Node(node_id="sensor_flexible", registered_at=now, last_seen=now, status="registered")
    
    service = ConfigService(session_manager=mock_db)
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=mock_node)), \
         patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=None)):
        
        cfg = await service.validate_node_config(
            node_id="sensor_flexible",
            config={"rate": 123.45, "mode": 1, "batch": 100, "beat": 10, "sd_en": False},
            require_capabilities=False
        )
        assert cfg.rate == 123.45

@pytest.mark.asyncio
async def test_config_service_deploy_with_generic_publisher():
    # test deploy_config with publisher that has publish method instead of publish_config
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    generic_pub = MagicMock()
    generic_pub.publish = AsyncMock()
    # explicitly delete publish_config attribute if present
    if hasattr(generic_pub, "publish_config"):
        del generic_pub.publish_config
        
    service = ConfigService(session_manager=mock_db, publisher=generic_pub, topic_prefix="v1")
    
    now = datetime.now(timezone.utc)
    mock_node = Node(node_id="sensor_gen", registered_at=now, last_seen=now, status="registered")
    mock_caps = NodeCapabilities(node_id="sensor_gen", accel_freqs=[100.0], enabled_ops=[])
    
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=mock_node)), \
         patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=mock_caps)), \
         patch("app.services.config_service.update_node_config", new=AsyncMock(return_value=True)):
        
        await service.deploy_config(
            "sensor_gen",
            {"rate": 100.0, "mode": 1, "batch": 100, "beat": 10, "sd_en": False},
            qos=0
        )
        generic_pub.publish.assert_awaited_once()
        topic_arg = generic_pub.publish.call_args[0][0]
        assert topic_arg == "v1/sensor_gen/config"

@pytest.mark.asyncio
async def test_config_service_deploy_invalid_publisher():
    # test deploy_config when publisher has no publish methods
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    invalid_pub = object() # Has neither publish nor publish_config
    service = ConfigService(session_manager=mock_db, publisher=invalid_pub)
    
    now = datetime.now(timezone.utc)
    mock_node = Node(node_id="sensor_bad_pub", registered_at=now, last_seen=now, status="registered")
    mock_caps = NodeCapabilities(node_id="sensor_bad_pub", accel_freqs=[100.0], enabled_ops=[])
    
    with patch("app.services.config_service.get_node", new=AsyncMock(return_value=mock_node)), \
         patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=mock_caps)):
        
        with pytest.raises(ConfigDeployError, match="Publisher client missing publish method"):
            await service.deploy_config(
                "sensor_bad_pub",
                {"rate": 100.0, "mode": 1, "batch": 100, "beat": 10, "sd_en": False}
            )

@pytest.mark.asyncio
async def test_config_service_get_node_capabilities_method():
    # test get_node_capabilities method on ConfigService
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    mock_caps = NodeCapabilities(node_id="sensor_caps_check", accel_freqs=[100.0], enabled_ops=["CONV_2D"])
    service = ConfigService(session_manager=mock_db)
    
    with patch("app.services.config_service.get_node_capabilities", new=AsyncMock(return_value=mock_caps)):
        caps = await service.get_node_capabilities("sensor_caps_check")
        assert caps is not None
        assert caps.node_id == "sensor_caps_check"
        assert "CONV_2D" in caps.enabled_ops

@pytest.mark.asyncio
async def test_mqtt_publisher_publish_non_bytes_error():
    # test MQTTPublisher raises TypeError when non-bytes payload is provided
    pub = MQTTPublisher()
    with pytest.raises(TypeError, match="payload must be bytes-like"):
        await pub.publish("v1/test/topic", "string_instead_of_bytes") # Non-bytes

@pytest.mark.asyncio
async def test_mqtt_publisher_direct_publish_transient_client():
    # test MQTTPublisher.publish when no client is pre-connected
    pub = MQTTPublisher(broker="localhost", port=1883, username="user", password="pwd", client_id="client_123")
    mock_client = AsyncMock()
    
    with patch("aiomqtt.Client", return_value=mock_client):
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        
        await pub.publish("v1/direct/topic", b"binary_payload", qos=1, retain=False)
        mock_client.publish.assert_awaited_once_with(
            "v1/direct/topic",
            payload=b"binary_payload",
            qos=1,
            retain=False
        )

