# tests for database schema, models, session and config
import os
import sys
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# ensure backend is in python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.config import Settings, get_settings
from app.database.session import DatabaseSessionManager
from app.database.models import (
    Node,
    NodeCapabilities,
    NodeHealth,
    ModelPackage,
    EnsembleConfig,
    RawTelemetry,
    InferenceResult,
    register_node,
    get_node,
    get_all_nodes,
    update_node_last_seen,
    upsert_node_capabilities,
    get_node_capabilities,
    insert_node_health,
    get_latest_node_health,
    insert_raw_telemetry,
    insert_raw_telemetry_batch,
    get_raw_telemetry_for_segment,
    get_raw_telemetry_volume_bytes,
    insert_inference_result,
    get_inference_results,
    update_inference_recalculated,
    insert_model,
    get_model,
    insert_ensemble,
    get_latest_ensemble,
)

# test sql schema file
def test_init_sql_schema_content():
    # verify init.sql exists and contains all required DDL statements
    sql_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../database/init.sql"))
    assert os.path.exists(sql_path), f"init.sql not found at {sql_path}"
    
    with open(sql_path, "r") as f:
        sql = f.read()
    
    # TimescaleDB extension init
    assert "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;" in sql
    
    # Relational tables
    assert "CREATE TABLE IF NOT EXISTS nodes" in sql
    assert "CREATE TABLE IF NOT EXISTS node_capabilities" in sql
    assert "CREATE TABLE IF NOT EXISTS node_health" in sql
    assert "CREATE TABLE IF NOT EXISTS models" in sql
    assert "CREATE TABLE IF NOT EXISTS ensembles" in sql
    
    # Time-series tables
    assert "CREATE TABLE IF NOT EXISTS raw_telemetry" in sql
    assert "CREATE TABLE IF NOT EXISTS inference_results" in sql
    
    # Hypertables
    assert "SELECT create_hypertable('raw_telemetry', 'timestamp', if_not_exists => TRUE);" in sql
    assert "SELECT create_hypertable('inference_results', 'timestamp', if_not_exists => TRUE);" in sql
    
    # Indexes
    assert "idx_node_health_node_ts" in sql
    assert "idx_raw_telemetry_node_ts" in sql
    assert "idx_raw_telemetry_segment" in sql
    assert "idx_inference_results_node_ts" in sql
    assert "idx_inference_results_anomaly" in sql

# test configuration
def test_config_settings_defaults():
    # test default settings and computed dsn
    settings = Settings(
        POSTGRES_HOST="postgres_test",
        POSTGRES_PORT=5433,
        POSTGRES_USER="test_user",
        POSTGRES_PASSWORD="test_password",
        POSTGRES_DB="test_db"
    )
    assert settings.postgres_host == "postgres_test"
    assert settings.postgres_port == 5433
    assert settings.postgres_user == "test_user"
    assert settings.postgres_password == "test_password"
    assert settings.postgres_db == "test_db"
    assert settings.dsn == "postgresql://test_user:test_password@postgres_test:5433/test_db"
    assert settings.nas_data_threshold_bytes == 1048576
    assert settings.ram_limit_bytes == 4194304

def test_config_custom_database_url():
    # test explicit database url override
    custom_url = "postgresql://custom:secret@remotehost:9999/customdb"
    settings = Settings(DATABASE_URL=custom_url)
    assert settings.dsn == custom_url

# test data models serialization
def test_node_model():
    # test Node pydantic model
    now = datetime.now(timezone.utc)
    node = Node(node_id="node_001", name="sensor_vibe_1", registered_at=now, last_seen=now, status="active")
    assert node.node_id == "node_001"
    assert node.name == "sensor_vibe_1"
    assert node.status == "active"
    data = node.model_dump()
    assert data["node_id"] == "node_001"

def test_node_capabilities_model():
    # test NodeCapabilities pydantic model
    caps = NodeCapabilities(
        node_id="node_001",
        accel_freqs=[12.5, 26.0, 52.0],
        gyro_freqs=[12.5, 26.0, 52.0],
        enabled_ops=["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "RESHAPE"]
    )
    assert len(caps.accel_freqs) == 3
    assert len(caps.gyro_freqs) == 3
    assert "CONV_2D" in caps.enabled_ops
    data = caps.model_dump()
    assert data["node_id"] == "node_001"

def test_node_health_model():
    # test NodeHealth pydantic model
    health = NodeHealth(
        node_id="node_001",
        ram_free=180000,
        sd_status=1,
        cached_models=[101, 102],
        last_segment_id=42,
        status="streaming_dump",
        ips=15.5
    )
    assert health.node_id == "node_001"
    assert health.ram_free == 180000
    assert health.cached_models == [101, 102]
    assert health.ips == 15.5

def test_model_package_model():
    # test ModelPackage pydantic model
    pkg = ModelPackage(
        id=101,
        tag=1,
        model_type=1, # autoencoder
        tflite_binary=b"\x00\x01\x02\x03fake_tflite_model",
        size_bytes=24,
        config={"accel_bins": 3, "gyro_bins": 3, "tsteps": 32, "limit": 0.045}
    )
    assert pkg.id == 101
    assert pkg.model_type == 1
    assert pkg.tflite_binary.startswith(b"\x00\x01")
    assert pkg.config["limit"] == 0.045

def test_ensemble_config_model():
    # test EnsembleConfig pydantic model
    ensemble = EnsembleConfig(
        id=1,
        node_id="node_001",
        router_model_id=201,
        memory_model_id=301,
        warmup=5,
        routes=[{"out_ix": 0, "m_id": 101}, {"out_ix": 1, "m_id": 102}],
        total_size_bytes=3500000
    )
    assert ensemble.id == 1
    assert ensemble.router_model_id == 201
    assert len(ensemble.routes) == 2
    assert ensemble.total_size_bytes < 4194304 # within 4MB

def test_raw_telemetry_model():
    # test RawTelemetry pydantic model
    now = datetime.now(timezone.utc)
    telem = RawTelemetry(
        node_id="node_001",
        timestamp=now,
        segment_id=1001,
        chunk_id=1,
        sample_rate=200.0,
        emit_reason=1,
        accel_x=[0.1] * 512,
        accel_y=[0.2] * 512,
        accel_z=[0.3] * 512,
        gyro_x=[0.01] * 512,
        gyro_y=[0.02] * 512,
        gyro_z=[0.03] * 512,
        raw_bytes_count=12288
    )
    assert telem.node_id == "node_001"
    assert len(telem.accel_x) == 512
    assert len(telem.gyro_z) == 512
    assert telem.raw_bytes_count == 12288

def test_inference_result_model():
    # test InferenceResult pydantic model
    now = datetime.now(timezone.utc)
    res = InferenceResult(
        node_id="node_001",
        timestamp=now,
        segment_id=1001,
        emit_reason=2,
        router_model_id=201,
        autoencoder_model_id=101,
        mse=0.082,
        anomaly=True,
        is_recalculated=False
    )
    assert res.node_id == "node_001"
    assert res.segment_id == 1001
    assert res.mse == 0.082
    assert res.anomaly is True
    assert res.is_recalculated is False

# test session manager lifecycle
@pytest.mark.asyncio
async def test_database_session_manager_mock():
    # test pool creation, acquire and close with mocked asyncpg
    manager = DatabaseSessionManager()
    
    mock_pool = MagicMock()
    mock_pool._closed = False
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_pool.acquire.return_value.__aexit__.return_value = None
    mock_pool.close = AsyncMock()
    
    with patch("asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)):
        await manager.init("postgresql://test:test@localhost:5432/test")
        assert manager.is_connected() is True
        
        async with manager.acquire() as conn:
            assert conn == mock_conn
        
        mock_conn.execute.return_value = "SELECT 1"
        res = await manager.execute("SELECT 1")
        assert res == "SELECT 1"
        
        await manager.close()
        mock_pool.close.assert_awaited_once()

# test repository queries with mock connection
@pytest.mark.asyncio
async def test_register_node_query():
    # test register_node helper
    mock_conn = AsyncMock()
    now = datetime.now(timezone.utc)
    mock_conn.fetchrow.return_value = {
        "node_id": "node_test_123",
        "name": "vibration_sensor_alpha",
        "registered_at": now,
        "last_seen": now,
        "status": "registered"
    }
    
    node = await register_node(mock_conn, "node_test_123", "vibration_sensor_alpha")
    assert node.node_id == "node_test_123"
    assert node.name == "vibration_sensor_alpha"
    assert node.status == "registered"
    mock_conn.fetchrow.assert_awaited_once()

@pytest.mark.asyncio
async def test_upsert_capabilities_query():
    # test upsert_node_capabilities helper
    mock_conn = AsyncMock()
    now = datetime.now(timezone.utc)
    mock_conn.fetchrow.return_value = {
        "node_id": "node_test_123",
        "accel_freqs": [12.5, 26.0],
        "gyro_freqs": [12.5, 26.0],
        "enabled_ops": ["CONV_2D", "FULLY_CONNECTED"],
        "updated_at": now
    }
    
    caps = await upsert_node_capabilities(
        mock_conn,
        "node_test_123",
        [12.5, 26.0],
        [12.5, 26.0],
        ["CONV_2D", "FULLY_CONNECTED"]
    )
    assert caps.node_id == "node_test_123"
    assert len(caps.enabled_ops) == 2
    assert "CONV_2D" in caps.enabled_ops

@pytest.mark.asyncio
async def test_insert_node_health_query():
    # test insert_node_health helper
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = 1
    
    health = NodeHealth(
        node_id="node_test_123",
        ram_free=210000,
        sd_status=1,
        cached_models=[101],
        last_segment_id=50,
        status="idle",
        ips=12.0
    )
    health_id = await insert_node_health(mock_conn, health)
    assert health_id == 1
    mock_conn.fetchval.assert_awaited_once()

@pytest.mark.asyncio
async def test_insert_raw_telemetry_batch_query():
    # test insert_raw_telemetry_batch helper
    mock_conn = AsyncMock()
    
    now = datetime.now(timezone.utc)
    items = [
        RawTelemetry(
            node_id="node_test_123",
            timestamp=now,
            segment_id=1,
            chunk_id=1,
            sample_rate=200.0,
            emit_reason=1,
            accel_x=[0.0] * 512,
            accel_y=[0.0] * 512,
            accel_z=[0.0] * 512,
            gyro_x=[0.0] * 512,
            gyro_y=[0.0] * 512,
            gyro_z=[0.0] * 512,
            raw_bytes_count=6144
        ),
        RawTelemetry(
            node_id="node_test_123",
            timestamp=now,
            segment_id=1,
            chunk_id=2,
            sample_rate=200.0,
            emit_reason=1,
            accel_x=[0.0] * 512,
            accel_y=[0.0] * 512,
            accel_z=[0.0] * 512,
            gyro_x=[0.0] * 512,
            gyro_y=[0.0] * 512,
            gyro_z=[0.0] * 512,
            raw_bytes_count=6144
        )
    ]
    
    await insert_raw_telemetry_batch(mock_conn, items)
    mock_conn.executemany.assert_awaited_once()

@pytest.mark.asyncio
async def test_telemetry_volume_query():
    # test get_raw_telemetry_volume_bytes helper
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = 1048576 # 1MB
    
    vol = await get_raw_telemetry_volume_bytes(mock_conn, "node_test_123")
    assert vol == 1048576
    mock_conn.fetchval.assert_awaited_once()

@pytest.mark.asyncio
async def test_inference_recalculated_query():
    # test update_inference_recalculated helper
    mock_conn = AsyncMock()
    mock_conn.execute.return_value = "UPDATE 1"
    
    ok = await update_inference_recalculated(mock_conn, "node_test_123", 1001, 0.012, False)
    assert ok is True
    mock_conn.execute.assert_awaited_once()

@pytest.mark.asyncio
async def test_model_and_ensemble_queries():
    # test model and ensemble save/fetch helpers
    mock_conn = AsyncMock()
    now = datetime.now(timezone.utc)
    
    mock_conn.fetchval.return_value = 101
    model = ModelPackage(
        id=101,
        tag=1,
        model_type=1,
        tflite_binary=b"tflite_bytes",
        size_bytes=12,
        config={"tsteps": 32}
    )
    saved_id = await insert_model(mock_conn, model)
    assert saved_id == 101
    
    mock_conn.fetchrow.return_value = {
        "id": 101,
        "tag": 1,
        "model_type": 1,
        "tflite_binary": b"tflite_bytes",
        "size_bytes": 12,
        "config": json.dumps({"tsteps": 32}),
        "created_at": now
    }
    fetched_model = await get_model(mock_conn, 101)
    assert fetched_model is not None
    assert fetched_model.id == 101
    assert fetched_model.config["tsteps"] == 32

# additional edge case and boundary tests

@pytest.mark.asyncio
async def test_get_node_not_found():
    # test get_node when node does not exist
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None
    node = await get_node(mock_conn, "non_existent_node")
    assert node is None

@pytest.mark.asyncio
async def test_get_all_nodes_multiple():
    # test get_all_nodes returns list of Node objects
    mock_conn = AsyncMock()
    now = datetime.now(timezone.utc)
    mock_conn.fetch.return_value = [
        {"node_id": "node_1", "name": "sensor_1", "registered_at": now, "last_seen": now, "status": "active"},
        {"node_id": "node_2", "name": "sensor_2", "registered_at": now, "last_seen": now, "status": "idle"}
    ]
    nodes = await get_all_nodes(mock_conn)
    assert len(nodes) == 2
    assert nodes[0].node_id == "node_1"
    assert nodes[1].node_id == "node_2"

@pytest.mark.asyncio
async def test_update_node_last_seen():
    # test updating last_seen timestamp
    mock_conn = AsyncMock()
    await update_node_last_seen(mock_conn, "node_1")
    mock_conn.execute.assert_awaited_once_with(
        "UPDATE nodes SET last_seen = NOW() WHERE node_id = $1;",
        "node_1"
    )

@pytest.mark.asyncio
async def test_get_capabilities_not_found():
    # test get_node_capabilities when none exists
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None
    caps = await get_node_capabilities(mock_conn, "unknown_node")
    assert caps is None

@pytest.mark.asyncio
async def test_get_latest_health_not_found():
    # test get_latest_node_health when none exists
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None
    health = await get_latest_node_health(mock_conn, "unknown_node")
    assert health is None

@pytest.mark.asyncio
async def test_insert_single_raw_telemetry():
    # test insert_raw_telemetry helper
    mock_conn = AsyncMock()
    now = datetime.now(timezone.utc)
    telem = RawTelemetry(
        node_id="node_1",
        timestamp=now,
        segment_id=5,
        chunk_id=1,
        sample_rate=100.0,
        emit_reason=1,
        accel_x=[0.1] * 512,
        accel_y=[0.2] * 512,
        accel_z=[0.3] * 512,
        gyro_x=[0.01] * 512,
        gyro_y=[0.02] * 512,
        gyro_z=[0.03] * 512,
        raw_bytes_count=6144
    )
    await insert_raw_telemetry(mock_conn, telem)
    mock_conn.execute.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_raw_telemetry_for_segment_ordered():
    # test get_raw_telemetry_for_segment returns ordered chunks
    mock_conn = AsyncMock()
    now = datetime.now(timezone.utc)
    mock_conn.fetch.return_value = [
        {
            "id": 1, "node_id": "node_1", "timestamp": now, "segment_id": 5, "chunk_id": 1,
            "sample_rate": 100.0, "emit_reason": 1, "accel_x": [0.1]*512, "accel_y": [0.2]*512,
            "accel_z": [0.3]*512, "gyro_x": [0.01]*512, "gyro_y": [0.02]*512, "gyro_z": [0.03]*512,
            "raw_bytes_count": 6144, "created_at": now
        },
        {
            "id": 2, "node_id": "node_1", "timestamp": now, "segment_id": 5, "chunk_id": 2,
            "sample_rate": 100.0, "emit_reason": 1, "accel_x": [0.4]*512, "accel_y": [0.5]*512,
            "accel_z": [0.6]*512, "gyro_x": [0.04]*512, "gyro_y": [0.05]*512, "gyro_z": [0.06]*512,
            "raw_bytes_count": 6144, "created_at": now
        }
    ]
    chunks = await get_raw_telemetry_for_segment(mock_conn, "node_1", 5)
    assert len(chunks) == 2
    assert chunks[0].chunk_id == 1
    assert chunks[1].chunk_id == 2

@pytest.mark.asyncio
async def test_get_inference_results_pagination():
    # test get_inference_results with limit and offset
    mock_conn = AsyncMock()
    now = datetime.now(timezone.utc)
    mock_conn.fetch.return_value = [
        {
            "id": 10, "node_id": "node_1", "timestamp": now, "segment_id": 2, "emit_reason": 1,
            "router_model_id": 201, "autoencoder_model_id": 101, "mse": 0.005, "anomaly": False,
            "is_recalculated": False, "created_at": now
        }
    ]
    results = await get_inference_results(mock_conn, "node_1", limit=10, offset=5)
    assert len(results) == 1
    assert results[0].mse == 0.005
    mock_conn.fetch.assert_awaited_once()

@pytest.mark.asyncio
async def test_get_model_not_found():
    # test get_model when model does not exist
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None
    model = await get_model(mock_conn, 9999)
    assert model is None

@pytest.mark.asyncio
async def test_get_latest_ensemble_not_found():
    # test get_latest_ensemble when none exists
    mock_conn = AsyncMock()
    mock_conn.fetchrow.return_value = None
    ens = await get_latest_ensemble(mock_conn, "unknown_node")
    assert ens is None

@pytest.mark.asyncio
async def test_insert_and_get_ensemble():
    # test ensemble insertion and fetch with json routes
    mock_conn = AsyncMock()
    now = datetime.now(timezone.utc)
    mock_conn.fetchval.return_value = 501
    
    ensemble = EnsembleConfig(
        id=501,
        node_id="node_deploy_1",
        router_model_id=201,
        memory_model_id=None,
        warmup=10,
        routes=[{"out_ix": 0, "m_id": 101}],
        total_size_bytes=102400
    )
    ens_id = await insert_ensemble(mock_conn, ensemble)
    assert ens_id == 501
    
    mock_conn.fetchrow.return_value = {
        "id": 501,
        "node_id": "node_deploy_1",
        "router_model_id": 201,
        "memory_model_id": None,
        "warmup": 10,
        "routes": json.dumps([{"out_ix": 0, "m_id": 101}]),
        "total_size_bytes": 102400,
        "deployed_at": now
    }
    fetched = await get_latest_ensemble(mock_conn, "node_deploy_1")
    assert fetched is not None
    assert fetched.id == 501
    assert fetched.warmup == 10
    assert len(fetched.routes) == 1
