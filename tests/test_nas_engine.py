# Comprehensive tests for Optuna NAS search, ClearML tracking, Op filtering, Memory penalty, and CBOR deployment
import os
import sys
import math
import struct
import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

# ensure backend is in python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.config import Settings, get_settings
from app.cbor.schemas import (
    ModelType,
    LossMode,
    ArchitectureTag,
    RouterModelPackage,
    AutoencoderModelPackage,
    MemoryModelPackage,
    EnsembleConfig,
    RouteEntry,
)
from app.cbor.codec import (
    encode_ensemble_config,
    decode_ensemble_config,
    encode_model_package,
    decode_model_package,
)
from app.nas.penalty import (
    calculate_memory_penalty,
    calculate_ensemble_size,
    evaluate_models_memory_penalty,
)
from app.nas.op_filter import (
    ALL_SUPPORTED_OPS,
    BUILTIN_CODE_TO_NAME,
    extract_tflite_ops,
    is_model_compatible,
    validate_candidate_architecture,
    create_tflite_binary,
)
from app.nas.clearml_tracker import (
    ClearMLTracker,
    get_clearml_tracker,
)
from app.nas.monitor import (
    get_accumulated_telemetry_bytes,
    get_latest_segment_id,
    check_volume_threshold,
    DataVolumeMonitor,
)
from app.nas.search import (
    NASSearchEngine,
    NASSearchResult,
    run_nas_search,
)
from app.nas.publisher import (
    package_ensemble,
    persist_ensemble_and_models,
    publish_ensemble_to_mqtt,
    deploy_nas_ensemble,
)

# ==============================================================================
# 1. MEMORY PENALTY TESTS (<4MB vs >4MB)
# ==============================================================================

def test_calculate_ensemble_size():
    # test calculating ensemble size
    sz = calculate_ensemble_size(router_size=1000, memory_size=2000, autoencoder_sizes=[3000, 4000])
    assert sz == 10000

    sz_no_mem = calculate_ensemble_size(router_size=5000, autoencoder_sizes=[10000])
    assert sz_no_mem == 15000

def test_memory_penalty_under_4mb():
    # zero penalty for sizes under or equal to 4MB (4,194,304 bytes)
    assert calculate_memory_penalty(0) == 0.0
    assert calculate_memory_penalty(1024 * 1024) == 0.0
    assert calculate_memory_penalty(2 * 1024 * 1024) == 0.0
    assert calculate_memory_penalty(4 * 1024 * 1024) == 0.0
    assert calculate_memory_penalty(4194304) == 0.0

def test_memory_penalty_over_4mb():
    # positive penalty scaling with excess size
    ram_limit = 4 * 1024 * 1024
    factor = 0.05

    # 5MB total size -> 1MB excess -> penalty = 0.05 * 1.0 = 0.05
    size_5mb = 5 * 1024 * 1024
    p_5mb = calculate_memory_penalty(size_5mb, ram_limit_bytes=ram_limit, lambda_factor=factor)
    assert pytest.approx(p_5mb, rel=1e-5) == 0.05

    # 6MB total size -> 2MB excess -> penalty = 0.05 * 2.0 = 0.10
    size_6mb = 6 * 1024 * 1024
    p_6mb = calculate_memory_penalty(size_6mb, ram_limit_bytes=ram_limit, lambda_factor=factor)
    assert pytest.approx(p_6mb, rel=1e-5) == 0.10

    # 8MB total size -> 4MB excess -> penalty = 0.05 * 4.0 = 0.20
    size_8mb = 8 * 1024 * 1024
    p_8mb = calculate_memory_penalty(size_8mb, ram_limit_bytes=ram_limit, lambda_factor=factor)
    assert pytest.approx(p_8mb, rel=1e-5) == 0.20

def test_memory_penalty_raw_byte_mode():
    # test penalty calculation in raw byte mode
    ram_limit = 4000
    p = calculate_memory_penalty(5000, ram_limit_bytes=ram_limit, lambda_factor=0.01, per_mb=False)
    assert pytest.approx(p, rel=1e-5) == 10.0

def test_evaluate_models_memory_penalty_helper():
    # test penalty evaluation directly from model objects or sizes
    m1 = 1 * 1024 * 1024
    m2 = 2 * 1024 * 1024
    assert evaluate_models_memory_penalty([m1, m2]) == 0.0

    m3 = 3 * 1024 * 1024
    # total 6MB -> excess 2MB -> penalty 0.10
    assert pytest.approx(evaluate_models_memory_penalty([m1, m2, m3]), rel=1e-5) == 0.10

# ==============================================================================
# 2. TFLITE OPERATOR EXTRACTION & ENABLED_OPS FILTER TESTS
# ==============================================================================

def test_all_supported_ops_master_table():
    # verify 118 ops compiled in firmware are present in master table
    assert len(ALL_SUPPORTED_OPS) == 118
    assert "CONV_2D" in ALL_SUPPORTED_OPS
    assert "FULLY_CONNECTED" in ALL_SUPPORTED_OPS
    assert "UNIDIRECTIONAL_SEQUENCE_LSTM" in ALL_SUPPORTED_OPS
    assert "RELU" in ALL_SUPPORTED_OPS
    assert "SOFTMAX" in ALL_SUPPORTED_OPS
    assert "SUB" in ALL_SUPPORTED_OPS
    assert "SQUARE" in ALL_SUPPORTED_OPS
    assert "MEAN" in ALL_SUPPORTED_OPS

def test_tflite_binary_generator_and_extractor():
    # test generation of valid flatbuffer and parsing of opcodes
    expected_ops = ["CONV_2D", "FULLY_CONNECTED", "UNIDIRECTIONAL_SEQUENCE_LSTM", "RELU", "SOFTMAX"]
    tflite_buf = create_tflite_binary(expected_ops)
    
    assert isinstance(tflite_buf, bytes)
    assert len(tflite_buf) > 32
    assert tflite_buf[4:8] == b"TFL3"

    extracted = extract_tflite_ops(tflite_buf)
    for op in expected_ops:
        assert op in extracted

def test_tflite_custom_op_handling():
    # test custom op handling in flatbuffer
    custom_ops = ["CUSTOM_FFT_VIB", "FULLY_CONNECTED"]
    tflite_buf = create_tflite_binary(custom_ops)
    extracted = extract_tflite_ops(tflite_buf)
    assert "CUSTOM_FFT_VIB" in extracted
    assert "FULLY_CONNECTED" in extracted

def test_validate_candidate_architecture_compliant():
    # test valid architecture where all ops are present in node enabled_ops
    enabled = ["CONV_2D", "FULLY_CONNECTED", "RELU", "SOFTMAX", "SUB", "SQUARE", "MEAN"]
    candidate = ["FULLY_CONNECTED", "RELU", "SOFTMAX"]
    is_valid, missing = validate_candidate_architecture(candidate, enabled)
    assert is_valid is True
    assert len(missing) == 0

def test_validate_candidate_architecture_incompatible():
    # test invalid architecture where ops are missing
    enabled = ["FULLY_CONNECTED", "RELU", "SOFTMAX"]
    candidate = ["CONV_2D", "FULLY_CONNECTED", "UNIDIRECTIONAL_SEQUENCE_LSTM"]
    is_valid, missing = validate_candidate_architecture(candidate, enabled)
    assert is_valid is False
    assert "CONV_2D" in missing
    assert "UNIDIRECTIONAL_SEQUENCE_LSTM" in missing

def test_is_model_compatible_helper():
    # test helper function with raw flatbuffer bytes and op lists
    enabled = ["FULLY_CONNECTED", "RELU", "SOFTMAX"]
    tflite_valid = create_tflite_binary(["FULLY_CONNECTED", "RELU"])
    tflite_invalid = create_tflite_binary(["CONV_2D", "FULLY_CONNECTED"])

    assert is_model_compatible(tflite_valid, enabled) is True
    assert is_model_compatible(tflite_invalid, enabled) is False
    assert is_model_compatible(["FULLY_CONNECTED"], enabled) is True
    assert is_model_compatible(["TRANSPOSE_CONV"], enabled) is False

# ==============================================================================
# 3. CLEARML EXPERIMENT LOGGER & FALLBACK TESTS
# ==============================================================================

def test_clearml_tracker_offline_fallback():
    # verify tracker initializes cleanly in offline fallback mode without crashing
    tracker = ClearMLTracker(
        project_name="test_project",
        task_name="test_nas_task",
        force_offline=True
    )
    assert tracker.is_offline is True
    assert tracker.task is None

    # log parameters
    tracker.log_parameters({"lr": 0.001, "batch_size": 32})
    assert tracker._logged_parameters["lr"] == 0.001

    # log scalars
    tracker.log_scalar("Train", "loss", 0.123, iteration=1)
    tracker.log_scalar("Train", "loss", 0.098, iteration=2)
    scalars = tracker.get_logged_scalars()
    assert len(scalars) == 2
    assert scalars[0]["value"] == 0.123

    # log trial
    tracker.log_trial(
        trial_id=1,
        params={"hidden": 64},
        val_loss=0.045,
        memory_penalty=0.0,
        total_score=0.045
    )
    trials = tracker.get_logged_trials()
    assert len(trials) == 1
    assert trials[0]["trial_id"] == 1
    assert trials[0]["val_loss"] == 0.045

    # log artifact
    tracker.log_artifact(name="model.tflite", artifact=b"TFL3_dummy_bytes")
    artifacts = tracker.get_logged_artifacts()
    assert "model.tflite" in artifacts
    assert artifacts["model.tflite"]["data"] == b"TFL3_dummy_bytes"

    tracker.close()

def test_clearml_tracker_with_mocked_server():
    # test tracker interaction when clearml server is available
    mock_task = MagicMock()
    mock_logger = MagicMock()
    mock_task.get_logger.return_value = mock_logger

    with patch("clearml.Task.init", return_value=mock_task):
        tracker = ClearMLTracker(
            project_name="mock_proj",
            task_name="mock_task",
            force_offline=False
        )
        assert tracker.is_offline is False
        assert tracker.task == mock_task

        tracker.log_parameters({"alpha": 0.5})
        mock_task.connect.assert_called()

        tracker.log_scalar("Loss", "val", 0.05, iteration=10)
        mock_logger.report_scalar.assert_called_with(
            title="Loss",
            series="val",
            value=0.05,
            iteration=10
        )

        tracker.close()
        mock_task.close.assert_called_once()

# ==============================================================================
# 4. TELEMETRY DATA VOLUME TRACKER & 1MB TRIGGER TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_get_accumulated_telemetry_bytes():
    # test querying accumulated bytes from DB
    mock_conn = AsyncMock()
    mock_conn.fetchval.return_value = 1500000

    vol = await get_accumulated_telemetry_bytes(mock_conn, "node_01")
    assert vol == 1500000
    mock_conn.fetchval.assert_called_once()

    # with since_segment_id
    mock_conn.fetchval.reset_mock()
    mock_conn.fetchval.return_value = 500000
    vol_since = await get_accumulated_telemetry_bytes(mock_conn, "node_01", since_segment_id=100)
    assert vol_since == 500000

@pytest.mark.asyncio
async def test_check_volume_threshold_below_and_above():
    # test 1MB threshold check (1,048,576 bytes)
    mock_conn = AsyncMock()

    # case 1: 500 KB -> below threshold -> no trigger
    mock_conn.fetchval.return_value = 500 * 1024
    should_trigger, volume = await check_volume_threshold(mock_conn, "node_01", threshold_bytes=1048576)
    assert should_trigger is False
    assert volume == 512000

    # case 2: Exactly 1MB -> no trigger (must be > 1MB)
    mock_conn.fetchval.return_value = 1048576
    should_trigger, volume = await check_volume_threshold(mock_conn, "node_01", threshold_bytes=1048576)
    assert should_trigger is False
    assert volume == 1048576

    # case 3: 1MB + 100 bytes -> above threshold -> trigger!
    mock_conn.fetchval.return_value = 1048576 + 100
    should_trigger, volume = await check_volume_threshold(mock_conn, "node_01", threshold_bytes=1048576)
    assert should_trigger is True
    assert volume == 1048676

@pytest.mark.asyncio
async def test_data_volume_monitor_state_and_trigger():
    # test stateful monitor lifecycle
    mock_conn = AsyncMock()
    callback_called = False
    triggered_node = None
    triggered_vol = 0

    async def mock_callback(node_id: str, volume: int):
        nonlocal callback_called, triggered_node, triggered_vol
        callback_called = True
        triggered_node = node_id
        triggered_vol = volume

    monitor = DataVolumeMonitor(threshold_bytes=1048576, on_trigger_callback=mock_callback)

    # 1. below threshold
    mock_conn.fetchval.return_value = 800000
    res = await monitor.check_and_trigger(mock_conn, "sensor_alpha")
    assert res is False
    assert callback_called is False

    # 2. above threshold -> triggers callback
    mock_conn.fetchval.return_value = 1200000
    res = await monitor.check_and_trigger(mock_conn, "sensor_alpha")
    assert res is True
    assert callback_called is True
    assert triggered_node == "sensor_alpha"
    assert triggered_vol == 1200000

    # 3. second call while in progress -> does not re-trigger
    callback_called = False
    res = await monitor.check_and_trigger(mock_conn, "sensor_alpha")
    assert res is False
    assert callback_called is False

    # 4. mark trained up to segment 50 -> in progress cleared and checkpoint updated
    monitor.mark_trained("sensor_alpha", segment_id=50)
    assert monitor.get_last_trained_segment("sensor_alpha") == 50

# ==============================================================================
# 5. OPTUNA MULTI-MODEL NAS SEARCH ENGINE TESTS
# ==============================================================================

def test_nas_search_loop_execution():
    # test complete Optuna search loop generating valid result
    tracker = ClearMLTracker(force_offline=True)
    engine = NASSearchEngine(
        node_id="test_node_nas",
        enabled_ops=list(ALL_SUPPORTED_OPS),
        accel_bins=64,
        gyro_bins=64,
        tsteps=8,
        num_routes=2,
        n_trials=8,
        tracker=tracker,
        include_memory_model=True
    )

    result = engine.search()

    assert isinstance(result, NASSearchResult)
    assert result.best_trial_number >= 0
    assert result.best_score > 0
    assert isinstance(result.best_params, dict)
    
    # check router package
    assert isinstance(result.router_model, RouterModelPackage)
    assert result.router_model.type == int(ModelType.ROUTER)
    assert len(result.router_model.data) > 0
    assert result.router_model.class_count == 2

    # check memory package
    assert isinstance(result.memory_model, MemoryModelPackage)
    assert result.memory_model.type == int(ModelType.MEMORY)
    assert len(result.memory_model.data) > 0

    # check autoencoder packages
    assert len(result.autoencoder_models) == 2
    for ae in result.autoencoder_models:
        assert isinstance(ae, AutoencoderModelPackage)
        assert ae.type == int(ModelType.AUTOENCODER)
        assert len(ae.data) > 0
        assert ae.limit > 0

    # check ensemble config
    assert isinstance(result.ensemble_config, EnsembleConfig)
    assert result.ensemble_config.r_m_id == result.router_model.m_id
    assert result.ensemble_config.mem_id == result.memory_model.m_id
    assert len(result.ensemble_config.routes) == 2
    assert result.ensemble_config.routes[0].out_ix == 0
    assert result.ensemble_config.routes[0].m_id == result.autoencoder_models[0].m_id
    assert result.ensemble_config.routes[1].out_ix == 1
    assert result.ensemble_config.routes[1].m_id == result.autoencoder_models[1].m_id

    # check ClearML logging occurred
    assert len(tracker.get_logged_trials()) >= 8

def test_nas_search_with_op_constraints_pruning():
    # test that search engine strictly respects enabled_ops and prunes unsupported architectures
    restricted_ops = [
        "FULLY_CONNECTED",
        "RELU",
        "SOFTMAX",
        "SUB",
        "SQUARE",
        "MEAN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "TANH",
        "LOGISTIC",
        "ADD",
        "MUL",
    ]  # note: CONV_2D and TRANSPOSE_CONV are omitted!

    engine = NASSearchEngine(
        node_id="restricted_node",
        enabled_ops=restricted_ops,
        accel_bins=32,
        gyro_bins=32,
        tsteps=4,
        num_routes=2,
        n_trials=10,
        tracker=ClearMLTracker(force_offline=True)
    )

    result = engine.search()

    # verify all generated model ops are strictly within restricted_ops
    router_ops = extract_tflite_ops(result.router_model.data)
    for op in router_ops:
        assert op in restricted_ops, f"Router op '{op}' not in enabled ops!"

    for i, ae in enumerate(result.autoencoder_models):
        ae_ops = extract_tflite_ops(ae.data)
        for op in ae_ops:
            assert op in restricted_ops, f"AE op '{op}' not in enabled ops!"

def test_nas_search_with_telemetry_samples():
    # test NAS search guided by real/synthetic vibration data array
    synthetic_samples = np.random.randn(256).astype(np.float32)
    result = run_nas_search(
        node_id="sensor_vibration_node",
        enabled_ops=list(ALL_SUPPORTED_OPS),
        n_trials=5,
        telemetry_samples=synthetic_samples
    )
    assert result.validation_loss > 0
    assert result.best_score > 0

# ==============================================================================
# 6. CBOR PACKAGING AND MQTT DEPLOYMENT TESTS
# ==============================================================================

def test_cbor_ensemble_and_model_packaging():
    # test packaging NAS search results into valid CBOR payloads
    result = run_nas_search(
        node_id="cbor_pack_node",
        enabled_ops=list(ALL_SUPPORTED_OPS),
        n_trials=4
    )

    ens_config, models = package_ensemble(result, node_id="cbor_pack_node", warmup=15)
    assert ens_config.warmup == 15
    assert len(models) >= 3  # router + 2 AEs (+ optional memory)

    # test CBOR encode and decode roundtrip for EnsembleConfig
    ens_cbor = encode_ensemble_config(ens_config)
    assert isinstance(ens_cbor, bytes)
    assert len(ens_cbor) > 0
    decoded_ens = decode_ensemble_config(ens_cbor)
    assert decoded_ens.warmup == 15
    assert decoded_ens.r_m_id == ens_config.r_m_id
    assert len(decoded_ens.routes) == len(ens_config.routes)

    # test CBOR encode and decode roundtrip for each model package
    for m in models:
        m_cbor = encode_model_package(m)
        assert isinstance(m_cbor, bytes)
        decoded_m = decode_model_package(m_cbor)
        assert decoded_m.m_id == m.m_id
        assert decoded_m.type == m.type
        assert decoded_m.data == m.data

@pytest.mark.asyncio
async def test_persist_ensemble_and_models():
    # test persistence to TimescaleDB models and ensembles tables
    mock_conn = AsyncMock()
    mock_conn.fetchval.side_effect = [1001, 1002, 1003, 1004, 50001]

    result = run_nas_search(
        node_id="db_persist_node",
        enabled_ops=list(ALL_SUPPORTED_OPS),
        n_trials=3
    )
    ens_config, models = package_ensemble(result, node_id="db_persist_node")

    ens_id, model_ids = await persist_ensemble_and_models(
        conn=mock_conn,
        node_id="db_persist_node",
        ensemble=ens_config,
        models=models,
        ensemble_id=9999
    )

    assert ens_id == 9999
    assert len(model_ids) == len(models)
    assert mock_conn.fetchval.call_count >= len(models) + 1

@pytest.mark.asyncio
async def test_publish_ensemble_to_mqtt():
    # test publishing ensemble and model CBOR packages via MQTT
    mock_publisher = AsyncMock()
    mock_publisher.topic_prefix = "v1"
    mock_publisher.publish = AsyncMock()

    result = run_nas_search(
        node_id="mqtt_pub_node",
        enabled_ops=list(ALL_SUPPORTED_OPS),
        n_trials=3
    )
    ens_config, models = package_ensemble(result, node_id="mqtt_pub_node")

    res = await publish_ensemble_to_mqtt(
        publisher=mock_publisher,
        node_id="mqtt_pub_node",
        ensemble=ens_config,
        models=models,
        publish_models=True,
        qos=1
    )

    assert res["ensemble_published"] is True
    assert res["ensemble_topic"] == "v1/mqtt_pub_node/ensemble"
    assert res["models_published_count"] == len(models)
    # verify publish was called for ensemble topic + each model topic
    assert mock_publisher.publish.call_count == 1 + len(models)

@pytest.mark.asyncio
async def test_full_deploy_nas_ensemble_pipeline():
    # test end-to-end deploy_nas_ensemble helper
    mock_db_manager = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetchval.side_effect = [2001, 2002, 2003, 2004, 8888]

    # async context manager mock for db_manager.acquire()
    class MockAcquire:
        async def __aenter__(self):
            return mock_conn
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_db_manager.acquire.return_value = MockAcquire()

    mock_publisher = AsyncMock()
    mock_publisher.topic_prefix = "v1"
    mock_publisher.publish = AsyncMock()

    result = run_nas_search(
        node_id="e2e_node",
        enabled_ops=list(ALL_SUPPORTED_OPS),
        n_trials=4
    )

    deploy_res = await deploy_nas_ensemble(
        node_id="e2e_node",
        search_result=result,
        session_manager=mock_db_manager,
        publisher=mock_publisher,
        warmup=12,
        publish_models=True
    )

    assert deploy_res["status"] == "deployed"
    assert "ensemble_id" in deploy_res
    assert len(deploy_res["model_ids"]) >= 3
    assert deploy_res["publish_result"]["ensemble_published"] is True
    assert mock_publisher.publish.call_count >= 4


# ==============================================================================
# 7. ADVERSARIAL AND BOUNDARY VALUE TESTS
# ==============================================================================

def test_nas_memory_penalty_exact_boundaries():
    # 4MB = 4,194,304 bytes
    assert calculate_memory_penalty(4194303) == 0.0
    assert calculate_memory_penalty(4194304) == 0.0
    
    # 1 byte over
    p_1byte = calculate_memory_penalty(4194305, lambda_factor=0.05)
    assert p_1byte > 0.0
    
    # Exactly 1MB over (5,242,880 bytes)
    p_1mb_over = calculate_memory_penalty(5242880, lambda_factor=0.05)
    assert pytest.approx(p_1mb_over, rel=1e-5) == 0.05

    # Exactly 2MB over (6,291,456 bytes)
    p_2mb_over = calculate_memory_penalty(6291456, lambda_factor=0.05)
    assert pytest.approx(p_2mb_over, rel=1e-5) == 0.10

    # Custom lambda factor (0.1)
    p_custom_lambda = calculate_memory_penalty(5242880, lambda_factor=0.1)
    assert pytest.approx(p_custom_lambda, rel=1e-5) == 0.10

@pytest.mark.asyncio
async def test_data_volume_threshold_exact_boundaries():
    # 1MB = 1,048,576 bytes
    mock_conn = AsyncMock()

    # 1 byte below
    mock_conn.fetchval.return_value = 1048575
    trigger, val = await check_volume_threshold(mock_conn, "test_node", threshold_bytes=1048576)
    assert trigger is False
    assert val == 1048575

    # Exact threshold
    mock_conn.fetchval.return_value = 1048576
    trigger, val = await check_volume_threshold(mock_conn, "test_node", threshold_bytes=1048576)
    assert trigger is False
    assert val == 1048576

    # 1 byte above
    mock_conn.fetchval.return_value = 1048577
    trigger, val = await check_volume_threshold(mock_conn, "test_node", threshold_bytes=1048576)
    assert trigger is True
    assert val == 1048577

def test_tflite_corrupt_flatbuffer_safety():
    # test corrupt and edge-case byte buffers
    assert extract_tflite_ops(b"") == []
    assert extract_tflite_ops(b"short") == []
    assert extract_tflite_ops(b"0123456789abcdef") == []
    assert extract_tflite_ops(bytes([0xFF] * 64)) == []

def test_cbor_ensemble_boundary_16_routes():
    # max allowed routes in cddl is 16
    routes = [RouteEntry(out_ix=i, m_id=2000 + i) for i in range(16)]
    ens = EnsembleConfig(warmup=5, r_m_id=1999, routes=routes)
    encoded = encode_ensemble_config(ens)
    decoded = decode_ensemble_config(encoded)
    assert len(decoded.routes) == 16
    assert decoded.routes[15].out_ix == 15
    assert decoded.routes[15].m_id == 2015

def test_nas_search_multiple_routes_architecture():
    # test search with 4 routes
    engine = NASSearchEngine(
        node_id="multi_route_node",
        enabled_ops=list(ALL_SUPPORTED_OPS),
        num_routes=4,
        n_trials=5,
        tracker=ClearMLTracker(force_offline=True)
    )
    result = engine.search()
    assert result.router_model.class_count == 4
    assert len(result.autoencoder_models) == 4
    assert len(result.ensemble_config.routes) == 4
    for i in range(4):
        assert result.ensemble_config.routes[i].out_ix == i
