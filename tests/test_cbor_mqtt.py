# Comprehensive tests for CBOR schemas, codec, and MQTT ingestion service
import os
import sys
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# ensure backend is in python path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.cbor.schemas import (
    EmitReason,
    StreamMode,
    ModelType,
    ArchitectureTag,
    LossMode,
    Datapoints,
    SegmentData,
    Segment,
    InferencePacket,
    NodeCapabilities,
    NodeHealthInfo,
    NodeAlert,
    Command,
    RuntimeConfig,
    RouteEntry,
    EnsembleConfig,
    BaseModelPackage,
    MemoryModelPackage,
    AutoencoderModelPackage,
    RouterModelPackage,
)
from app.cbor.codec import (
    CBORError,
    CBOREncodeError,
    CBORDecodeError,
    CBORValidationError,
    encode_segment,
    decode_segment,
    encode_inference_packet,
    decode_inference_packet,
    encode_node_capabilities,
    decode_node_capabilities,
    encode_node_health,
    decode_node_health,
    encode_runtime_config,
    decode_runtime_config,
    encode_ensemble_config,
    decode_ensemble_config,
    encode_model_package,
    decode_model_package,
    encode_payload,
    decode_payload,
    encode_node_alert,
    decode_node_alert,
    encode_command,
    decode_command,
    encode,
    decode,
)
from app.mqtt.consumer import MQTTConsumer
from app.database.models import RawTelemetry, InferenceResult, NodeHealth

# ==============================================================================
# 1. ENUM DEFINITIONS AND SCHEMA VALIDATION TESTS
# ==============================================================================

def test_emit_reason_enum_values():
    # verify emission reason tags
    assert EmitReason.CONTINUOUS == 1
    assert EmitReason.ANOMALY == 2
    assert EmitReason.PERIODIC == 3
    assert EmitReason.MANUAL_DUMP == 4

def test_stream_mode_enum_values():
    # verify stream mode tags
    assert StreamMode.CONTINUOUS == 1
    assert StreamMode.ANOMALY_ONLY == 2
    assert StreamMode.ANOMALY_OR_PERIODIC == 3
    assert StreamMode.CONTINUOUS_SCORE_RAW_ANOMALY == 4

def test_model_type_enum_values():
    # verify model archetype tags
    assert ModelType.AUTOENCODER == 1
    assert ModelType.ROUTER == 2
    assert ModelType.MEMORY == 3

def test_architecture_tag_values():
    # verify architecture tag mapping
    assert ArchitectureTag.VA == 1
    assert ArchitectureTag.CA_1D == 2
    assert ArchitectureTag.CA_2D == 3
    assert ArchitectureTag.CLSTM == 4
    assert ArchitectureTag.DA == 5
    assert ArchitectureTag.FITS == 6
    assert ArchitectureTag.TN == 7
    assert ArchitectureTag.DENSE_R == 10
    assert ArchitectureTag.STFT_MCNN == 12
    assert ArchitectureTag.ONE_D_CNN_R == 13
    assert ArchitectureTag.LSTM_MEM == 20

def test_loss_mode_enum_values():
    # verify loss mode tags
    assert LossMode.LOG_MSE == 1
    assert LossMode.LINEAR_MSE == 2

# ==============================================================================
# 2. CBOR ROUNDTRIP SERIALIZATION / DESERIALIZATION TESTS
# ==============================================================================

def test_segment_cbor_roundtrip():
    # test Segment serialization and deserialization
    seg = Segment(
        id=1001,
        rate=833.0,
        reason=EmitReason.CONTINUOUS,
        chunk=1,
        data=SegmentData(
            gyro=Datapoints(x=[0.1, 0.2, 0.3], y=[0.4, 0.5, 0.6], z=[0.7, 0.8, 0.9]),
            accel=Datapoints(x=[1.1, 1.2, 1.3], y=[1.4, 1.5, 1.6], z=[1.7, 1.8, 1.9])
        )
    )
    
    encoded = encode_segment(seg)
    assert isinstance(encoded, bytes)
    assert len(encoded) > 0
    
    decoded = decode_segment(encoded)
    assert decoded.id == 1001
    assert decoded.rate == 833.0
    assert decoded.reason == 1
    assert decoded.chunk == 1
    assert len(decoded.data.gyro.x) == 3
    assert decoded.data.accel.x[0] == pytest.approx(1.1, rel=1e-5)
    assert decoded.data.gyro.z[2] == pytest.approx(0.9, rel=1e-5)

def test_segment_full_512_points_roundtrip():
    # test Segment with full 512 points per axis
    points_x = [float(i) * 0.01 for i in range(512)]
    points_y = [float(i) * 0.02 for i in range(512)]
    points_z = [float(i) * 0.03 for i in range(512)]
    
    seg = Segment(
        id=42,
        rate=1666.0,
        reason=EmitReason.ANOMALY,
        chunk=2,
        data=SegmentData(
            gyro=Datapoints(x=points_x, y=points_y, z=points_z),
            accel=Datapoints(x=points_x, y=points_y, z=points_z)
        )
    )
    
    encoded = encode_segment(seg)
    decoded = decode_segment(encoded)
    assert decoded.id == 42
    assert decoded.chunk == 2
    assert len(decoded.data.accel.x) == 512
    assert len(decoded.data.gyro.y) == 512
    assert decoded.data.accel.x[511] == pytest.approx(5.11, rel=1e-4)

def test_inference_packet_cbor_roundtrip():
    # test InferencePacket roundtrip
    pkt = InferencePacket(
        id=505,
        reason=EmitReason.ANOMALY,
        r_m_id=201,
        ae_id=101,
        mse=0.0845,
        anom=True
    )
    
    encoded = encode_inference_packet(pkt)
    assert isinstance(encoded, bytes)
    
    decoded = decode_inference_packet(encoded)
    assert decoded.id == 505
    assert decoded.reason == 2
    assert decoded.r_m_id == 201
    assert decoded.ae_id == 101
    assert decoded.mse == pytest.approx(0.0845, rel=1e-5)
    assert decoded.anom is True

def test_node_capabilities_cbor_roundtrip():
    # test NodeCapabilities roundtrip
    caps = NodeCapabilities(
        accel_freqs=[12.5, 26.0, 52.0, 104.0, 208.0, 416.0, 833.0, 1666.0, 3332.0, 6667.0],
        gyro_freqs=[12.5, 26.0, 52.0, 104.0, 208.0, 416.0, 833.0, 1666.0, 3332.0, 6667.0],
        enabled_ops=["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "RESHAPE", "SOFTMAX"]
    )
    
    encoded = encode_node_capabilities(caps)
    decoded = decode_node_capabilities(encoded)
    assert len(decoded.accel_freqs) == 10
    assert len(decoded.gyro_freqs) == 10
    assert decoded.enabled_ops == ["CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "RESHAPE", "SOFTMAX"]

def test_node_health_info_cbor_roundtrip_full():
    # test NodeHealthInfo with all optional fields present
    caps = NodeCapabilities(
        accel_freqs=[100.0, 200.0],
        gyro_freqs=[100.0, 200.0],
        enabled_ops=["ABS", "ADD"]
    )
    health = NodeHealthInfo(
        ram=195000,
        sd=1,
        cache=[101, 102, 201],
        last=789,
        status="streaming_dump",
        caps=caps,
        dump_t=150,
        dump_r=False,
        ips=22.4
    )
    
    encoded = encode_node_health(health)
    decoded = decode_node_health(encoded)
    assert decoded.ram == 195000
    assert decoded.sd == 1
    assert decoded.cache == [101, 102, 201]
    assert decoded.last == 789
    assert decoded.status == "streaming_dump"
    assert decoded.caps is not None
    assert decoded.caps.enabled_ops == ["ABS", "ADD"]
    assert decoded.dump_t == 150
    assert decoded.dump_r is False
    assert decoded.ips == pytest.approx(22.4, rel=1e-4)

def test_node_health_info_cbor_roundtrip_minimal():
    # test NodeHealthInfo with optional fields omitted
    health = NodeHealthInfo(
        ram=180000,
        sd=0,
        cache=[],
        last=10
    )
    
    encoded = encode_node_health(health)
    decoded = decode_node_health(encoded)
    assert decoded.ram == 180000
    assert decoded.sd == 0
    assert decoded.cache == []
    assert decoded.last == 10
    assert decoded.status == "idle"
    assert decoded.caps is None
    assert decoded.dump_t is None

def test_runtime_config_cbor_roundtrip():
    # test RuntimeConfig roundtrip with cadence
    cfg = RuntimeConfig(
        rate=833.0,
        mode=StreamMode.ANOMALY_OR_PERIODIC,
        batch=512,
        cad=10,
        beat=15,
        sd_en=True
    )
    
    encoded = encode_runtime_config(cfg)
    decoded = decode_runtime_config(encoded)
    assert decoded.rate == 833.0
    assert decoded.mode == 3
    assert decoded.batch == 512
    assert decoded.cad == 10
    assert decoded.beat == 15
    assert decoded.sd_en is True

def test_ensemble_config_cbor_roundtrip():
    # test EnsembleConfig roundtrip
    ens = EnsembleConfig(
        warmup=5,
        r_m_id=201,
        routes=[
            RouteEntry(out_ix=0, m_id=101),
            RouteEntry(out_ix=1, m_id=102),
            RouteEntry(out_ix=2, m_id=103)
        ],
        mem_id=301
    )
    
    encoded = encode_ensemble_config(ens)
    decoded = decode_ensemble_config(encoded)
    assert decoded.warmup == 5
    assert decoded.r_m_id == 201
    assert len(decoded.routes) == 3
    assert decoded.routes[0].out_ix == 0
    assert decoded.routes[0].m_id == 101
    assert decoded.routes[2].m_id == 103
    assert decoded.mem_id == 301

def test_autoencoder_model_package_roundtrip():
    # test AutoencoderModelPackage roundtrip
    pkg = AutoencoderModelPackage(
        m_id=101,
        tag=ArchitectureTag.CA_1D,
        data=b"\x00\x01\x02\x03fake_tflite_ae_weights",
        accel_bins=128,
        gyro_bins=128,
        tsteps=16,
        mem_d=32,
        limit=0.045,
        loss=LossMode.LOG_MSE,
        skip=2
    )
    
    encoded = encode_model_package(pkg)
    decoded = decode_model_package(encoded)
    assert isinstance(decoded, AutoencoderModelPackage)
    assert decoded.m_id == 101
    assert decoded.type == 1
    assert decoded.tag == 2
    assert decoded.data == b"\x00\x01\x02\x03fake_tflite_ae_weights"
    assert decoded.accel_bins == 128
    assert decoded.limit == pytest.approx(0.045, rel=1e-5)
    assert decoded.loss == 1
    assert decoded.skip == 2

def test_router_model_package_roundtrip():
    # test RouterModelPackage roundtrip
    pkg = RouterModelPackage(
        m_id=201,
        tag=ArchitectureTag.DENSE_R,
        data=b"router_tflite_binary",
        accel_bins=64,
        gyro_bins=64,
        tsteps=16,
        mem_d=None,
        class_count=3
    )
    
    encoded = encode_model_package(pkg)
    decoded = decode_model_package(encoded)
    assert isinstance(decoded, RouterModelPackage)
    assert decoded.m_id == 201
    assert decoded.type == 2
    assert decoded.tag == 10
    assert decoded.data == b"router_tflite_binary"
    assert decoded.class_count == 3

def test_memory_model_package_roundtrip():
    # test MemoryModelPackage roundtrip
    pkg = MemoryModelPackage(
        m_id=301,
        tag=ArchitectureTag.LSTM_MEM,
        data=b"memory_lstm_binary",
        accel_bins=128,
        gyro_bins=128,
        outdim=32,
        state=32
    )
    
    encoded = encode_model_package(pkg)
    decoded = decode_model_package(encoded)
    assert isinstance(decoded, MemoryModelPackage)
    assert decoded.m_id == 301
    assert decoded.type == 3
    assert decoded.tag == 20
    assert decoded.outdim == 32
    assert decoded.state == 32

def test_payload_union_polymorphic_dispatch():
    # test payload union generic encoder and decoder
    ens = EnsembleConfig(
        warmup=3,
        r_m_id=201,
        routes=[RouteEntry(out_ix=0, m_id=101)]
    )
    enc_ens = encode_payload(ens)
    dec_ens = decode_payload(enc_ens)
    assert isinstance(dec_ens, EnsembleConfig)
    assert dec_ens.warmup == 3
    
    ae = AutoencoderModelPackage(
        m_id=101,
        data=b"ae_data",
        accel_bins=128,
        gyro_bins=128,
        tsteps=16,
        limit=0.05,
        loss=1
    )
    enc_ae = encode_payload(ae)
    dec_ae = decode_payload(enc_ae)
    assert isinstance(dec_ae, AutoencoderModelPackage)
    assert dec_ae.m_id == 101

def test_node_alert_and_command_roundtrip():
    # test alert and command messages
    alert = NodeAlert(code=404, detail="Submodel file not found on SD")
    enc_alert = encode_node_alert(alert)
    dec_alert = decode_node_alert(enc_alert)
    assert dec_alert.code == 404
    assert dec_alert.detail == "Submodel file not found on SD"
    
    cmd = Command(dump=True, dump_i=False, reboot=False)
    enc_cmd = encode_command(cmd)
    dec_cmd = decode_command(enc_cmd)
    assert dec_cmd.dump is True
    assert dec_cmd.reboot is False

def test_generic_encode_decode_dispatcher():
    # test top level generic encode and decode functions
    cfg = RuntimeConfig(rate=100.0, mode=1, batch=256, beat=5, sd_en=False)
    enc = encode(cfg)
    dec = decode(enc, RuntimeConfig)
    assert dec.rate == 100.0
    assert dec.sd_en is False

# ==============================================================================
# 3. ADVERSARIAL AND MALFORMED CBOR PAYLOAD TESTS
# ==============================================================================

def test_corrupt_cbor_byte_stream_decode_error():
    # test decoding empty or truncated binary stream
    with pytest.raises(CBORDecodeError):
        decode_segment(b"")
    
    with pytest.raises(CBORDecodeError):
        decode_inference_packet(b"\xa5\x01\x02") # truncated map
    
    with pytest.raises(CBORDecodeError):
        decode_node_capabilities(b"\x1b\x00\x00") # truncated 64-bit uint
    
    with pytest.raises(CBORValidationError):
        decode_node_capabilities(b"\xff\xff\x00\x00corrupt") # non-map simple values

def test_invalid_cbor_root_type():
    # test when root is not a map
    invalid_bytes = encode([1, 2, 3]) # array instead of map
    with pytest.raises(CBORValidationError):
        decode_segment(invalid_bytes)
    with pytest.raises(CBORValidationError):
        decode_inference_packet(invalid_bytes)
    with pytest.raises(CBORValidationError):
        decode_node_capabilities(invalid_bytes)

def test_missing_required_fields():
    # test when required keys are missing
    import cbor2
    incomplete_seg = cbor2.dumps({"id": 1, "rate": 100.0}) # missing reason, chunk, data
    with pytest.raises(CBORValidationError):
        decode_segment(incomplete_seg)
    
    incomplete_pkt = cbor2.dumps({"id": 1, "reason": 1}) # missing r_m_id, ae_id, etc.
    with pytest.raises(CBORValidationError):
        decode_inference_packet(incomplete_pkt)

def test_invalid_enum_ranges():
    # test invalid emit reason or stream mode
    import cbor2
    bad_reason_seg = cbor2.dumps({
        "id": 1,
        "rate": 100.0,
        "reason": 99, # invalid enum
        "chunk": 1,
        "data": {"gyro": {"x":[],"y":[],"z":[]}, "accel": {"x":[],"y":[],"z":[]}}
    })
    with pytest.raises(CBORValidationError):
        decode_segment(bad_reason_seg)
    
    bad_mode_cfg = cbor2.dumps({
        "rate": 100.0,
        "mode": 0, # invalid stream mode
        "batch": 256,
        "beat": 5,
        "sd_en": True
    })
    with pytest.raises(CBORValidationError):
        decode_runtime_config(bad_mode_cfg)

def test_ensemble_routes_constraints():
    # test empty routes and routes > 16
    with pytest.raises(CBORValidationError):
        encode_ensemble_config(EnsembleConfig(warmup=1, r_m_id=1, routes=[]))
    
    routes_17 = [RouteEntry(out_ix=i, m_id=100+i) for i in range(17)]
    with pytest.raises(CBORValidationError):
        encode_ensemble_config(EnsembleConfig(warmup=1, r_m_id=1, routes=routes_17))

# ==============================================================================
# 4. ASYNC MQTT INGESTION CONSUMER AND DB PERSISTENCE TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_mqtt_consumer_caps_registration():
    # test node capabilities handling and registration
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    consumer = MQTTConsumer(broker="localhost", session_manager=mock_db)
    
    caps = NodeCapabilities(
        accel_freqs=[12.5, 26.0, 52.0],
        gyro_freqs=[12.5, 26.0, 52.0],
        enabled_ops=["CONV_2D", "FULLY_CONNECTED"]
    )
    payload = encode_node_capabilities(caps)
    
    with patch("app.mqtt.consumer.register_node", new=AsyncMock()) as mock_reg, \
         patch("app.mqtt.consumer.upsert_node_capabilities", new=AsyncMock()) as mock_upsert:
        
        handled = await consumer.handle_message("v1/node_vibe_01/info/caps", payload)
        assert handled is True
        mock_reg.assert_awaited_once_with(mock_conn, node_id="node_vibe_01", status="registered")
        mock_upsert.assert_awaited_once_with(
            mock_conn,
            node_id="node_vibe_01",
            accel_freqs=caps.accel_freqs,
            gyro_freqs=caps.gyro_freqs,
            enabled_ops=caps.enabled_ops
        )

@pytest.mark.asyncio
async def test_mqtt_consumer_telemetry_ingestion():
    # test raw telemetry ingestion into hypertable
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    consumer = MQTTConsumer(broker="localhost", session_manager=mock_db)
    
    seg = Segment(
        id=2001,
        rate=400.0,
        reason=EmitReason.CONTINUOUS,
        chunk=1,
        data=SegmentData(
            gyro=Datapoints(x=[0.1]*512, y=[0.2]*512, z=[0.3]*512),
            accel=Datapoints(x=[1.1]*512, y=[1.2]*512, z=[1.3]*512)
        )
    )
    payload = encode_segment(seg)
    
    with patch("app.mqtt.consumer.insert_raw_telemetry", new=AsyncMock()) as mock_insert, \
         patch("app.mqtt.consumer.update_node_last_seen", new=AsyncMock()) as mock_seen:
        
        handled = await consumer.handle_message("v1/node_vibe_01/data/sensor", payload)
        assert handled is True
        
        mock_insert.assert_awaited_once()
        telem_arg: RawTelemetry = mock_insert.call_args[0][1]
        assert telem_arg.node_id == "node_vibe_01"
        assert telem_arg.segment_id == 2001
        assert telem_arg.chunk_id == 1
        assert telem_arg.sample_rate == 400.0
        assert telem_arg.emit_reason == 1
        assert len(telem_arg.accel_x) == 512
        assert telem_arg.raw_bytes_count == len(payload)
        
        mock_seen.assert_awaited_once_with(mock_conn, "node_vibe_01")

@pytest.mark.asyncio
async def test_mqtt_consumer_inference_ingestion():
    # test inference packet ingestion into hypertable
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    consumer = MQTTConsumer(broker="localhost", session_manager=mock_db)
    
    pkt = InferencePacket(
        id=2001,
        reason=EmitReason.ANOMALY,
        r_m_id=201,
        ae_id=101,
        mse=0.062,
        anom=True
    )
    payload = encode_inference_packet(pkt)
    
    with patch("app.mqtt.consumer.insert_inference_result", new=AsyncMock()) as mock_insert, \
         patch("app.mqtt.consumer.update_node_last_seen", new=AsyncMock()) as mock_seen:
        
        handled = await consumer.handle_message("v1/node_vibe_01/inference", payload)
        assert handled is True
        
        mock_insert.assert_awaited_once()
        res_arg: InferenceResult = mock_insert.call_args[0][1]
        assert res_arg.node_id == "node_vibe_01"
        assert res_arg.segment_id == 2001
        assert res_arg.router_model_id == 201
        assert res_arg.autoencoder_model_id == 101
        assert res_arg.mse == 0.062
        assert res_arg.anomaly is True
        
        mock_seen.assert_awaited_once_with(mock_conn, "node_vibe_01")

@pytest.mark.asyncio
async def test_mqtt_consumer_health_ingestion():
    # test health telemetry storage
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    consumer = MQTTConsumer(broker="localhost", session_manager=mock_db)
    
    health = NodeHealthInfo(
        ram=184000,
        sd=1,
        cache=[101, 102],
        last=50,
        status="idle",
        ips=14.2
    )
    payload = encode_node_health(health)
    
    with patch("app.mqtt.consumer.insert_node_health", new=AsyncMock()) as mock_insert, \
         patch("app.mqtt.consumer.update_node_last_seen", new=AsyncMock()) as mock_seen:
        
        handled = await consumer.handle_message("v1/node_vibe_01/info/health", payload)
        assert handled is True
        
        mock_insert.assert_awaited_once()
        h_arg: NodeHealth = mock_insert.call_args[0][1]
        assert h_arg.node_id == "node_vibe_01"
        assert h_arg.ram_free == 184000
        assert h_arg.sd_status == 1
        assert h_arg.cached_models == [101, 102]
        assert h_arg.last_segment_id == 50
        assert h_arg.status == "idle"
        assert h_arg.ips == 14.2
        
        mock_seen.assert_awaited_once_with(mock_conn, "node_vibe_01")

@pytest.mark.asyncio
async def test_mqtt_consumer_corrupt_payload_graceful_handling():
    # test that corrupt payloads do not crash the handler
    mock_db = MagicMock()
    consumer = MQTTConsumer(broker="localhost", session_manager=mock_db)
    
    handled = await consumer.handle_message("v1/node_01/data/sensor", b"\x00\xffcorrupt_bytes")
    assert handled is False

@pytest.mark.asyncio
async def test_mqtt_consumer_unhandled_topic():
    # test unhandled topic
    mock_db = MagicMock()
    consumer = MQTTConsumer(broker="localhost", session_manager=mock_db)
    
    handled = await consumer.handle_message("v1/node_01/unknown/path", b"\x01\x02")
    assert handled is False

@pytest.mark.asyncio
async def test_mqtt_consumer_subscription_topics_list():
    # test that subscription_topics covers all required endpoints
    consumer = MQTTConsumer(topic_prefix="v1")
    subs = consumer.subscription_topics
    assert "v1/+/info/caps" in subs
    assert "v1/+/data/sensor" in subs
    assert "v1/+/inference" in subs
    assert "v1/+/info/health" in subs

@pytest.mark.asyncio
async def test_mqtt_consumer_health_with_embedded_caps():
    # test health payload that includes embedded capabilities
    mock_db = MagicMock()
    mock_conn = AsyncMock()
    mock_db.acquire.return_value.__aenter__.return_value = mock_conn
    mock_db.acquire.return_value.__aexit__.return_value = None
    
    consumer = MQTTConsumer(broker="localhost", session_manager=mock_db)
    
    caps = NodeCapabilities(
        accel_freqs=[25.0, 50.0],
        gyro_freqs=[25.0, 50.0],
        enabled_ops=["CONV_2D", "MAX_POOL_2D"]
    )
    health = NodeHealthInfo(
        ram=200000,
        sd=1,
        cache=[101],
        last=10,
        status="idle",
        caps=caps,
        ips=10.0
    )
    payload = encode_node_health(health)
    
    with patch("app.mqtt.consumer.insert_node_health", new=AsyncMock()) as mock_health, \
         patch("app.mqtt.consumer.upsert_node_capabilities", new=AsyncMock()) as mock_caps, \
         patch("app.mqtt.consumer.update_node_last_seen", new=AsyncMock()) as mock_seen:
        
        handled = await consumer.handle_message("v1/node_embedded/info/health", payload)
        assert handled is True
        mock_health.assert_awaited_once()
        mock_caps.assert_awaited_once_with(
            mock_conn,
            node_id="node_embedded",
            accel_freqs=caps.accel_freqs,
            gyro_freqs=caps.gyro_freqs,
            enabled_ops=caps.enabled_ops
        )
        mock_seen.assert_awaited_once_with(mock_conn, "node_embedded")

@pytest.mark.asyncio
async def test_mqtt_consumer_alert_message():
    # test alert message handling
    mock_db = MagicMock()
    consumer = MQTTConsumer(broker="localhost", session_manager=mock_db)
    
    alert = NodeAlert(code=500, detail="Hardware I2C failure")
    payload = encode_node_alert(alert)
    
    handled = await consumer.handle_message("v1/node_alert_01/alert/critical", payload)
    assert handled is True

@pytest.mark.asyncio
async def test_mqtt_consumer_publish_unconnected_raises_error():
    # test publish raises error when client is not connected
    consumer = MQTTConsumer(broker="localhost")
    with pytest.raises(RuntimeError, match="MQTT client is not connected"):
        await consumer.publish("v1/test/topic", b"test_data")

@pytest.mark.asyncio
async def test_mqtt_consumer_publish_connected():
    # test publish delegates to client when connected
    consumer = MQTTConsumer(broker="localhost")
    mock_client = AsyncMock()
    consumer._client = mock_client
    
    await consumer.publish("v1/test/topic", b"test_payload", qos=1, retain=True)
    mock_client.publish.assert_awaited_once_with("v1/test/topic", b"test_payload", qos=1, retain=True)

def test_model_package_unsupported_type():
    # test unsupported type tag in model package decoding
    import cbor2
    invalid_pkg = cbor2.dumps({"m_id": 1, "type": 99, "data": b"bytes"})
    with pytest.raises(CBORValidationError, match="Unsupported model type tag: 99"):
        decode_model_package(invalid_pkg)

def test_model_package_missing_type():
    # test missing type in model package decoding
    import cbor2
    invalid_pkg = cbor2.dumps({"m_id": 1, "data": b"bytes"})
    with pytest.raises(CBORValidationError, match="Missing required 'type' field"):
        decode_model_package(invalid_pkg)

def test_payload_invalid_structure():
    # test payload that matches neither model nor ensemble
    import cbor2
    invalid_payload = cbor2.dumps({"unrelated": "field"})
    with pytest.raises(CBORValidationError, match="Payload is neither EnsembleConfig nor ModelPackage"):
        decode_payload(invalid_payload)

def test_datapoints_empty_lists():
    # test empty coordinate arrays in Datapoints
    dp = Datapoints(x=[], y=[], z=[])
    seg = Segment(
        id=1,
        rate=100.0,
        reason=1,
        chunk=1,
        data=SegmentData(gyro=dp, accel=dp)
    )
    enc = encode_segment(seg)
    dec = decode_segment(enc)
    assert dec.data.gyro.x == []
    assert dec.data.accel.z == []

def test_segment_invalid_reason_encode():
    # test validation error when encoding segment with invalid reason
    seg = Segment(
        id=1,
        rate=100.0,
        reason=99,
        chunk=1,
        data=SegmentData(gyro=Datapoints(), accel=Datapoints())
    )
    with pytest.raises(CBORValidationError, match="Invalid emit_reason"):
        encode_segment(seg)

def test_inference_packet_invalid_reason_encode():
    # test validation error when encoding inference packet with invalid reason
    pkt = InferencePacket(
        id=1,
        reason=0,
        r_m_id=1,
        ae_id=1,
        mse=0.1,
        anom=False
    )
    with pytest.raises(CBORValidationError, match="Invalid emit_reason"):
        encode_inference_packet(pkt)

def test_runtime_config_invalid_mode_encode():
    # test validation error when encoding runtime config with invalid mode
    cfg = RuntimeConfig(
        rate=100.0,
        mode=10,
        batch=256,
        beat=5,
        sd_en=True
    )
    with pytest.raises(CBORValidationError, match="Invalid StreamMode"):
        encode_runtime_config(cfg)

