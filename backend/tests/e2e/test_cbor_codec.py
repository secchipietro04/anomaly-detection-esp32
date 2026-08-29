# tier 1: cbor encoding/decoding roundtrip tests
import pytest
from app.cbor.schemas import (
    RuntimeConfig, StreamMode, NodeCapabilities, NodeHealthInfo,
    InferencePacket, Segment, LossMode,
)
from app.cbor.codec import (
    encode_runtime_config, decode_runtime_config,
    encode_cbor, decode_cbor,
)


def test_runtime_config_roundtrip():
    cfg = RuntimeConfig(mode=int(StreamMode.CONTINUOUS), rate=800.0, beat=30, batch=4, sd_en=False)
    encoded = encode_runtime_config(cfg)
    assert isinstance(encoded, bytes)
    decoded = decode_runtime_config(encoded)
    assert decoded.mode == cfg.mode
    assert abs(decoded.rate - cfg.rate) < 1e-3
    assert decoded.beat == cfg.beat
    assert decoded.batch == cfg.batch
    assert decoded.sd_en == cfg.sd_en


def test_runtime_config_with_cadence():
    # ANOMALY_ONLY is mode=2
    cfg = RuntimeConfig(mode=int(StreamMode.ANOMALY_ONLY), rate=1600.0, beat=60, batch=8, sd_en=True, cad=5)
    encoded = encode_runtime_config(cfg)
    decoded = decode_runtime_config(encoded)
    assert decoded.cad == 5
    assert decoded.sd_en is True


def test_cbor_generic_encode_decode():
    data = {"node_id": "esp32-abc", "value": 42, "nested": [1, 2, 3]}
    encoded = encode_cbor(data)
    decoded = decode_cbor(encoded)
    assert decoded["node_id"] == "esp32-abc"
    assert decoded["value"] == 42
    assert decoded["nested"] == [1, 2, 3]


def test_runtime_config_invalid_mode():
    # pydantic won't raise, but the config service validator does
    from app.services.config_service import validate_runtime_config, ConfigValidationError
    with pytest.raises((ConfigValidationError, Exception)):
        validate_runtime_config({"mode": 99, "rate": 800.0, "beat": 30, "batch": 4, "sd_en": False})
