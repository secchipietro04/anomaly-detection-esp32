# tier 2: config service validation logic tests (no db/mqtt required)
import pytest
from app.services.config_service import (
    validate_runtime_config,
    ConfigValidationError,
)
from app.cbor.schemas import RuntimeConfig, StreamMode
from app.database.models import NodeCapabilities


def test_valid_config_passes():
    cfg = {"mode": 1, "rate": 800.0, "beat": 30, "batch": 4, "sd_en": False}
    result = validate_runtime_config(cfg)
    assert result.mode == 1
    assert result.rate == 800.0


def test_invalid_mode_rejected():
    with pytest.raises(ConfigValidationError):
        validate_runtime_config({"mode": 0, "rate": 800.0, "beat": 30, "batch": 4, "sd_en": False})


def test_invalid_rate_rejected():
    with pytest.raises(ConfigValidationError):
        validate_runtime_config({"mode": 1, "rate": -1.0, "beat": 30, "batch": 4, "sd_en": False})


def test_unsupported_rate_rejected_by_caps():
    caps = NodeCapabilities(
        node_id="test",
        accel_freqs=[800.0, 1600.0],
        gyro_freqs=[800.0],
        enabled_ops=[]
    )
    with pytest.raises(ConfigValidationError, match="not supported"):
        validate_runtime_config({"mode": 1, "rate": 400.0, "beat": 30, "batch": 4, "sd_en": False}, capabilities=caps)


def test_supported_rate_passes_caps():
    caps = NodeCapabilities(node_id="test", accel_freqs=[800.0, 1600.0], gyro_freqs=[800.0], enabled_ops=[])
    result = validate_runtime_config({"mode": 2, "rate": 1600.0, "beat": 60, "batch": 8, "sd_en": True}, capabilities=caps)
    assert result.rate == 1600.0


def test_invalid_beat_rejected():
    with pytest.raises(ConfigValidationError):
        validate_runtime_config({"mode": 1, "rate": 800.0, "beat": 0, "batch": 4, "sd_en": False})


def test_invalid_batch_rejected():
    with pytest.raises(ConfigValidationError):
        validate_runtime_config({"mode": 1, "rate": 800.0, "beat": 30, "batch": -1, "sd_en": False})


def test_sd_en_not_bool_rejected():
    with pytest.raises(ConfigValidationError):
        validate_runtime_config({"mode": 1, "rate": 800.0, "beat": 30, "batch": 4, "sd_en": 1})
