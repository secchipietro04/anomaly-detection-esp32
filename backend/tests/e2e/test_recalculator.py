# tier 4: local recalculator unit tests (no db, mocked data)
import numpy as np
import pytest
from app.api.recalculator import _build_input_window, _run_tflite_inference, FREQ_BINS
from app.database.models import RawTelemetry
from datetime import datetime, timezone


def _make_chunk(node_id="esp32-test", segment_id=1, chunk_id=1, n=43):
    # helper: create a fake telemetry chunk with n samples per axis
    return RawTelemetry(
        node_id=node_id,
        segment_id=segment_id,
        chunk_id=chunk_id,
        sample_rate=800.0,
        emit_reason=1,
        accel_x=list(np.random.randn(n).astype(float)),
        accel_y=list(np.random.randn(n).astype(float)),
        accel_z=list(np.random.randn(n).astype(float)),
        gyro_x=list(np.random.randn(n).astype(float)),
        gyro_y=list(np.random.randn(n).astype(float)),
        gyro_z=list(np.random.randn(n).astype(float)),
        raw_bytes_count=n * 6 * 4,
    )


def test_build_input_window_shape():
    chunks = [_make_chunk(chunk_id=i) for i in range(8)]
    window = _build_input_window(chunks, tsteps=8)
    assert window is not None
    assert window.shape == (8, FREQ_BINS)
    assert window.dtype == np.float32


def test_build_input_window_padding():
    # fewer chunks than tsteps: should pad with zeros at start
    chunks = [_make_chunk(chunk_id=i) for i in range(3)]
    window = _build_input_window(chunks, tsteps=8)
    assert window is not None
    assert window.shape == (8, FREQ_BINS)


def test_build_input_window_empty():
    result = _build_input_window([], tsteps=8)
    assert result is None


def test_run_tflite_inference_no_runtime():
    # no tflite runtime installed in test env, fallback should return float
    arr = np.random.randn(8, FREQ_BINS).astype(np.float32)
    # fake model binary (won't be loaded without runtime)
    fake_model = b"\x00" * 64
    mse = _run_tflite_inference(fake_model, arr)
    assert isinstance(mse, float)
    assert mse >= 0.0
