# unit test for local recalculator window builder
import numpy as np
from app.database.models import RawTelemetryModel
from app.api.recalculator import _build_input_window

def test_recalculator_input_window_builder():
    chunks = [
        RawTelemetryModel(
            node_id="node_01",
            segment_id=1,
            chunk_id=i,
            sample_rate=800.0,
            emit_reason=1,
            accel_x=[1.0] * 40,
            accel_y=[2.0] * 40,
            accel_z=[3.0] * 48,
            gyro_x=[0.1] * 40,
            gyro_y=[0.2] * 40,
            gyro_z=[0.3] * 48,
            raw_bytes_count=256
        ) for i in range(1, 9)
    ]
    window = _build_input_window(chunks, tsteps=8, accel_bins=128, gyro_bins=128)
    assert window is not None
    assert window.shape == (8, 256)
    assert window.dtype == np.float32
