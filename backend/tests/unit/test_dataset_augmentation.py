# unit test for contiguous sequence grouping and stride augmentation
import numpy as np
from app.database.models import RawTelemetryModel
from app.nas.dataset import group_contiguous_sequences, augment_sequence_sliding_window

def _mock_chunk(seg_id: int, chunk_id: int = 1):
    return RawTelemetryModel(
        node_id="test",
        segment_id=seg_id,
        chunk_id=chunk_id,
        sample_rate=800.0,
        emit_reason=1,
        accel_x=[1.0] * 10,
        accel_y=[2.0] * 10,
        accel_z=[3.0] * 10,
        gyro_x=[0.1] * 10,
        gyro_y=[0.2] * 10,
        gyro_z=[0.3] * 10,
        raw_bytes_count=60
    )

def test_group_contiguous_sequences_with_gaps():
    # chunks with ids 1, 2, 4, 5 (missing 3) -> 2 separate sequences
    chunks = [
        _mock_chunk(seg_id=1),
        _mock_chunk(seg_id=2),
        _mock_chunk(seg_id=4),
        _mock_chunk(seg_id=5),
    ]
    seqs = group_contiguous_sequences(chunks)
    assert len(seqs) == 2
    assert [c.segment_id for c in seqs[0]] == [1, 2]
    assert [c.segment_id for c in seqs[1]] == [4, 5]

def test_sliding_window_with_stride():
    # 20 contiguous chunks with window_size=8 and stride=4
    chunks = [_mock_chunk(seg_id=i) for i in range(1, 21)]
    windows = augment_sequence_sliding_window(chunks, window_size=8, stride=4, accel_bins=30, gyro_bins=30)
    # total steps: (20 - 8) // 4 + 1 = 4 windows
    assert len(windows) == 4
    for w in windows:
        assert w.shape == (8, 60)
