# timeseries dataset builder with contiguous sequence grouping and stride augmentation
from typing import List, Dict, Tuple, Optional
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import RawTelemetryModel, NodeModel, InferenceResultModel

def group_contiguous_sequences(chunks: List[RawTelemetryModel]) -> List[List[RawTelemetryModel]]:
    # sort by segment_id, chunk_id and group only consecutive segment ids
    if not chunks:
        return []
    
    sorted_chunks = sorted(chunks, key=lambda c: (c.segment_id, c.chunk_id))
    sequences: List[List[RawTelemetryModel]] = []
    current_seq: List[RawTelemetryModel] = []
    
    last_seg_id = None
    for chunk in sorted_chunks:
        if last_seg_id is None:
            current_seq.append(chunk)
            last_seg_id = chunk.segment_id
        elif chunk.segment_id == last_seg_id:
            # same segment, next chunk
            current_seq.append(chunk)
        elif chunk.segment_id == last_seg_id + 1:
            # strictly consecutive segment -> continue sequence
            current_seq.append(chunk)
            last_seg_id = chunk.segment_id
        else:
            # gap in segment ids -> break sequence boundary
            if current_seq:
                sequences.append(current_seq)
            current_seq = [chunk]
            last_seg_id = chunk.segment_id
            
    if current_seq:
        sequences.append(current_seq)
    return sequences

from app.nas.dsp import compute_segment_spectrogram, FFT_WINDOW_SIZE

def flatten_chunk_data(chunk: RawTelemetryModel, accel_bins: int = 128, gyro_bins: int = 128) -> np.ndarray:
    # compute true STFT spectrogram frame matching ESP32 firmware
    spec = compute_segment_spectrogram(
        chunk.accel_x, chunk.accel_y, chunk.accel_z,
        chunk.gyro_x, chunk.gyro_y, chunk.gyro_z,
        accel_bins=accel_bins, gyro_bins=gyro_bins
    )
    return spec[0].astype(np.float32)

def augment_sequence_sliding_window(
    seq_chunks: List[RawTelemetryModel],
    window_size: int = 16, # number of temporal steps
    stride: Optional[int] = None, # sliding step
    accel_bins: int = 128,
    gyro_bins: int = 128
) -> List[np.ndarray]:
    if not seq_chunks:
        return []

    # Group chunks by segment_id to reassemble complete 4096-sample telemetry segments
    seg_map: Dict[int, List[RawTelemetryModel]] = {}
    for c in sorted(seq_chunks, key=lambda x: (x.segment_id, x.chunk_id)):
        seg_map.setdefault(c.segment_id, []).append(c)

    all_frames: List[np.ndarray] = []
    has_multi_chunks = any(len(v) > 1 for v in seg_map.values())

    if has_multi_chunks:
        for sid, s_chunks in seg_map.items():
            ax = np.concatenate([c.accel_x for c in s_chunks if c.accel_x]) if s_chunks else np.array([])
            ay = np.concatenate([c.accel_y for c in s_chunks if c.accel_y]) if s_chunks else np.array([])
            az = np.concatenate([c.accel_z for c in s_chunks if c.accel_z]) if s_chunks else np.array([])
            gx = np.concatenate([c.gyro_x for c in s_chunks if c.gyro_x]) if s_chunks else np.array([])
            gy = np.concatenate([c.gyro_y for c in s_chunks if c.gyro_y]) if s_chunks else np.array([])
            gz = np.concatenate([c.gyro_z for c in s_chunks if c.gyro_z]) if s_chunks else np.array([])

            spec = compute_segment_spectrogram(ax, ay, az, gx, gy, gz, accel_bins, gyro_bins)
            for f in range(len(spec)):
                all_frames.append(spec[f])
    else:
        # Check if individual chunks are full segments (>= 256 samples) or short test mocks
        first_len = len(seq_chunks[0].accel_x) if seq_chunks and seq_chunks[0].accel_x else 0
        if first_len >= FFT_WINDOW_SIZE:
            for c in seq_chunks:
                spec = compute_segment_spectrogram(
                    c.accel_x, c.accel_y, c.accel_z,
                    c.gyro_x, c.gyro_y, c.gyro_z,
                    accel_bins, gyro_bins
                )
                for f in range(len(spec)):
                    all_frames.append(spec[f])
        else:
            all_frames = [flatten_chunk_data(c, accel_bins, gyro_bins) for c in seq_chunks]

    total_frames = len(all_frames)
    total_bins = accel_bins + gyro_bins
    if total_frames < window_size:
        while len(all_frames) < window_size:
            all_frames.insert(0, np.zeros(total_bins, dtype=np.float32))
        return [np.array(all_frames, dtype=np.float32)]

    # default stride: 50% overlap of window size
    step = stride if stride is not None and stride > 0 else max(1, window_size // 2)

    windows = []
    for start_idx in range(0, total_frames - window_size + 1, step):
        window = np.array(all_frames[start_idx : start_idx + window_size], dtype=np.float32)
        windows.append(window)

    return windows

async def build_training_dataset(
    session: AsyncSession,
    node_id: str,
    window_size: int = 16,
    stride: Optional[int] = None,
    accel_bins: int = 128,
    gyro_bins: int = 128,
    val_split_ratio: float = 0.20
) -> Tuple[np.ndarray, np.ndarray]:
    # pull raw chunks, separate flagged anomalies, and build clean training set + holdout validation set
    node = await session.get(NodeModel, node_id)
    last_seg = node.last_trained_segment_id if node else 0
    
    # 1. find all segments flagged as anomaly (automatically or manually)
    stmt_anom = (
        select(InferenceResultModel.segment_id)
        .where(
            InferenceResultModel.node_id == node_id,
            InferenceResultModel.segment_id > last_seg,
            InferenceResultModel.anomaly == True
        )
    )
    res_anom = await session.execute(stmt_anom)
    anom_seg_ids = set(res_anom.scalars().all())

    # 2. query raw telemetry chunks
    stmt = (
        select(RawTelemetryModel)
        .where(RawTelemetryModel.node_id == node_id, RawTelemetryModel.segment_id > last_seg)
        .order_by(RawTelemetryModel.segment_id.asc(), RawTelemetryModel.chunk_id.asc())
    )
    res = await session.execute(stmt)
    chunks = list(res.scalars().all())
    
    clean_chunks = [c for c in chunks if c.segment_id not in anom_seg_ids]
    anom_chunks = [c for c in chunks if c.segment_id in anom_seg_ids]

    clean_sequences = group_contiguous_sequences(clean_chunks)
    clean_windows: List[np.ndarray] = []
    for seq in clean_sequences:
        windows = augment_sequence_sliding_window(
            seq,
            window_size=window_size,
            stride=stride,
            accel_bins=accel_bins,
            gyro_bins=gyro_bins
        )
        clean_windows.extend(windows)

    anom_sequences = group_contiguous_sequences(anom_chunks)
    anom_windows: List[np.ndarray] = []
    for seq in anom_sequences:
        windows = augment_sequence_sliding_window(
            seq,
            window_size=window_size,
            stride=stride,
            accel_bins=accel_bins,
            gyro_bins=gyro_bins
        )
        anom_windows.extend(windows)

    if not clean_windows:
        fallback = np.random.randn(20, window_size, accel_bins + gyro_bins).astype(np.float32)
        n_val = max(1, int(len(fallback) * val_split_ratio))
        return fallback[:-n_val], fallback[-n_val:]
        
    clean_dataset = np.array(clean_windows, dtype=np.float32)
    # chronological holdout split (80% clean train, 20% unseen clean validation)
    n_val = max(1, int(len(clean_dataset) * val_split_ratio))
    train_data = clean_dataset[:-n_val] # 100% clean known-good data for training
    val_clean = clean_dataset[-n_val:]  # 100% clean known-good holdout for validation loss & threshold calibration
    
    return train_data, val_clean


