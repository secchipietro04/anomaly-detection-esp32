# timeseries dataset builder with contiguous sequence grouping and stride augmentation
from typing import List, Dict, Tuple, Optional
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import RawTelemetryModel, NodeModel

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

def flatten_chunk_data(chunk: RawTelemetryModel, accel_bins: int = 128, gyro_bins: int = 128) -> np.ndarray:
    # extract flattened frequency bins from chunk
    ax = np.array(chunk.accel_x, dtype=np.float32)
    ay = np.array(chunk.accel_y, dtype=np.float32)
    az = np.array(chunk.accel_z, dtype=np.float32)
    gx = np.array(chunk.gyro_x, dtype=np.float32)
    gy = np.array(chunk.gyro_y, dtype=np.float32)
    gz = np.array(chunk.gyro_z, dtype=np.float32)
    
    total_bins = accel_bins + gyro_bins
    frame = np.concatenate([ax, ay, az, gx, gy, gz]) if len(ax) > 0 else np.zeros(total_bins, dtype=np.float32)
    if len(frame) < total_bins:
        frame = np.pad(frame, (0, total_bins - len(frame)))
    return frame[:total_bins].astype(np.float32)

def augment_sequence_sliding_window(
    seq_chunks: List[RawTelemetryModel],
    window_size: int = 16, # number of temporal steps
    stride: Optional[int] = None, # sliding step
    accel_bins: int = 128,
    gyro_bins: int = 128
) -> List[np.ndarray]:
    # generate sliding windows strictly within contiguous sequence
    if len(seq_chunks) < window_size:
        # sequence too short, pad initial steps
        frames = [flatten_chunk_data(c, accel_bins, gyro_bins) for c in seq_chunks]
        while len(frames) < window_size:
            frames.insert(0, np.zeros_like(frames[0]))
        return [np.array(frames, dtype=np.float32)]
    
    frames = [flatten_chunk_data(c, accel_bins, gyro_bins) for c in seq_chunks]
    total_frames = len(frames)
    
    # default stride: 50% overlap of window size (e.g. window=16 -> stride=8)
    step = stride if stride is not None and stride > 0 else max(1, window_size // 2)
    
    windows = []
    for start_idx in range(0, total_frames - window_size + 1, step):
        window = np.array(frames[start_idx : start_idx + window_size], dtype=np.float32)
        windows.append(window)
        
    return windows

async def build_training_dataset(
    session: AsyncSession,
    node_id: str,
    window_size: int = 16,
    stride: Optional[int] = None,
    accel_bins: int = 128,
    gyro_bins: int = 128
) -> np.ndarray:
    # pull raw chunks, group into contiguous sequences and augment
    node = await session.get(NodeModel, node_id)
    last_seg = node.last_trained_segment_id if node else 0
    
    stmt = (
        select(RawTelemetryModel)
        .where(RawTelemetryModel.node_id == node_id, RawTelemetryModel.segment_id > last_seg)
        .order_by(RawTelemetryModel.segment_id.asc(), RawTelemetryModel.chunk_id.asc())
    )
    res = await session.execute(stmt)
    chunks = list(res.scalars().all())
    
    sequences = group_contiguous_sequences(chunks)
    all_windows: List[np.ndarray] = []
    
    for seq in sequences:
        windows = augment_sequence_sliding_window(
            seq,
            window_size=window_size,
            stride=stride,
            accel_bins=accel_bins,
            gyro_bins=gyro_bins
        )
        all_windows.extend(windows)
        
    if not all_windows:
        # fallback synthetic window if no data
        return np.random.randn(10, window_size, accel_bins + gyro_bins).astype(np.float32)
        
    return np.array(all_windows, dtype=np.float32)
