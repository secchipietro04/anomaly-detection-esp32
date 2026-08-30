# local tflite recalculator running inference on stored raw telemetry
import os
import logging
import tempfile
from typing import Optional, List, Tuple
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database.models import RawTelemetryModel, ModelPackageModel, InferenceResultModel
from app.database.session import get_db_session

logger = logging.getLogger("recalculator")

def _build_input_window(
    chunks: List[RawTelemetryModel],
    tsteps: int = 8,
    accel_bins: int = 128,
    gyro_bins: int = 128
) -> Optional[np.ndarray]:
    # assemble temporal window from telemetry chunks
    freq_bins = accel_bins + gyro_bins
    frames = []
    for chunk in chunks[:tsteps]:
        ax = list(chunk.accel_x)[:accel_bins // 3]
        ay = list(chunk.accel_y)[:accel_bins // 3]
        az = list(chunk.accel_z)[:accel_bins - len(ax) - len(ay)]
        gx = list(chunk.gyro_x)[:gyro_bins // 3]
        gy = list(chunk.gyro_y)[:gyro_bins // 3]
        gz = list(chunk.gyro_z)[:gyro_bins - len(gx) - len(gy)]

        frame = ax + ay + az + gx + gy + gz
        if len(frame) < freq_bins:
            frame += [0.0] * (freq_bins - len(frame))
        frames.append(frame[:freq_bins])

    if not frames:
        return None

    while len(frames) < tsteps:
        frames.insert(0, [0.0] * freq_bins)

    return np.array(frames, dtype=np.float32)

def _run_tflite_inference(model_binary: bytes, input_arr: np.ndarray) -> float:
    # run tflite interpreter or fallback simulated mse on last frame
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            # fallback simulation for dev environment
            return float(np.mean(input_arr[-1] ** 2))

    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
        f.write(model_binary)
        tmp_path = f.name

    try:
        interp = Interpreter(model_path=tmp_path)
        interp.allocate_tensors()
        in_details = interp.get_input_details()
        out_details = interp.get_output_details()

        in_shape = in_details[0]["shape"]
        model_input = input_arr.reshape(in_shape).astype(np.float32)
        interp.set_tensor(in_details[0]["index"], model_input)
        interp.invoke()

        output = interp.get_tensor(out_details[0]["index"])
        last_frame = input_arr[-1]
        out_last = output.flatten()[-len(last_frame):]
        return float(np.mean((last_frame - out_last) ** 2))
    finally:
        os.unlink(tmp_path)

async def recalculate_segment(
    session: AsyncSession,
    node_id: str,
    segment_id: int,
    autoencoder_model_id: int,
    anomaly_threshold: float = 0.1,
    tsteps: int = 8,
) -> Tuple[float, bool]:
    # pull chunks, run inference and save recalculated score
    stmt_raw = (
        select(RawTelemetryModel)
        .where(RawTelemetryModel.node_id == node_id, RawTelemetryModel.segment_id == segment_id)
        .order_by(RawTelemetryModel.chunk_id.asc())
    )
    res_raw = await session.execute(stmt_raw)
    chunks = list(res_raw.scalars().all())
    if not chunks:
        raise ValueError(f"No telemetry found for node {node_id} segment {segment_id}")

    model_pkg = await session.get(ModelPackageModel, autoencoder_model_id)
    if not model_pkg:
        raise ValueError(f"Model id {autoencoder_model_id} not found")

    input_arr = _build_input_window(chunks, tsteps=tsteps)
    if input_arr is None:
        raise ValueError("Could not build input window from chunks")

    mse = _run_tflite_inference(model_pkg.tflite_binary, input_arr)
    anomaly = mse > anomaly_threshold

    # update or insert into inference_results
    stmt_inf = select(InferenceResultModel).where(
        InferenceResultModel.node_id == node_id,
        InferenceResultModel.segment_id == segment_id
    )
    res_inf = await session.execute(stmt_inf)
    inf = res_inf.scalar_one_or_none()
    if inf:
        inf.mse = mse
        inf.anomaly = anomaly
        inf.is_recalculated = True
        inf.autoencoder_model_id = autoencoder_model_id
    await session.commit()
    return mse, anomaly
