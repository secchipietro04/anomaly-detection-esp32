# local tflite recalculator - runs inference on raw segments stored in DB
import io
import logging
import struct
import tempfile
import os
from typing import Optional, List, Tuple

import numpy as np

from app.database.models import (
    RawTelemetry,
    get_raw_telemetry_for_segment,
    get_model,
    update_inference_recalculated,
)
from app.database.session import db_manager

logger = logging.getLogger("recalculator")

# frequency bins expected by models (matches firmware stft output)
ACCEL_BINS = 128
GYRO_BINS = 128
FREQ_BINS = ACCEL_BINS + GYRO_BINS  # 256 total


def _build_input_window(
    chunks: List[RawTelemetry],
    tsteps: int = 8,
) -> Optional[np.ndarray]:
    # assemble flattened accel+gyro freq bins from telemetry chunks
    # each chunk contributes one temporal step of shape [FREQ_BINS]
    frames = []
    for chunk in chunks[:tsteps]:
        # flatten accel and gyro bins from chunk arrays
        ax = list(chunk.accel_x)[:ACCEL_BINS // 3]
        ay = list(chunk.accel_y)[:ACCEL_BINS // 3]
        az = list(chunk.accel_z)[:ACCEL_BINS - len(ax) - len(ay)]
        gx = list(chunk.gyro_x)[:GYRO_BINS // 3]
        gy = list(chunk.gyro_y)[:GYRO_BINS // 3]
        gz = list(chunk.gyro_z)[:GYRO_BINS - len(gx) - len(gy)]

        frame = ax + ay + az + gx + gy + gz
        # pad if needed
        if len(frame) < FREQ_BINS:
            frame += [0.0] * (FREQ_BINS - len(frame))
        frames.append(frame[:FREQ_BINS])

    if not frames:
        return None

    # pad temporal dimension with zeros
    while len(frames) < tsteps:
        frames.insert(0, [0.0] * FREQ_BINS)

    arr = np.array(frames, dtype=np.float32)  # [tsteps, FREQ_BINS]
    return arr


def _run_tflite_inference(model_binary: bytes, input_arr: np.ndarray) -> float:
    # run tflite interpreter on input window, return reconstruction mse on last frame
    # try importing tflite_runtime first, fall back to tensorflow lite
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            # no tflite available - return simulated mse for dev/test
            logger.warning("no tflite runtime found, returning simulated mse")
            return float(np.mean(input_arr[-1] ** 2))

    # write model bytes to temp file (tflite requires file path)
    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
        f.write(model_binary)
        tmp_path = f.name

    try:
        interp = Interpreter(model_path=tmp_path)
        interp.allocate_tensors()

        in_details = interp.get_input_details()
        out_details = interp.get_output_details()

        in_shape = in_details[0]["shape"]
        # reshape input to match model expectations
        model_input = input_arr.reshape(in_shape).astype(np.float32)
        interp.set_tensor(in_details[0]["index"], model_input)
        interp.invoke()

        output = interp.get_tensor(out_details[0]["index"])
        # mse computed on last frame reconstruction vs input last frame
        last_frame = input_arr[-1]
        out_last = output.flatten()[-FREQ_BINS:]
        mse = float(np.mean((last_frame - out_last[:len(last_frame)]) ** 2))
        return mse
    finally:
        os.unlink(tmp_path)


async def recalculate_segment(
    node_id: str,
    segment_id: int,
    autoencoder_model_id: int,
    anomaly_threshold: float = 0.1,
    tsteps: int = 8,
) -> Tuple[float, bool]:
    """
    pull raw telemetry for segment, run local tflite inference,
    update inference_results with recalculated mse and anomaly flag.
    returns (mse, anomaly).
    """
    async with db_manager.acquire() as conn:
        chunks = await get_raw_telemetry_for_segment(conn, node_id, segment_id)
        if not chunks:
            raise ValueError(f"no telemetry for node={node_id} segment={segment_id}")

        model_pkg = await get_model(conn, autoencoder_model_id)
        if not model_pkg:
            raise ValueError(f"model id={autoencoder_model_id} not found")

        input_arr = _build_input_window(chunks, tsteps=tsteps)
        if input_arr is None:
            raise ValueError("could not build input window from chunks")

        mse = _run_tflite_inference(model_pkg.tflite_binary, input_arr)
        anomaly = mse > anomaly_threshold

        await update_inference_recalculated(conn, node_id, segment_id, mse, anomaly)
        logger.info(f"recalculated node={node_id} seg={segment_id} mse={mse:.4f} anomaly={anomaly}")

    return mse, anomaly
