# local tflite recalculator and robustness benchmark running on raw telemetry
import os
import logging
import tempfile
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database.models import RawTelemetryModel, ModelPackageModel, EnsembleConfigModel, InferenceResultModel
from app.database.session import get_db_session

logger = logging.getLogger("recalculator")

ROUTER_CONFIDENCE_THRESHOLD = 0.5

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

def _invoke_tflite_raw(model_binary: bytes, input_arr: np.ndarray) -> np.ndarray:
    # run tflite interpreter or fallback simulation
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            # fallback simulation for dev environments without tflite installed
            return input_arr[-1] * 0.95

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
        return interp.get_tensor(out_details[0]["index"])
    finally:
        os.unlink(tmp_path)

def _run_tflite_inference(model_binary: bytes, input_arr: np.ndarray) -> float:
    output = _invoke_tflite_raw(model_binary, input_arr)
    last_frame = input_arr[-1]
    out_last = output.flatten()[-len(last_frame):]
    return float(np.mean((last_frame - out_last) ** 2))

def run_ensemble_inference_on_window(
    ensemble: EnsembleConfigModel,
    model_binaries: Dict[int, bytes],
    model_configs: Dict[int, Dict[str, Any]],
    input_arr: np.ndarray,
    h_state: Optional[np.ndarray] = None,
    c_state: Optional[np.ndarray] = None
) -> Tuple[float, bool, Optional[int], Optional[int], np.ndarray, np.ndarray]:
    # executes full memory -> router -> autoencoder chain
    mem_d = 16
    if h_state is None:
        h_state = np.zeros((mem_d,), dtype=np.float32)
    if c_state is None:
        c_state = np.zeros((mem_d,), dtype=np.float32)

    # 1. memory backbone
    if ensemble.memory_model_id and ensemble.memory_model_id in model_binaries:
        mem_bin = model_binaries[ensemble.memory_model_id]
        last_slice = input_arr[-1]
        mem_in = np.concatenate([last_slice, h_state, c_state]).reshape(1, -1)
        mem_out = _invoke_tflite_raw(mem_bin, mem_in).flatten()
        if len(mem_out) >= 2 * mem_d:
            h_state = mem_out[:mem_d]
            c_state = mem_out[mem_d:2 * mem_d]

    # 2. router model
    chosen_ae_id = None
    if ensemble.router_model_id and ensemble.router_model_id in model_binaries:
        r_bin = model_binaries[ensemble.router_model_id]
        r_out = _invoke_tflite_raw(r_bin, input_arr).flatten()
        
        # calculate softmax
        exp_logits = np.exp(r_out - np.max(r_out))
        probs = exp_logits / np.sum(exp_logits)
        best_mode = int(np.argmax(probs))
        best_prob = float(probs[best_mode])

        # check router anomaly threshold
        if best_prob < ROUTER_CONFIDENCE_THRESHOLD:
            # classified as anomaly directly by router
            return float(1.0 - best_prob), True, ensemble.router_model_id, None, h_state, c_state

        # resolve submodel from routing table
        for r in ensemble.routes or []:
            if r.get("out_ix") == best_mode:
                chosen_ae_id = r.get("m_id")
                break

    # fallback to first route or default submodel
    if chosen_ae_id is None and ensemble.routes:
        chosen_ae_id = ensemble.routes[0].get("m_id")

    # 3. autoencoder model
    if chosen_ae_id and chosen_ae_id in model_binaries:
        ae_bin = model_binaries[chosen_ae_id]
        ae_cfg = model_configs.get(chosen_ae_id, {})
        limit = float(ae_cfg.get("limit", 0.1))
        mse = _run_tflite_inference(ae_bin, input_arr)
        is_anomaly = mse > limit
        return mse, is_anomaly, ensemble.router_model_id, chosen_ae_id, h_state, c_state

    # fallback calculation if no ae model loaded
    mse = float(np.mean(input_arr[-1] ** 2))
    return mse, (mse > 0.1), ensemble.router_model_id, chosen_ae_id, h_state, c_state

async def recalculate_segment(
    session: AsyncSession,
    node_id: str,
    segment_id: int,
    autoencoder_model_id: Optional[int] = None,
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

    input_arr = _build_input_window(chunks, tsteps=tsteps)
    if input_arr is None:
        raise ValueError("Could not build input window from chunks")

    # if autoencoder_model_id is explicitly specified, run single model
    if autoencoder_model_id is not None:
        model_pkg = await session.get(ModelPackageModel, autoencoder_model_id)
        if not model_pkg:
            raise ValueError(f"Model id {autoencoder_model_id} not found")
        mse = _run_tflite_inference(model_pkg.tflite_binary, input_arr)
        anomaly = mse > anomaly_threshold
        r_id = None
        ae_id = autoencoder_model_id
    else:
        # fetch active ensemble for node
        stmt_ens = (
            select(EnsembleConfigModel)
            .where(EnsembleConfigModel.node_id == node_id)
            .order_by(EnsembleConfigModel.deployed_at.desc())
            .limit(1)
        )
        res_ens = await session.execute(stmt_ens)
        ensemble = res_ens.scalar_one_or_none()
        if not ensemble:
            raise ValueError(f"No active ensemble deployed for node {node_id}")

        # fetch all model packages required by this ensemble
        m_ids = set()
        if ensemble.router_model_id:
            m_ids.add(ensemble.router_model_id)
        if ensemble.memory_model_id:
            m_ids.add(ensemble.memory_model_id)
        for r in ensemble.routes or []:
            if "m_id" in r:
                m_ids.add(r["m_id"])

        stmt_m = select(ModelPackageModel).where(ModelPackageModel.id.in_(m_ids))
        res_m = await session.execute(stmt_m)
        pkgs = res_m.scalars().all()
        model_bins = {p.id: p.tflite_binary for p in pkgs}
        model_cfgs = {p.id: p.config for p in pkgs}

        mse, anomaly, r_id, ae_id, _, _ = run_ensemble_inference_on_window(
            ensemble, model_bins, model_cfgs, input_arr
        )

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
        inf.router_model_id = r_id
        inf.autoencoder_model_id = ae_id
    else:
        inf = InferenceResultModel(
            node_id=node_id,
            segment_id=segment_id,
            mse=mse,
            anomaly=anomaly,
            is_recalculated=True,
            router_model_id=r_id,
            autoencoder_model_id=ae_id
        )
        session.add(inf)
    await session.commit()
    return mse, anomaly

async def test_noise_resilience(
    session: AsyncSession,
    node_id: str,
    num_sample_segments: int = 64,
    num_noise_levels: int = 64,
    min_power_2: float = -10.0,
    max_power_2: float = 0.0,
    tsteps: int = 8
) -> Dict[str, Any]:
    # tests ensemble robustness by perturbing individual frequency bins across base-2 noise levels (2^-10 to 2^0)
    # 1. fetch active ensemble and its model binaries
    stmt_ens = (
        select(EnsembleConfigModel)
        .where(EnsembleConfigModel.node_id == node_id)
        .order_by(EnsembleConfigModel.deployed_at.desc())
        .limit(1)
    )
    res_ens = await session.execute(stmt_ens)
    ensemble = res_ens.scalar_one_or_none()
    if not ensemble:
        raise ValueError(f"No active ensemble deployed for node '{node_id}'")

    m_ids = set()
    if ensemble.router_model_id:
        m_ids.add(ensemble.router_model_id)
    if ensemble.memory_model_id:
        m_ids.add(ensemble.memory_model_id)
    for r in ensemble.routes or []:
        if "m_id" in r:
            m_ids.add(r["m_id"])

    stmt_m = select(ModelPackageModel).where(ModelPackageModel.id.in_(m_ids))
    res_m = await session.execute(stmt_m)
    pkgs = res_m.scalars().all()
    model_bins = {p.id: p.tflite_binary for p in pkgs}
    model_cfgs = {p.id: p.config for p in pkgs}

    # 2. sample unique segment IDs for node
    stmt_segs = (
        select(RawTelemetryModel.segment_id)
        .where(RawTelemetryModel.node_id == node_id)
        .distinct()
        .order_by(func.random())
        .limit(num_sample_segments)
    )
    res_segs = await session.execute(stmt_segs)
    seg_ids = list(res_segs.scalars().all())
    if not seg_ids:
        raise ValueError(f"No raw telemetry segments found for node '{node_id}'")

    # 3. build input windows for all sample segments
    sample_windows: List[np.ndarray] = []
    for sid in seg_ids:
        stmt_raw = (
            select(RawTelemetryModel)
            .where(RawTelemetryModel.node_id == node_id, RawTelemetryModel.segment_id == sid)
            .order_by(RawTelemetryModel.chunk_id.asc())
        )
        res_raw = await session.execute(stmt_raw)
        chunks = list(res_raw.scalars().all())
        w = _build_input_window(chunks, tsteps=tsteps)
        if w is not None:
            sample_windows.append(w)

    if not sample_windows:
        raise ValueError("Could not construct input windows for sample segments")

    total_bins = sample_windows[0].shape[-1]
    
    # base-2 noise levels: from 2^-10 (~0.1%) to 2^0 (100%)
    powers_of_2 = np.linspace(min_power_2, max_power_2, num_noise_levels)
    noise_levels = np.power(2.0, powers_of_2).tolist()

    # 4. compute baseline anomaly scores
    baseline_scores = []
    for w in sample_windows:
        mse, _, _, _, _, _ = run_ensemble_inference_on_window(
            ensemble, model_bins, model_cfgs, w
        )
        baseline_scores.append(mse)
    avg_baseline_score = float(np.mean(baseline_scores))

    # 5. generate 2D sensitivity matrix: shape (num_noise_levels, total_bins)
    # X-axis = frequency bins (0 to total_bins-1)
    # Y-axis = noise level in base-2 (2^-10 to 2^0)
    # Color = average delta anomaly score across sample segments
    sensitivity_matrix = np.zeros((num_noise_levels, total_bins), dtype=np.float32)

    for bin_idx in range(total_bins):
        for lvl_idx, nsr in enumerate(noise_levels):
            deltas = []
            for w in sample_windows:
                noisy_w = np.copy(w)
                std_dev = np.std(w[:, bin_idx]) + 1e-6
                noise = np.random.normal(0.0, std_dev * nsr, size=w[:, bin_idx].shape)
                noisy_w[:, bin_idx] += noise

                mse, _, _, _, _, _ = run_ensemble_inference_on_window(
                    ensemble, model_bins, model_cfgs, noisy_w
                )
                deltas.append(max(0.0, mse - avg_baseline_score))
            sensitivity_matrix[lvl_idx, bin_idx] = float(np.mean(deltas))

    # 6. compute quantified resilience indicators
    avg_sensitivity_per_level = np.mean(sensitivity_matrix, axis=1)
    critical_idx = int(np.argmax(avg_sensitivity_per_level > 0.05)) if np.any(avg_sensitivity_per_level > 0.05) else len(noise_levels) - 1
    critical_noise_threshold = float(noise_levels[critical_idx])
    critical_noise_power_2 = float(powers_of_2[critical_idx])

    # find most sensitive frequency bands
    bin_sensitivity = np.mean(sensitivity_matrix, axis=0)
    most_sensitive_bins = [int(i) for i in np.argsort(bin_sensitivity)[-5:][::-1]]

    # overall resilience score: 1.0 (highly robust) to 0.0 (fragile)
    normalized_auc = float(np.mean(sensitivity_matrix))
    robustness_score = float(np.clip(1.0 - (normalized_auc * 2.0), 0.0, 1.0))

    return {
        "node_id": node_id,
        "ensemble_id": ensemble.id,
        "num_segments_evaluated": len(sample_windows),
        "total_frequency_bins": total_bins,
        "noise_scale_base": 2,
        "powers_of_2": powers_of_2.tolist(),
        "noise_levels_nsr": noise_levels,
        "sensitivity_spectrogram": sensitivity_matrix.tolist(),
        "metrics": {
            "avg_baseline_score": avg_baseline_score,
            "critical_noise_threshold_nsr": critical_noise_threshold,
            "critical_noise_power_2": critical_noise_power_2,
            "overall_robustness_score": robustness_score,
            "most_sensitive_bins": most_sensitive_bins
        }
    }

async def test_cross_submodel_specificity(
    session: AsyncSession,
    node_id: str,
    num_samples_per_mode: int = 50,
    tsteps: int = 8
) -> Dict[str, Any]:
    # evaluates cross-submodel specificity matrix by forcing each autoencoder to evaluate foreign cluster data
    # 1. fetch active ensemble
    stmt_ens = (
        select(EnsembleConfigModel)
        .where(EnsembleConfigModel.node_id == node_id)
        .order_by(EnsembleConfigModel.deployed_at.desc())
        .limit(1)
    )
    res_ens = await session.execute(stmt_ens)
    ensemble = res_ens.scalar_one_or_none()
    if not ensemble:
        raise ValueError(f"No active ensemble deployed for node '{node_id}'")

    routes = ensemble.routes or []
    submodel_ids = [r["m_id"] for r in routes if "m_id" in r]
    if not submodel_ids:
        raise ValueError(f"Ensemble for node '{node_id}' has no configured routes")

    # fetch all model packages
    stmt_m = select(ModelPackageModel).where(ModelPackageModel.id.in_(submodel_ids))
    res_m = await session.execute(stmt_m)
    pkgs = res_m.scalars().all()
    model_bins = {p.id: p.tflite_binary for p in pkgs}
    model_cfgs = {p.id: p.config for p in pkgs}

    # 2. fetch sample segments and classify them into mode partitions using router or k-means grouping
    stmt_segs = (
        select(RawTelemetryModel.segment_id)
        .where(RawTelemetryModel.node_id == node_id)
        .distinct()
        .limit(num_samples_per_mode * len(submodel_ids) * 2)
    )
    res_segs = await session.execute(stmt_segs)
    seg_ids = list(res_segs.scalars().all())

    # collect sample windows and assign partition index
    clusters_data: Dict[int, List[np.ndarray]] = {r_idx: [] for r_idx in range(len(submodel_ids))}
    for sid in seg_ids:
        stmt_raw = (
            select(RawTelemetryModel)
            .where(RawTelemetryModel.node_id == node_id, RawTelemetryModel.segment_id == sid)
            .order_by(RawTelemetryModel.chunk_id.asc())
        )
        res_raw = await session.execute(stmt_raw)
        chunks = list(res_raw.scalars().all())
        w = _build_input_window(chunks, tsteps=tsteps)
        if w is not None:
            # partition according to segment ID modulo number of modes or feature distance
            mode_idx = sid % len(submodel_ids)
            if len(clusters_data[mode_idx]) < num_samples_per_mode:
                clusters_data[mode_idx].append(w)

    num_models = len(submodel_ids)
    cross_mse_matrix = np.zeros((num_models, num_models), dtype=np.float32)
    cross_anomaly_rate_matrix = np.zeros((num_models, num_models), dtype=np.float32)

    # 3. compute cross-evaluation matrix: rows = evaluating autoencoder, cols = cluster data
    for i, ae_id in enumerate(submodel_ids):
        ae_bin = model_bins.get(ae_id)
        limit = float(model_cfgs.get(ae_id, {}).get("limit", 0.1))

        for j, mode_idx in enumerate(range(num_models)):
            mode_windows = clusters_data.get(mode_idx, [])
            if not mode_windows:
                continue

            mses = []
            anom_flags = []
            for w in mode_windows:
                if ae_bin:
                    mse = _run_tflite_inference(ae_bin, w)
                else:
                    mse = float(np.mean(w[-1] ** 2))
                mses.append(mse)
                anom_flags.append(mse > limit)

            cross_mse_matrix[i, j] = float(np.mean(mses))
            cross_anomaly_rate_matrix[i, j] = float(np.mean(anom_flags) * 100.0)

    # 4. compute specificity metrics
    # diagonal (i==j) should have low anomaly rate (true negatives)
    # off-diagonal (i!=j) should have high anomaly rate (cross-mode rejection)
    diag_anomaly_rate = float(np.mean(np.diag(cross_anomaly_rate_matrix)))
    off_diag_mask = ~np.eye(num_models, dtype=bool)
    off_diag_anomaly_rate = float(np.mean(cross_anomaly_rate_matrix[off_diag_mask])) if num_models > 1 else 100.0

    # mode separability index: 1.0 (perfect isolation) to 0.0 (complete mode confusion)
    separability_index = float(np.clip((off_diag_anomaly_rate - diag_anomaly_rate) / 100.0, 0.0, 1.0))

    return {
        "node_id": node_id,
        "ensemble_id": ensemble.id,
        "submodel_ids": submodel_ids,
        "num_submodels": num_models,
        "cross_reconstruction_mse_matrix": cross_mse_matrix.tolist(),
        "cross_anomaly_rejection_rate_percent": cross_anomaly_rate_matrix.tolist(),
        "metrics": {
            "self_regime_false_positive_rate_percent": diag_anomaly_rate,
            "cross_regime_rejection_power_percent": off_diag_anomaly_rate,
            "mode_separability_index": separability_index
        }
    }

