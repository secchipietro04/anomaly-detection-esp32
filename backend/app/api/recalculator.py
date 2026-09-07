# local tflite recalculator and robustness benchmark running on raw telemetry
import os
import logging
import tempfile
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.database.models import RawTelemetryModel, ModelPackageModel, EnsembleConfigModel, InferenceResultModel
from app.database.session import get_db_session
from app.nas.dsp import compute_segment_spectrogram

logger = logging.getLogger("recalculator")

ROUTER_CONFIDENCE_THRESHOLD = 0.5

def _build_input_window(
    chunks: List[RawTelemetryModel],
    tsteps: int = 8,
    accel_bins: int = 128,
    gyro_bins: int = 128
) -> Optional[np.ndarray]:
    if not chunks:
        return None

    # sort chunks by chunk_id
    sorted_chunks = sorted(chunks, key=lambda c: getattr(c, "chunk_id", 0))

    ax = np.concatenate([c.accel_x for c in sorted_chunks if c.accel_x]) if sorted_chunks else np.array([])
    ay = np.concatenate([c.accel_y for c in sorted_chunks if c.accel_y]) if sorted_chunks else np.array([])
    az = np.concatenate([c.accel_z for c in sorted_chunks if c.accel_z]) if sorted_chunks else np.array([])
    gx = np.concatenate([c.gyro_x for c in sorted_chunks if c.gyro_x]) if sorted_chunks else np.array([])
    gy = np.concatenate([c.gyro_y for c in sorted_chunks if c.gyro_y]) if sorted_chunks else np.array([])
    gz = np.concatenate([c.gyro_z for c in sorted_chunks if c.gyro_z]) if sorted_chunks else np.array([])

    freq_bins = accel_bins + gyro_bins
    if len(ax) == 0:
        return np.zeros((tsteps, freq_bins), dtype=np.float32)

    # compute true STFT spectrogram matching firmware
    spec = compute_segment_spectrogram(
        ax, ay, az, gx, gy, gz,
        accel_bins=accel_bins,
        gyro_bins=gyro_bins
    )

    if len(spec) >= tsteps:
        return spec[:tsteps].astype(np.float32)
    else:
        padded = np.zeros((tsteps, freq_bins), dtype=np.float32)
        padded[-len(spec):] = spec
        return padded

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

def _run_tflite_inference(model_binary: bytes, input_arr: np.ndarray, loss_mode: int = 1) -> float:
    output = _invoke_tflite_raw(model_binary, input_arr)
    last_frame = input_arr[-1]
    out_last = output.flatten()[-len(last_frame):]
    if loss_mode == 1:
        # exact LogMSE: (log(1 + |x|) - log(1 + |x_hat|))^2
        log_in = np.log1p(np.abs(last_frame))
        log_out = np.log1p(np.abs(out_last))
        return float(np.mean((log_in - log_out) ** 2))
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
        
        num_routes = len(ensemble.routes or [])
        if num_routes > 0:
            route_logits = r_out[:num_routes] if len(r_out) >= num_routes else r_out
            exp_logits = np.exp(route_logits - np.max(route_logits))
            probs = exp_logits / np.sum(exp_logits)
            best_mode = int(np.argmax(probs))
            best_prob = float(probs[best_mode])

            # resolve submodel from routing table
            for r in ensemble.routes or []:
                if r.get("out_idx", r.get("out_ix")) == best_mode:
                    chosen_ae_id = r.get("m_id")
                    break

    # fallback to first route or default submodel
    if chosen_ae_id is None and ensemble.routes:
        chosen_ae_id = ensemble.routes[0].get("m_id")

    # 3. autoencoder model
    if chosen_ae_id and chosen_ae_id in model_binaries:
        ae_bin = model_binaries[chosen_ae_id]
        ae_cfg = model_configs.get(chosen_ae_id, {})
        limit = float(ae_cfg.get("limit", 0.15))
        loss_mode = int(ae_cfg.get("loss_mode", 1))
        mse = _run_tflite_inference(ae_bin, input_arr, loss_mode=loss_mode)
        is_anomaly = mse > limit
        return mse, is_anomaly, ensemble.router_model_id, chosen_ae_id, h_state, c_state

    # fallback calculation if no ae model loaded
    mse = float(np.mean(np.log1p(np.abs(input_arr[-1])) ** 2))
    return mse, mse > 0.15, ensemble.router_model_id, None, h_state, c_state

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

async def recalculate_time_range(
    session: AsyncSession,
    node_id: str,
    from_time_ms: Optional[int] = None,
    to_time_ms: Optional[int] = None,
    from_time_str: Optional[str] = None,
    to_time_str: Optional[str] = None,
    autoencoder_model_id: Optional[int] = None,
    anomaly_threshold: float = 0.15,
    tsteps: int = 8
) -> Dict[str, Any]:
    # recalculates all raw segments in the given time window
    conditions = [RawTelemetryModel.node_id == node_id]
    if from_time_ms is not None:
        dt_from = datetime.fromtimestamp(from_time_ms / 1000.0, tz=timezone.utc)
        conditions.append(RawTelemetryModel.timestamp >= dt_from)
    elif from_time_str is not None:
        try:
            dt_from = datetime.fromisoformat(from_time_str.replace("Z", "+00:00"))
            conditions.append(RawTelemetryModel.timestamp >= dt_from)
        except Exception:
            pass

    if to_time_ms is not None:
        dt_to = datetime.fromtimestamp(to_time_ms / 1000.0, tz=timezone.utc)
        conditions.append(RawTelemetryModel.timestamp <= dt_to)
    elif to_time_str is not None:
        try:
            dt_to = datetime.fromisoformat(to_time_str.replace("Z", "+00:00"))
            conditions.append(RawTelemetryModel.timestamp <= dt_to)
        except Exception:
            pass

    stmt_segs = (
        select(RawTelemetryModel.segment_id)
        .where(*conditions)
        .group_by(RawTelemetryModel.segment_id)
        .order_by(RawTelemetryModel.segment_id.asc())
    )
    res_segs = await session.execute(stmt_segs)
    seg_ids = list(res_segs.scalars().all())

    if not seg_ids:
        raise ValueError(f"No raw segments found for node '{node_id}' in the specified time window")

    recalc_count = 0
    anomalies_count = 0
    scores = []

    for sid in seg_ids:
        try:
            mse, is_anom = await recalculate_segment(
                session=session,
                node_id=node_id,
                segment_id=sid,
                autoencoder_model_id=autoencoder_model_id,
                anomaly_threshold=anomaly_threshold,
                tsteps=tsteps
            )
            recalc_count += 1
            if is_anom:
                anomalies_count += 1
            scores.append(mse)
        except Exception as e:
            logger.warning(f"Failed recalculating segment {sid}: {e}")

    return {
        "node_id": node_id,
        "total_segments_in_range": len(seg_ids),
        "recalculated_count": recalc_count,
        "anomalies_detected": anomalies_count,
        "avg_mse": float(np.mean(scores)) if scores else 0.0,
        "min_mse": float(np.min(scores)) if scores else 0.0,
        "max_mse": float(np.max(scores)) if scores else 0.0
    }


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
        .group_by(RawTelemetryModel.segment_id)
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
        seg_id = getattr(sid, "segment_id", sid)
        stmt_raw = (
            select(RawTelemetryModel)
            .where(RawTelemetryModel.node_id == node_id, RawTelemetryModel.segment_id == seg_id)
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
    
    # base-2 noise levels: from 2^-6 (~1.5%) to 2^4 (1600%)
    if min_power_2 == -10.0 and max_power_2 == 0.0:
        min_power_2 = -6.0
        max_power_2 = 4.0
    powers_of_2 = np.linspace(min_power_2, max_power_2, num_noise_levels)
    noise_levels = np.power(2.0, powers_of_2).tolist()

    # 4. compute baseline anomaly scores and baseline reconstructions across full temporal window
    sample_stack = np.array(sample_windows, dtype=np.float32) # (N, tsteps, bins)
    baseline_scores = []
    baseline_reconstructions = []
    for w in sample_stack:
        mse, _, _, ae_id, _, _ = run_ensemble_inference_on_window(
            ensemble, model_bins, model_cfgs, w
        )
        baseline_scores.append(mse)
        # compute normal reconstruction across temporal frames
        if ae_id and ae_id in model_bins:
            out_flat = _invoke_tflite_raw(model_bins[ae_id], w).flatten()
            if len(out_flat) >= total_bins * tsteps:
                out_w = out_flat[-total_bins * tsteps:].reshape((tsteps, total_bins))
            else:
                out_w = np.tile(out_flat[-total_bins:], (tsteps, 1))
        else:
            out_w = w * 0.95
        baseline_reconstructions.append(out_w)

    avg_baseline_score = float(np.mean(baseline_scores))
    base_recs_window = np.array(baseline_reconstructions, dtype=np.float32) # (N, tsteps, bins)
    bin_means = np.maximum(np.mean(np.abs(sample_stack), axis=(0, 1)), 0.1) # (bins,)

    # 5. fast vectorized 2D sensitivity matrix across full temporal window: shape (num_noise_levels, total_bins)
    # X-axis = noise level in base-2 (2^-6 to 2^4)
    # Y-axis = frequency bins (0 to total_bins-1)
    # Color = delta anomaly score against full temporal baseline reconstruction
    sensitivity_matrix = np.zeros((num_noise_levels, total_bins), dtype=np.float32)

    for lvl_idx, nsr in enumerate(noise_levels):
        for bin_idx in range(total_bins):
            # perturb selected frequency bin across all temporal frames of the segment
            noise_scale = float(bin_means[bin_idx] * nsr)
            noisy_w = np.copy(sample_stack)
            noisy_w[:, :, bin_idx] += np.random.normal(0.0, noise_scale, size=(len(sample_stack), tsteps))
            
            # compute LogMSE across all evaluated temporal frames
            log_noisy = np.log1p(np.abs(noisy_w))
            log_clean = np.log1p(np.abs(base_recs_window))
            mse_perturbed = np.mean((log_noisy - log_clean) ** 2, axis=(1, 2)) # (N,)
            delta = np.maximum(0.0, mse_perturbed - avg_baseline_score)
            sensitivity_matrix[lvl_idx, bin_idx] = float(np.mean(delta))

    # 6. compute quantified resilience indicators
    avg_sensitivity_per_level = np.mean(sensitivity_matrix, axis=1)
    # Anomaly threshold crossing (e.g. delta > 0.15)
    anom_crossings = np.where(avg_sensitivity_per_level >= 0.15)[0]
    critical_idx = int(anom_crossings[0]) if len(anom_crossings) > 0 else len(noise_levels) - 1
    critical_noise_threshold = float(noise_levels[critical_idx])
    critical_noise_power_2 = float(powers_of_2[critical_idx])

    # find most sensitive frequency bands
    bin_sensitivity = np.mean(sensitivity_matrix, axis=0)
    most_sensitive_bins = [int(i) for i in np.argsort(bin_sensitivity)[-5:][::-1]]

    # overall resilience score: ratio of noise levels safely below threshold
    safe_fraction = float(critical_idx) / float(max(1, len(noise_levels) - 1))
    robustness_score = float(np.clip(safe_fraction, 0.05, 0.95))

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
        seg_id = getattr(sid, "segment_id", sid)
        stmt_raw = (
            select(RawTelemetryModel)
            .where(RawTelemetryModel.node_id == node_id, RawTelemetryModel.segment_id == seg_id)
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

async def get_segment_stft_spectrogram(
    session: AsyncSession,
    node_id: str,
    segment_id: Optional[int] = None,
    from_time_ms: Optional[int] = None,
    to_time_ms: Optional[int] = None
) -> Dict[str, Any]:
    # computes full STFT frequency-time spectrogram for raw telemetry across the requested time window
    conditions = [RawTelemetryModel.node_id == node_id]
    if from_time_ms is not None:
        dt_from = datetime.fromtimestamp(from_time_ms / 1000.0, tz=timezone.utc)
        conditions.append(RawTelemetryModel.timestamp >= dt_from)
    if to_time_ms is not None:
        dt_to = datetime.fromtimestamp(to_time_ms / 1000.0, tz=timezone.utc)
        conditions.append(RawTelemetryModel.timestamp <= dt_to)

    if segment_id is not None:
        conditions.append(RawTelemetryModel.segment_id == segment_id)
    elif from_time_ms is None and to_time_ms is None:
        stmt_max = select(func.max(RawTelemetryModel.segment_id)).where(RawTelemetryModel.node_id == node_id)
        res_max = await session.execute(stmt_max)
        latest_sid = res_max.scalar()
        if latest_sid is not None:
            conditions.append(RawTelemetryModel.segment_id == latest_sid)

    stmt_raw = (
        select(RawTelemetryModel)
        .where(*conditions)
        .order_by(RawTelemetryModel.segment_id.asc(), RawTelemetryModel.chunk_id.asc())
        .limit(300)
    )
    res_raw = await session.execute(stmt_raw)
    chunks = list(res_raw.scalars().all())
    if not chunks:
        raise ValueError(f"No raw telemetry found for node '{node_id}' in the requested time range")

    ax, ay, az = [], [], []
    gx, gy, gz = [], [], []
    for c in chunks:
        ax.extend(c.accel_x)
        ay.extend(c.accel_y)
        az.extend(c.accel_z)
        gx.extend(c.gyro_x)
        gy.extend(c.gyro_y)
        gz.extend(c.gyro_z)

    accel_mag = np.sqrt(np.array(ax, dtype=np.float32)**2 + np.array(ay, dtype=np.float32)**2 + np.array(az, dtype=np.float32)**2)
    gyro_mag = np.sqrt(np.array(gx, dtype=np.float32)**2 + np.array(gy, dtype=np.float32)**2 + np.array(gz, dtype=np.float32)**2)

    total_pts = len(accel_mag)
    win_size = 256
    target_frames = min(150, max(24, total_pts // 256))
    hop = max(64, (total_pts - win_size) // target_frames) if total_pts > win_size else 128

    def stft_calc(sig):
        window = np.hanning(win_size)
        frames = []
        for i in range(0, len(sig) - win_size + 1, hop):
            slice_ = sig[i:i+win_size] * window
            spec = np.abs(np.fft.rfft(slice_))
            frames.append(spec[:128])
        return np.log1p(np.array(frames, dtype=np.float32)) if frames else np.zeros((1, 128), dtype=np.float32)

    accel_stft = stft_calc(accel_mag)
    gyro_stft = stft_calc(gyro_mag)
    combined = np.hstack([accel_stft, gyro_stft]) # shape: (frames, 256)

async def flag_segments_in_range(
    session: AsyncSession,
    node_id: str,
    from_time_ms: Optional[int] = None,
    to_time_ms: Optional[int] = None,
    from_time_str: Optional[str] = None,
    to_time_str: Optional[str] = None,
    is_anomaly: bool = True,
    exclude_from_training: bool = True
) -> Dict[str, Any]:
    # manual tagging: flag all telemetry segments within time range as anomaly or nominal
    if from_time_ms is not None:
        dt_from = datetime.fromtimestamp(from_time_ms / 1000.0, tz=timezone.utc)
    elif from_time_str:
        dt_from = datetime.fromisoformat(from_time_str.replace("Z", "+00:00"))
    else:
        dt_from = datetime.min.replace(tzinfo=timezone.utc)

    if to_time_ms is not None:
        dt_to = datetime.fromtimestamp(to_time_ms / 1000.0, tz=timezone.utc)
    elif to_time_str:
        dt_to = datetime.fromisoformat(to_time_str.replace("Z", "+00:00"))
    else:
        dt_to = datetime.max.replace(tzinfo=timezone.utc)

    stmt_segs = (
        select(RawTelemetryModel.segment_id)
        .where(
            RawTelemetryModel.node_id == node_id,
            RawTelemetryModel.timestamp >= dt_from,
            RawTelemetryModel.timestamp <= dt_to
        )
        .group_by(RawTelemetryModel.segment_id)
    )
    res_segs = await session.execute(stmt_segs)
    segment_ids = list(res_segs.scalars().all())

    if not segment_ids:
        return {"node_id": node_id, "updated_segments_count": 0, "segment_ids": []}

    stmt_upd = (
        update(InferenceResultModel)
        .where(
            InferenceResultModel.node_id == node_id,
            InferenceResultModel.segment_id.in_(segment_ids)
        )
        .values(
            anomaly=is_anomaly,
            is_recalculated=True
        )
    )
    await session.execute(stmt_upd)
    await session.commit()

    return {
        "node_id": node_id,
        "is_anomaly": is_anomaly,
        "exclude_from_training": exclude_from_training,
        "updated_segments_count": len(segment_ids),
        "affected_segment_ids": segment_ids
    }


