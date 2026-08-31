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

def test_ensemble_inference_flow():
    from app.database.models import EnsembleConfigModel
    from app.api.recalculator import run_ensemble_inference_on_window
    
    ensemble = EnsembleConfigModel(
        id=999,
        node_id="node_01",
        router_model_id=1001,
        memory_model_id=None,
        warmup=5,
        routes=[{"out_ix": 0, "m_id": 2001}, {"out_ix": 1, "m_id": 2002}],
        total_size_bytes=50000
    )
    
    # dummy models
    input_arr = np.ones((8, 256), dtype=np.float32)
    model_bins = {1001: b"dummy_router", 2001: b"dummy_ae1", 2002: b"dummy_ae2"}
    model_cfgs = {2001: {"limit": 0.2}, 2002: {"limit": 0.2}}
    
    mse, is_anom, r_id, ae_id, h_state, c_state = run_ensemble_inference_on_window(
        ensemble=ensemble,
        model_binaries=model_bins,
        model_configs=model_cfgs,
        input_arr=input_arr
    )
    
    assert isinstance(mse, float)
    assert isinstance(is_anom, (bool, np.bool_))
    assert r_id == 1001
    assert ae_id in (2001, 2002) or ae_id is None

import pytest
from unittest.mock import AsyncMock, MagicMock

@pytest.mark.asyncio
async def test_base2_noise_resilience_and_cross_specificity():
    from app.database.models import EnsembleConfigModel, ModelPackageModel, RawTelemetryModel
    from app.api.recalculator import test_noise_resilience, test_cross_submodel_specificity

    session = AsyncMock()

    # mock active ensemble
    mock_ens = EnsembleConfigModel(
        id=999,
        node_id="node_01",
        router_model_id=1001,
        memory_model_id=None,
        warmup=5,
        routes=[{"out_ix": 0, "m_id": 2001}, {"out_ix": 1, "m_id": 2002}],
        total_size_bytes=50000
    )

    mock_models = [
        ModelPackageModel(id=1001, node_id="node_01", tflite_binary=b"dummy_r", config={}),
        ModelPackageModel(id=2001, node_id="node_01", tflite_binary=b"dummy_ae1", config={"limit": 0.2}),
        ModelPackageModel(id=2002, node_id="node_01", tflite_binary=b"dummy_ae2", config={"limit": 0.2}),
    ]

    mock_chunks = [
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

    # mock database query sequences
    async def mock_execute(stmt):
        mock_res = MagicMock()
        stmt_str = str(stmt)
        if "ensembles" in stmt_str:
            mock_res.scalar_one_or_none.return_value = mock_ens
        elif "models" in stmt_str:
            mock_res.scalars.return_value.all.return_value = mock_models
        elif "DISTINCT" in stmt_str or "distinct" in stmt_str:
            mock_res.scalars.return_value.all.return_value = [1, 2]
        else:
            mock_res.scalars.return_value.all.return_value = mock_chunks
        return mock_res

    session.execute.side_effect = mock_execute

    # test base-2 resilience
    resilience_report = await test_noise_resilience(
        session=session,
        node_id="node_01",
        num_sample_segments=2,
        num_noise_levels=8,
        min_power_2=-4.0,
        max_power_2=0.0,
        tsteps=8
    )

    assert resilience_report["noise_scale_base"] == 2
    assert len(resilience_report["powers_of_2"]) == 8
    assert len(resilience_report["sensitivity_spectrogram"]) == 8
    assert "metrics" in resilience_report

    # test cross specificity matrix
    specificity_report = await test_cross_submodel_specificity(
        session=session,
        node_id="node_01",
        num_samples_per_mode=2,
        tsteps=8
    )

    assert specificity_report["num_submodels"] == 2
    assert len(specificity_report["cross_reconstruction_mse_matrix"]) == 2
    assert len(specificity_report["cross_anomaly_rejection_rate_percent"]) == 2
    assert "mode_separability_index" in specificity_report["metrics"]
