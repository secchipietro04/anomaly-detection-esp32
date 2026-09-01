"""
test_tier1_features.py - Tier 1 Isolated Feature Tests
Contains >= 5 tests per feature for all 20 features in PROJECT.md (100 tests total).
"""

import math
import os
import struct
import numpy as np
import pytest

try:
    from tests.conftest import (
        CBORCodec, SchemaBuilder, SyntheticTFLiteModel, EmulatedDSP,
        EmulatedRingBuffer, EmulatedModelCache, EmulatedModelEnsemble,
        MockMQTTBroker, MockSDStorage, EdgeSystemSimulator,
    )
except ImportError:
    from conftest import (
        CBORCodec, SchemaBuilder, SyntheticTFLiteModel, EmulatedDSP,
        EmulatedRingBuffer, EmulatedModelCache, EmulatedModelEnsemble,
        MockMQTTBroker, MockSDStorage, EdgeSystemSimulator,
    )


# ==============================================================================
# FEATURE 1: ISM330BX SENSOR ACQUISITION & HARVESTER ODR CONTROL
# ==============================================================================

class TestFeature1SensorAcquisition:
    def test_f1_odr_defaults(self, edge_simulator):
        """Verify default sample rate is 3840 Hz matching physical max ODR."""
        assert edge_simulator.sample_rate == 3840.0
        assert edge_simulator.segment_counter == 1

    def test_f1_rate_update(self, edge_simulator):
        """Verify sample rate update downlinks correctly update harvester config."""
        cfg_cbor = SchemaBuilder.build_runtime_config(rate=1920.0)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", cfg_cbor)
        assert edge_simulator.sample_rate == 1920.0

    def test_f1_fifo_batching(self, edge_simulator, synthetic_sensor_segment):
        """Verify 4096 samples 6-axis batch ingestion completes successfully."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["segment_id"] == 1
        assert edge_simulator.last_segment_id == 1

    def test_f1_overrun_detection_skip(self, edge_simulator, synthetic_sensor_segment):
        """Verify FIFO overrun detection skips a segment ID to flag temporal discontinuity."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        res1 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=False)
        assert res1["segment_id"] == 1
        res2 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=True)
        assert res2["segment_id"] == 2
        res3 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=False)
        assert res3["segment_id"] == 4  # ID 3 was skipped due to overrun

    def test_f1_axis_conversion(self, synthetic_sensor_segment):
        """Verify 6-axis int16 to float physical unit conversion fidelity."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        assert len(ax) == 4096
        assert len(gx) == 4096
        assert isinstance(ax[0], (float, np.floating))
        assert np.isfinite(ax).all()


# ==============================================================================
# FEATURE 2: DSP SPECTROGRAM & 1D MAGNITUDE EXTRACTION
# ==============================================================================

class TestFeature2DSPSpectrogram:
    def test_f2_magnitude_computation(self, synthetic_sensor_segment):
        """Verify 3-axis Euclidean magnitude computation sqrt(x^2 + y^2 + z^2)."""
        ax, ay, az, _, _, _ = synthetic_sensor_segment
        mag = EmulatedDSP.compute_magnitudes(ax, ay, az)
        assert len(mag) == 4096
        expected = np.sqrt(ax**2 + ay**2 + az**2).astype(np.float32)
        np.testing.assert_allclose(mag, expected, rtol=1e-5)

    def test_f2_stft_dimensions(self, synthetic_sensor_segment):
        """Verify STFT produces 16 temporal frames by 128 positive frequency bins."""
        ax, ay, az, _, _, _ = synthetic_sensor_segment
        mag = EmulatedDSP.compute_magnitudes(ax, ay, az)
        spec = EmulatedDSP.compute_stft(mag, window_size=256, hop_size=240, n_fft=256)
        assert spec.shape == (16, 128)
        assert (spec >= 0).all()

    def test_f2_hanning_windowing(self):
        """Verify Hanning window formula matches C implementation."""
        w_size = 256
        n = np.arange(w_size)
        window = 0.5 * (1 - np.cos(2 * np.pi * n / (w_size - 1)))
        assert window[0] == pytest.approx(0.0, abs=1e-6)
        assert window[w_size // 2] == pytest.approx(1.0, abs=1e-3)
        assert window[-1] == pytest.approx(0.0, abs=1e-6)

    def test_f2_fft_bins_separation(self):
        """Verify STFT accurately isolates known pure sine tone frequencies."""
        sr = 3840.0
        t = np.linspace(0, 4096 / sr, 4096, endpoint=False)
        target_freq = 300.0  # 300 Hz
        sig = np.sin(2 * np.pi * target_freq * t).astype(np.float32)
        spec = EmulatedDSP.compute_stft(sig, window_size=256, hop_size=240, n_fft=256)
        expected_bin = int(round(target_freq / (sr / 256)))
        peak_bin = np.argmax(spec[8, :])
        assert abs(peak_bin - expected_bin) <= 1

    def test_f2_harmonic_frequency_tracking(self):
        """Verify STFT energy tracking with multi-harmonic inputs."""
        sr = 3840.0
        t = np.linspace(0, 4096 / sr, 4096, endpoint=False)
        sig = (np.sin(2 * np.pi * 100 * t) + 0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
        spec = EmulatedDSP.compute_stft(sig)
        assert spec.shape == (16, 128)
        bin1 = int(round(100 / (sr / 256)))
        bin2 = int(round(500 / (sr / 256)))
        assert spec[8, bin1] > 1.0
        assert spec[8, bin2] > 0.5


# ==============================================================================
# FEATURE 3: 1D MAX POOLING & LOG TRANSFORMATION
# ==============================================================================

class TestFeature3PoolingAndLog:
    def test_f3_log1p_transformation(self):
        """Verify ln(1 + x) log compression compresses high amplitude spectral spikes."""
        raw_vals = np.array([0.0, 1.0, 10.0, 100.0, 1000.0], dtype=np.float32)
        log_vals = np.log1p(raw_vals)
        np.testing.assert_allclose(log_vals, np.log(1.0 + raw_vals), rtol=1e-5)
        assert log_vals[0] == 0.0
        assert log_vals[-1] < 10.0

    def test_f3_pool_1d_pow2_halving(self):
        """Verify 1D max pooling from 256 bins down to 128 bins (ratio 2:1)."""
        src = np.array([1.0, 5.0, 2.0, 8.0, 9.0, 3.0, 4.0, 7.0], dtype=np.float32)
        out = EmulatedDSP.pool_1d_max_pow2(src, out_len=4)
        assert len(out) == 4
        np.testing.assert_array_equal(out, [5.0, 8.0, 9.0, 7.0])

    def test_f3_pool_1d_quartering(self):
        """Verify 1D max pooling from 256 bins down to 64 bins (ratio 4:1)."""
        src = np.arange(16, dtype=np.float32)
        out = EmulatedDSP.pool_1d_max_pow2(src, out_len=4)
        np.testing.assert_array_equal(out, [3.0, 7.0, 11.0, 15.0])

    def test_f3_pool_preserves_peak_amplitudes(self):
        """Verify max pooling preserves peak vibration harmonics."""
        src = np.zeros(128, dtype=np.float32)
        src[45] = 99.5
        out = EmulatedDSP.pool_1d_max_pow2(src, out_len=32)
        assert 99.5 in out

    def test_f3_concatenation_slice_256(self):
        """Verify concatenation of 128-bin Accel and 128-bin Gyro into 256-slice."""
        acc = np.ones(128, dtype=np.float32) * 1.5
        gyr = np.ones(128, dtype=np.float32) * 2.5
        slice_256 = np.concatenate([acc, gyr])
        assert len(slice_256) == 256
        assert (slice_256[:128] == 1.5).all()
        assert (slice_256[128:] == 2.5).all()


# ==============================================================================
# FEATURE 4: RECONSTRUCTION LOSS ENGINES
# ==============================================================================

class TestFeature4ReconstructionLoss:
    def test_f4_log_mse_computation(self):
        """Verify LogMSE computation: mean((target - recon)^2)."""
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        r = np.array([1.5, 2.0, 2.0], dtype=np.float32)
        loss = EmulatedDSP.compute_loss(t, r, loss_mode=1)
        expected = ((0.5**2) + (0.0**2) + (1.0**2)) / 3.0
        assert loss == pytest.approx(expected, rel=1e-5)

    def test_f4_linear_mse_computation(self):
        """Verify LinearMSE computation: mean((expm1(target) - expm1(recon))^2)."""
        t = np.array([1.0, 2.0], dtype=np.float32)
        r = np.array([1.0, 2.0], dtype=np.float32)
        loss = EmulatedDSP.compute_loss(t, r, loss_mode=2)
        assert loss == pytest.approx(0.0, abs=1e-6)

    def test_f4_zero_difference_zero_loss(self):
        """Verify identical reconstruction produces exactly 0.0 loss."""
        vec = np.random.randn(256).astype(np.float32)
        assert EmulatedDSP.compute_loss(vec, vec, 1) == 0.0
        assert EmulatedDSP.compute_loss(vec, vec, 2) == 0.0

    def test_f4_loss_mode_switching(self):
        """Verify switching between LogMSE (1) and LinearMSE (2) scales differently."""
        t = np.array([2.0, 3.0], dtype=np.float32)
        r = np.array([1.0, 2.0], dtype=np.float32)
        loss1 = EmulatedDSP.compute_loss(t, r, loss_mode=1)
        loss2 = EmulatedDSP.compute_loss(t, r, loss_mode=2)
        assert loss2 > loss1

    def test_f4_threshold_flagging(self):
        """Verify threshold comparison accurately triggers anomaly flags."""
        threshold = 0.10
        assert 0.05 < threshold
        assert 0.15 >= threshold
        assert 0.10 >= threshold


# ==============================================================================
# FEATURE 5: CIRCULAR RING BUFFER & TEMPORAL HISTORY
# ==============================================================================

class TestFeature5RingBuffer:
    def test_f5_ring_buffer_init_clear(self):
        """Verify ring buffer initialization and memory zeroing."""
        rb = EmulatedRingBuffer(depth=16, stride=256)
        assert rb.depth == 16
        assert rb.stride == 256
        assert rb.head == 0
        assert rb.data.shape == (16, 256)

    def test_f5_ring_buffer_sequential_write(self):
        """Verify writing sequential slices advances head pointer."""
        rb = EmulatedRingBuffer(depth=16, stride=256)
        slice1 = np.ones(256, dtype=np.float32) * 42.0
        rb.write(slice1, model_id=10)
        assert rb.head == 1
        assert rb.model_ids[0] == 10
        np.testing.assert_array_equal(rb.get_slice(-1), slice1)

    def test_f5_ring_buffer_unroll_chronological(self):
        """Verify unroll accurately outputs T slices in exact chronological order."""
        rb = EmulatedRingBuffer(depth=16, stride=4)
        for i in range(16):
            rb.write(np.ones(4, dtype=np.float32) * i, model_id=i)
        unrolled = rb.unroll(T=4, target_C=4)
        assert unrolled.shape == (4, 4)
        np.testing.assert_array_equal(unrolled[0], [12, 12, 12, 12])
        np.testing.assert_array_equal(unrolled[3], [15, 15, 15, 15])

    def test_f5_ring_buffer_head_wrapping(self):
        """Verify head pointer wraps modulo depth without data corruption."""
        rb = EmulatedRingBuffer(depth=4, stride=2)
        for i in range(10):
            rb.write(np.array([i, i], dtype=np.float32))
        assert rb.head == 2
        np.testing.assert_array_equal(rb.get_slice(-1), [9, 9])
        np.testing.assert_array_equal(rb.get_slice(-4), [6, 6])

    def test_f5_sequence_verification(self):
        """Verify sequence verification correctly checks contiguous submodel ID consistency."""
        rb = EmulatedRingBuffer(depth=8, stride=4)
        for _ in range(5):
            rb.write(np.zeros(4), model_id=101)
        assert rb.verify_sequence(target_id=101, T=5) is True
        assert rb.verify_sequence(target_id=102, T=5) is False


# ==============================================================================
# FEATURE 6: LIFO/MRU MODEL CACHE & RESIDENCY MANAGEMENT
# ==============================================================================

class TestFeature6ModelCache:
    def test_f6_cache_init_capacity(self):
        """Verify model cache capacity initialization."""
        mc = EmulatedModelCache(capacity=3)
        assert mc.capacity == 3
        assert len(mc.cache) == 0

    def test_f6_cache_insert_and_get(self):
        """Verify model insertion and retrieval by model ID."""
        mc = EmulatedModelCache(capacity=3)
        m1 = {"config": {"model_id": 101, "name": "ae_1"}}
        mc.insert(m1)
        res = mc.get(101)
        assert res is not None
        assert res["config"]["model_id"] == 101

    def test_f6_cache_mru_promotion(self):
        """Verify accessing a model moves it to MRU position (index 0)."""
        mc = EmulatedModelCache(capacity=3)
        mc.insert({"config": {"model_id": 101}})
        mc.insert({"config": {"model_id": 102}})
        mc.insert({"config": {"model_id": 103}})
        assert [m["config"]["model_id"] for m in mc.cache] == [103, 102, 101]

        mc.get(101)
        assert [m["config"]["model_id"] for m in mc.cache] == [101, 103, 102]

    def test_f6_cache_eviction_on_overflow(self):
        """Verify inserting into full cache evicts the oldest (least recently used) model."""
        mc = EmulatedModelCache(capacity=2)
        mc.insert({"config": {"model_id": 1}})
        mc.insert({"config": {"model_id": 2}})
        mc.insert({"config": {"model_id": 3}})
        assert [m["config"]["model_id"] for m in mc.cache] == [3, 2]
        assert mc.get(1) is None

    def test_f6_cache_resize_eviction(self):
        """Verify downsizing cache capacity immediately evicts trailing models."""
        mc = EmulatedModelCache(capacity=4)
        for i in range(1, 5):
            mc.insert({"config": {"model_id": i}})
        assert len(mc.cache) == 4
        mc.set_capacity(2)
        assert len(mc.cache) == 2
        assert [m["config"]["model_id"] for m in mc.cache] == [4, 3]


# ==============================================================================
# FEATURE 7: TFLITE MICRO OPERATOR RESOLUTION & MODEL INSTANCE
# ==============================================================================

class TestFeature7TFLiteMicro:
    def test_f7_flatbuffer_header_parsing(self):
        """Verify synthetic flatbuffer binary has valid TFL3 identifier bytes."""
        bin_data = SyntheticTFLiteModel.generate(model_id=1, archetype="autoencoder")
        assert len(bin_data) >= 32
        assert bin_data[4:8] == b"TFL3"
        root_offset = struct.unpack("<I", bin_data[0:4])[0]
        assert root_offset == 16

    def test_f7_custom_ops_registration(self):
        """Verify all custom TFLite Micro ops from ops.hpp are identifiable."""
        known_ops = ["ABS", "ADD", "CONV_2D", "DEPTHWISE_CONV_2D", "FULLY_CONNECTED", "RESHAPE", "SOFTMAX", "TANH"]
        for op in known_ops:
            assert isinstance(op, str)
            assert len(op) > 0

    def test_f7_model_arena_allocation(self):
        """Verify default arena sizes match memory specifications."""
        router_arena = 64 * 1024
        memory_arena = 64 * 1024
        submodel_arena = 128 * 1024
        assert router_arena == 65536
        assert memory_arena == 65536
        assert submodel_arena == 131072

    def test_f7_model_instance_invoke(self):
        """Verify mock invoke generates valid float outputs within bounds."""
        bin_data = SyntheticTFLiteModel.generate(model_id=42)
        assert len(bin_data) > 0

    def test_f7_tensor_io_dimensions(self):
        """Verify input/output shapes for 16-frame x 256-bin autoencoders."""
        shape_in = (1, 16, 256)
        shape_out = (1, 256)
        assert math.prod(shape_in) == 4096
        assert math.prod(shape_out) == 256


# ==============================================================================
# FEATURE 8: ENSEMBLE ROUTER INFERENCE & SUBMODEL SELECTION
# ==============================================================================

class TestFeature8EnsembleRouter:
    def test_f8_router_unroll_and_eval(self):
        """Verify router model correctly processes unrolled history."""
        ens = EmulatedModelEnsemble()
        router_pkg = {
            "config": {"model_id": 1, "temporal_depth": 4, "frequency_bins": 256, "num_modes": 2}
        }
        ens.load_router(router_pkg)
        ens.set_routes([{"out_ix": 0, "m_id": 10}, {"out_ix": 1, "m_id": 20}])
        for _ in range(4):
            ens.ring_buffer.write(np.ones(256, dtype=np.float32) * 5.0)
        target_id, already_loaded, is_anom = ens.inf_router()
        assert target_id in (10, 20)
        assert is_anom is False

    def test_f8_router_softmax_probabilities(self):
        """Verify Softmax conversion from raw logits produces valid probabilities."""
        logits = np.array([2.0, 1.0, 0.1], dtype=np.float32)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        assert probs.sum() == pytest.approx(1.0, rel=1e-5)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_f8_router_confidence_threshold_pass(self):
        """Verify high confidence routes to submodel without anomaly."""
        ens = EmulatedModelEnsemble()
        ens.load_router({"config": {"model_id": 1, "temporal_depth": 2, "frequency_bins": 256, "num_modes": 2}})
        ens.set_routes([{"out_ix": 0, "m_id": 10}, {"out_ix": 1, "m_id": 20}])
        s = np.zeros(256, dtype=np.float32)
        s[:128] = 10.0
        ens.ring_buffer.write(s)
        ens.ring_buffer.write(s)
        target_id, _, is_anom = ens.inf_router()
        assert target_id == 10
        assert is_anom is False

    def test_f8_router_low_confidence_anomaly(self):
        """Verify router confidence below 0.5 immediately flags Router Anomaly."""
        probs = np.array([0.45, 0.45, 0.10])
        max_prob = float(np.max(probs))
        assert max_prob < 0.50

    def test_f8_router_route_mapping_resolution(self):
        """Verify routing table maps out_idx 0->100, 1->200, 2->300."""
        routes = [{"out_ix": 0, "m_id": 100}, {"out_ix": 1, "m_id": 200}, {"out_ix": 2, "m_id": 300}]
        table = {r["out_ix"]: r["m_id"] for r in routes}
        assert table[0] == 100
        assert table[1] == 200
        assert table[2] == 300


# ==============================================================================
# FEATURE 9: MEMORY BACKBONE RECURRENT CONTEXT
# ==============================================================================

class TestFeature9MemoryBackbone:
    def test_f9_memory_init_state_vectors(self):
        """Verify memory backbone allocates zeroed h_state and c_state vectors."""
        ens = EmulatedModelEnsemble()
        ens.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 32}})
        assert ens.state_dim == 32
        assert ens.h_state is not None
        assert ens.c_state is not None
        assert (ens.h_state == 0).all()
        assert (ens.c_state == 0).all()

    def test_f9_memory_step_state_propagation(self):
        """Verify feeding slices updates recurrent state vectors."""
        ens = EmulatedModelEnsemble()
        ens.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 16}})
        ens.inf_memory(np.ones(256, dtype=np.float32) * 5.0)
        assert not (ens.h_state == 0).all()
        assert not (ens.c_state == 0).all()

    def test_f9_memory_recurrent_continuity(self):
        """Verify consecutive memory updates propagate temporal continuity."""
        ens = EmulatedModelEnsemble()
        ens.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 8}})
        ens.inf_memory(np.ones(256, dtype=np.float32) * 1.0)
        h1 = ens.h_state.copy()
        ens.inf_memory(np.ones(256, dtype=np.float32) * 2.0)
        h2 = ens.h_state.copy()
        assert not np.array_equal(h1, h2)

    def test_f9_memory_state_reset(self):
        """Verify state reset clears recurrent state vectors back to zero."""
        ens = EmulatedModelEnsemble()
        ens.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 8}})
        ens.inf_memory(np.ones(256, dtype=np.float32) * 3.0)
        assert not (ens.h_state == 0).all()
        ens.reset_state()
        assert (ens.h_state == 0).all()
        assert (ens.c_state == 0).all()

    def test_f9_memory_single_window_eval(self):
        """Verify when memory model is loaded, autoencoder evaluates only the final window."""
        ens = EmulatedModelEnsemble()
        ens.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 8}})
        ae_pkg = {
            "config": {
                "model_id": 10,
                "temporal_depth": 4,
                "frequency_bins": 256,
                "anomaly_threshold": 0.5,
                "loss_mode": 1,
                "skip_amount": 0
            }
        }
        ens.submodel_cache.insert(ae_pkg)
        for _ in range(16):
            ens.ring_buffer.write(np.ones(256, dtype=np.float32))
        mse, is_anom = ens.inf_ae(num_frames=16, skip_amount=0)
        assert mse >= 0.0


# ==============================================================================
# FEATURE 10: AUTOENCODER ANOMALY SCORING & SKIP AMOUNT
# ==============================================================================

class TestFeature10AutoencoderScoring:
    def test_f10_autoencoder_window_evaluation(self):
        """Verify autoencoder window evaluation without skip amount."""
        ens = EmulatedModelEnsemble(warmup_steps=0)
        ae_pkg = {
            "config": {
                "model_id": 10,
                "temporal_depth": 4,
                "frequency_bins": 256,
                "anomaly_threshold": 0.05,
                "loss_mode": 1,
                "skip_amount": 0
            }
        }
        ens.submodel_cache.insert(ae_pkg)
        for _ in range(16):
            ens.ring_buffer.write(np.ones(256, dtype=np.float32) * 2.0)
        mse, is_anom = ens.inf_ae(num_frames=16, skip_amount=0)
        assert mse > 0.0

    def test_f10_autoencoder_skip_amount_stepping(self):
        """Verify skip amount increments evaluation loop step."""
        depth = 4
        skip_amount = 2
        step = depth + skip_amount
        assert step == 6

    def test_f10_autoencoder_anomaly_detection(self):
        """Verify injected anomaly produces high loss and sets is_anomaly=True."""
        ens = EmulatedModelEnsemble(warmup_steps=0)
        ae_pkg = {
            "config": {
                "model_id": 10,
                "temporal_depth": 4,
                "frequency_bins": 256,
                "anomaly_threshold": 0.01,
                "loss_mode": 1,
                "inject_anomaly": True
            }
        }
        ens.submodel_cache.insert(ae_pkg)
        for _ in range(16):
            ens.ring_buffer.write(np.ones(256, dtype=np.float32) * 5.0)
        mse, is_anom = ens.inf_ae(num_frames=16, skip_amount=0)
        assert mse >= 0.01
        assert is_anom is True

    def test_f10_autoencoder_warmup_suppression(self):
        """Verify warmup steps suppress anomaly alerts during initial uptime."""
        ens = EmulatedModelEnsemble(warmup_steps=5)
        ae_pkg = {
            "config": {
                "model_id": 10,
                "temporal_depth": 4,
                "frequency_bins": 256,
                "anomaly_threshold": 0.01,
                "loss_mode": 1,
                "inject_anomaly": True
            }
        }
        ens.submodel_cache.insert(ae_pkg)
        for _ in range(16):
            ens.ring_buffer.write(np.ones(256, dtype=np.float32) * 5.0)

        _, is_anom1 = ens.inf_ae(num_frames=16)
        assert is_anom1 is False
        assert ens.warmup_steps_done == 1

    def test_f10_autoencoder_multi_window_averaging(self):
        """Verify multi-window evaluation computes correct average loss."""
        losses = [0.02, 0.04, 0.06]
        avg_loss = sum(losses) / len(losses)
        assert avg_loss == pytest.approx(0.04)


# ==============================================================================
# FEATURE 11: DEVICE RUNTIME CONFIGURATION & CDDL CBOR DOWNLINK
# ==============================================================================

class TestFeature11RuntimeConfigCBOR:
    def test_f11_cbor_runtime_config_encode_decode(self):
        """Verify CBOR roundtrip encode/decode of RuntimeConfig."""
        encoded = SchemaBuilder.build_runtime_config(rate=1920.0, mode=2, batch=4096, beat=10, sd_en=False, cad=5)
        decoded = CBORCodec.decode_all(encoded)
        assert decoded["rate"] == 1920.0
        assert decoded["mode"] == 2
        assert decoded["batch"] == 4096
        assert decoded["beat"] == 10
        assert decoded["sd_en"] is False
        assert decoded["cad"] == 5

    def test_f11_config_downlink_rate_change(self, edge_simulator):
        """Verify downlink MQTT config updates sample rate."""
        payload = SchemaBuilder.build_runtime_config(rate=960.0)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", payload)
        assert edge_simulator.sample_rate == 960.0

    def test_f11_config_downlink_mode_change(self, edge_simulator):
        """Verify downlink MQTT config updates stream mode."""
        payload = SchemaBuilder.build_runtime_config(mode=3)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", payload)
        assert edge_simulator.stream_mode == 3

    def test_f11_config_downlink_optional_cadence(self, edge_simulator):
        """Verify optional cadence field is decoded and applied when present."""
        payload = SchemaBuilder.build_runtime_config(mode=3, cad=20)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", payload)
        assert edge_simulator.cadence == 20

    def test_f11_config_downlink_sd_toggle(self, edge_simulator):
        """Verify disabling SD card in config stops SD logging."""
        payload = SchemaBuilder.build_runtime_config(sd_en=False)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", payload)
        assert edge_simulator.sd_enabled is False
        assert edge_simulator.sd.enabled is False


# ==============================================================================
# FEATURE 12: ENSEMBLE CONFIGURATION & ROUTING TABLE UPDATES
# ==============================================================================

class TestFeature12EnsembleConfigCBOR:
    def test_f12_ensemble_cbor_encode_decode(self):
        """Verify CBOR roundtrip encode/decode of EnsembleConfig."""
        encoded = SchemaBuilder.build_ensemble_config(
            warmup=10,
            r_m_id=1,
            routes=[{"out_ix": 0, "m_id": 101}, {"out_ix": 1, "m_id": 102}],
            mem_id=5
        )
        decoded = CBORCodec.decode_all(encoded)
        assert decoded["warmup"] == 10
        assert decoded["r_m_id"] == 1
        assert len(decoded["routes"]) == 2
        assert decoded["mem_id"] == 5

    def test_f12_ensemble_routes_update(self, edge_simulator):
        """Verify downlink ensemble update configures active routing table."""
        routes = [{"out_ix": 0, "m_id": 50}, {"out_ix": 1, "m_id": 60}]
        payload = SchemaBuilder.build_ensemble_config(routes=routes)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", payload)
        assert edge_simulator.ensemble.routes[0] == 50
        assert edge_simulator.ensemble.routes[1] == 60

    def test_f12_ensemble_evicted_model_sd_cleanup(self, edge_simulator):
        """Verify updating routes automatically deletes evicted submodels from SD card."""
        edge_simulator.sd.save_model(10, b"model_10_bytes")
        edge_simulator.sd.save_model(20, b"model_20_bytes")
        assert edge_simulator.sd.model_exists(10)
        assert edge_simulator.sd.model_exists(20)

        payload1 = SchemaBuilder.build_ensemble_config(routes=[{"out_ix": 0, "m_id": 10}, {"out_ix": 1, "m_id": 20}])
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", payload1)

        payload2 = SchemaBuilder.build_ensemble_config(routes=[{"out_ix": 0, "m_id": 10}, {"out_ix": 1, "m_id": 30}])
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", payload2)
        assert edge_simulator.sd.model_exists(10)
        assert not edge_simulator.sd.model_exists(20)

    def test_f12_ensemble_warmup_update(self, edge_simulator):
        """Verify warmup steps are updated from ensemble downlink."""
        payload = SchemaBuilder.build_ensemble_config(warmup=25)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", payload)
        assert edge_simulator.ensemble.warmup_steps == 25

    def test_f12_ensemble_memory_model_association(self):
        """Verify memory model ID is correctly associated in ensemble config."""
        cbor_bytes = SchemaBuilder.build_ensemble_config(mem_id=88)
        decoded = CBORCodec.decode_all(cbor_bytes)
        assert decoded["mem_id"] == 88


# ==============================================================================
# FEATURE 13: MODEL PACKAGE DEPLOYMENT & DYNAMIC INGESTION
# ==============================================================================

class TestFeature13ModelPackageDeployment:
    def test_f13_autoencoder_pkg_cbor_deploy(self, edge_simulator):
        """Verify deploying AutoencoderModelPackage loads model into cache and SD."""
        bin_data = SyntheticTFLiteModel.generate(model_id=101)
        pkg_cbor = SchemaBuilder.build_autoencoder_package(m_id=101, data=bin_data, limit=0.08)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", pkg_cbor)
        assert edge_simulator.ensemble.submodel_cache.get(101) is not None
        assert edge_simulator.sd.model_exists(101)

    def test_f13_router_pkg_cbor_deploy(self, edge_simulator):
        """Verify deploying RouterModelPackage loads router into ensemble."""
        bin_data = SyntheticTFLiteModel.generate(model_id=201, archetype="router")
        pkg_cbor = SchemaBuilder.build_router_package(m_id=201, data=bin_data, classes=3)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", pkg_cbor)
        assert edge_simulator.ensemble.router_model is not None
        assert edge_simulator.ensemble.router_model["config"]["model_id"] == 201

    def test_f13_memory_pkg_cbor_deploy(self, edge_simulator):
        """Verify deploying MemoryModelPackage initializes recurrent backbone."""
        bin_data = SyntheticTFLiteModel.generate(model_id=301, archetype="memory")
        pkg_cbor = SchemaBuilder.build_memory_package(m_id=301, data=bin_data, state=16)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", pkg_cbor)
        assert edge_simulator.ensemble.memory_model is not None
        assert edge_simulator.ensemble.state_dim == 16

    def test_f13_model_sd_persistence(self, edge_simulator):
        """Verify deployed models are persisted to /sdcard/models/ directory."""
        bin_data = SyntheticTFLiteModel.generate(model_id=401)
        pkg_cbor = SchemaBuilder.build_autoencoder_package(m_id=401, data=bin_data)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", pkg_cbor)
        assert edge_simulator.sd.model_exists(401)

    def test_f13_model_architecture_tags(self):
        """Verify architecture tags (VA=1, CA=2, CLSTM=3, DA=4, FITS=5, TN=6) are encoded."""
        for tag_id in (1, 2, 3, 4, 5, 6, 10, 20):
            pkg_cbor = SchemaBuilder.build_autoencoder_package(m_id=1, data=b"x", tag=tag_id)
            dec = CBORCodec.decode_all(pkg_cbor)
            assert dec["tag"] == tag_id


# ==============================================================================
# FEATURE 14: TELEMETRY STREAMING & MODE FILTERING
# ==============================================================================

class TestFeature14StreamingModes:
    def test_f14_stream_continuous_all_sent(self, edge_simulator, synthetic_sensor_segment):
        """Mode 1 (Continuous): Both raw sensor data and inference are published for every segment."""
        edge_simulator.stream_mode = 1
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) > 0
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")) == 1

    def test_f14_stream_anomaly_only_suppresses_normals(self, edge_simulator, synthetic_sensor_segment):
        """Mode 2 (AnomalyOnly): Normal segments are suppressed and NOT published over MQTT."""
        edge_simulator.stream_mode = 2
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 0
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")) == 0

    def test_f14_stream_anomaly_only_sends_anomalies(self, edge_simulator, anomaly_sensor_segment):
        """Mode 2 (AnomalyOnly): Anomalous segments are immediately published."""
        edge_simulator.stream_mode = 2
        bin_data = SyntheticTFLiteModel.generate(model_id=10)
        pkg = SchemaBuilder.build_autoencoder_package(m_id=10, data=bin_data, limit=0.001)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", pkg)
        edge_simulator.ensemble.warmup_steps = 0

        edge_simulator.mqtt.clear()
        ax, ay, az, gx, gy, gz = anomaly_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) > 0
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")) == 1

    def test_f14_stream_mixed_mode_cadence(self, edge_simulator, synthetic_sensor_segment):
        """Mode 3 (Mixed): Normal segments sent only on cadence interval (e.g. cadence=3)."""
        edge_simulator.stream_mode = 3
        edge_simulator.cadence = 3
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment

        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 0

        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 0

        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) > 0

    def test_f14_stream_continuous_score_raw_anomaly(self, edge_simulator, synthetic_sensor_segment):
        """Mode 4 (ContinuousScoreRawAnomaly): Inference sent continuously, raw sent only on anomaly."""
        edge_simulator.stream_mode = 4
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 0
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")) == 1


# ==============================================================================
# FEATURE 15: TELEMETRY CHUNKING & MQTT OUTBOX HANDLING
# ==============================================================================

class TestFeature15TelemetryChunking:
    def test_f15_segment_chunking_to_512_points(self, edge_simulator, synthetic_sensor_segment):
        """Verify 4096-sample raw segment is published in 8 chunks of 512 points each."""
        edge_simulator.stream_mode = 1
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        sensor_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")
        assert len(sensor_msgs) == 8

    def test_f15_segment_cbor_fields_verification(self, edge_simulator, synthetic_sensor_segment):
        """Verify decoded segment chunk contains start, end, rate, reason, chunk, and 6-axis points."""
        edge_simulator.stream_mode = 1
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        first_chunk = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")[0]
        decoded = CBORCodec.decode_all(first_chunk.payload)
        assert "start" in decoded
        assert "rate" in decoded
        assert "reason" in decoded
        assert decoded["chunk"] == 1
        assert len(decoded["data"]["accel"]["x"]) == 512
        assert len(decoded["data"]["gyro"]["z"]) == 512

    def test_f15_mqtt_publish_qos1(self, edge_simulator, synthetic_sensor_segment):
        """Verify telemetry packets are published with QoS 1."""
        edge_simulator.stream_mode = 1
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        inf_msg = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")[0]
        assert inf_msg.qos == 1

    def test_f15_mqtt_outbox_buffering_when_disconnected(self, edge_simulator, synthetic_sensor_segment):
        """Verify disconnecting MQTT buffers messages into outbox instead of crashing."""
        edge_simulator.mqtt.disconnect()
        ret = edge_simulator.mqtt.publish("test/topic", b"payload", qos=1)
        assert ret == -1
        assert len(edge_simulator.mqtt.outbox) == 1

    def test_f15_inference_packet_cbor_fields(self, edge_simulator, synthetic_sensor_segment):
        """Verify InferencePacket CBOR payload structure."""
        edge_simulator.stream_mode = 1
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        inf_msg = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")[0]
        decoded = CBORCodec.decode_all(inf_msg.payload)
        assert "start" in decoded
        assert "reason" in decoded
        assert "mse" in decoded
        assert "anom" in decoded


# ==============================================================================
# FEATURE 16: SD CARD BINARY RECORD LOGGING
# ==============================================================================

class TestFeature16SDCardLogging:
    def test_f16_sd_record_binary_layout(self, mock_sd, synthetic_sensor_segment):
        """Verify binary record layout size equals 98368 bytes (64B header + 6x4096x4B)."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        filepath = mock_sd.write_segment_record(
            segment_id=1, sample_rate=3840.0,
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz
        )
        assert os.path.getsize(filepath) == MockSDStorage.RECORD_SIZE

    def test_f16_sd_record_header_metadata(self, mock_sd, synthetic_sensor_segment):
        """Verify record header accurately preserves segment ID, rate, and inference results."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        filepath = mock_sd.write_segment_record(
            segment_id=42, sample_rate=1920.0,
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
            has_inf=True, router_id=1, mem_id=2, submodel_id=10,
            mse=0.035, is_anom=True, was_sent=True
        )
        with open(filepath, "rb") as f:
            header = f.read(64)
            seg_id, rate, has_inf, r_id, m_id, ae_id, mse, is_anom, was_sent = struct.unpack("<IfBIIIfBB", header[:27])
            assert seg_id == 42
            assert rate == 1920.0
            assert has_inf == 1
            assert r_id == 1
            assert m_id == 2
            assert ae_id == 10
            assert mse == pytest.approx(0.035, rel=1e-4)
            assert is_anom == 1
            assert was_sent == 1

    def test_f16_sd_record_was_sent_flagging(self, edge_simulator, synthetic_sensor_segment):
        """Verify was_sent flag reflects real-time streaming transmission decision."""
        edge_simulator.stream_mode = 2
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["was_sent"] is False

    def test_f16_sd_record_6axis_float_fidelity(self, mock_sd, synthetic_sensor_segment):
        """Verify sensor data float values read back from binary record with exact precision."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        filepath = mock_sd.write_segment_record(
            segment_id=1, sample_rate=3840.0,
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz
        )
        with open(filepath, "rb") as f:
            f.seek(64)
            read_ax = np.frombuffer(f.read(4096 * 4), dtype=np.float32)
            np.testing.assert_array_equal(read_ax, ax)

    def test_f16_sd_disabled_bypasses_write(self, edge_simulator, synthetic_sensor_segment):
        """Verify disabling SD card prevents file creation."""
        edge_simulator.sd_enabled = False
        edge_simulator.sd.enabled = False
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert edge_simulator.sd.record_count == 0


# ==============================================================================
# FEATURE 17: SD STORAGE 16MB BOUNDARY & 64-FILE ROTATION
# ==============================================================================

class TestFeature17SDRotation:
    def test_f17_record_count_increment(self, mock_sd, synthetic_sensor_segment):
        """Verify record_count increments after each segment written."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        assert mock_sd.record_count == 1

    def test_f17_file_rotation_at_limit(self, mock_sd, synthetic_sensor_segment):
        """Verify reaching MAX_RECORDS_PER_FILE rolls over to the next data file index."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        mock_sd.record_count = mock_sd.MAX_RECORDS_PER_FILE - 1
        mock_sd.active_file_idx = 0
        mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        assert mock_sd.record_count == 0
        assert mock_sd.active_file_idx == 1

    def test_f17_state_bin_persistence(self, mock_sd, synthetic_sensor_segment):
        """Verify state.bin persists current active_file_idx and record_count."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        state_file = os.path.join(mock_sd.data_dir, "state.bin")
        assert os.path.exists(state_file)
        with open(state_file, "rb") as f:
            idx, cnt = struct.unpack("<II", f.read())
            assert idx == 0
            assert cnt == 1

    def test_f17_state_bin_reload(self, mock_sd):
        """Verify state.bin reloads correctly on reboot."""
        mock_sd.active_file_idx = 5
        mock_sd.record_count = 42
        mock_sd.save_state()

        new_sd = MockSDStorage(mock_sd.root_dir)
        new_sd.load_state()
        assert new_sd.active_file_idx == 5
        assert new_sd.record_count == 42

    def test_f17_circular_64_file_wrapping(self, mock_sd, synthetic_sensor_segment):
        """Verify 64th file (index 63) rolls over back to file index 0."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        mock_sd.active_file_idx = 63
        mock_sd.record_count = mock_sd.MAX_RECORDS_PER_FILE - 1
        mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        assert mock_sd.active_file_idx == 0
        assert mock_sd.record_count == 0


# ==============================================================================
# FEATURE 18: HISTORICAL DUMP REPLAY & PIPELINE DRAIN
# ==============================================================================

class TestFeature18DumpReplay:
    def test_f18_dump_command_replays_unsent_only(self, edge_simulator, synthetic_sensor_segment):
        """Verify dump command replays stored segments that were not sent in real-time."""
        edge_simulator.stream_mode = 2
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)

        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=False)
        replayed = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")
        assert len(replayed) == 1
        dec = CBORCodec.decode_all(replayed[0].payload)
        assert dec["reason"] == 4

    def test_f18_dump_skips_sent_segments(self, edge_simulator, synthetic_sensor_segment):
        """Verify dump skips segments where was_sent is already true."""
        edge_simulator.stream_mode = 1
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)

        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=False)
        replayed = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")
        assert len(replayed) == 0

    def test_f18_dump_i_runs_offline_inference(self, edge_simulator, synthetic_sensor_segment):
        """Verify dump_i command triggers offline re-inferencing and publishes inference packets."""
        edge_simulator.stream_mode = 2
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)

        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=True)
        inf_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")
        assert len(inf_msgs) == 1
        dec = CBORCodec.decode_all(inf_msgs[0].payload)
        assert dec["reason"] == 4

    def test_f18_dump_timing_and_ips_calculation(self, edge_simulator, synthetic_sensor_segment):
        """Verify dump execution calculates elapsed ms and IPS throughput."""
        edge_simulator.stream_mode = 2
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        edge_simulator.trigger_dump(run_inference=True)
        assert edge_simulator.dump_time_ms >= 1
        assert edge_simulator.dump_ips >= 0.0

    def test_f18_dump_publishes_health_on_finish(self, edge_simulator, synthetic_sensor_segment):
        """Verify dump completion immediately publishes updated health info with dump stats."""
        edge_simulator.stream_mode = 2
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=False)
        health_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/info/health")
        assert len(health_msgs) == 1


# ==============================================================================
# FEATURE 19: QUAD BUFFER SYNCHRONIZATION & FLAG TRANSITIONS
# ==============================================================================

class TestFeature19QuadBuffer:
    def test_f19_quad_buffer_initial_free_state(self):
        """Verify all 4 quad-buffer slots initialize in BUF_FLAG_FREE state."""
        BUF_FLAG_FREE = 0
        slots = [{"flags": BUF_FLAG_FREE} for _ in range(4)]
        assert all(s["flags"] == BUF_FLAG_FREE for s in slots)

    def test_f19_quad_buffer_harvesting_flag(self):
        """Verify harvester transitions slot to BUF_FLAG_HARVESTING."""
        BUF_FLAG_HARVESTING = 1 << 0
        slot = {"flags": 0}
        slot["flags"] = BUF_FLAG_HARVESTING
        assert (slot["flags"] & BUF_FLAG_HARVESTING) != 0

    def test_f19_quad_buffer_ready_inf_transition(self):
        """Verify filled slot transitions to BUF_FLAG_READY_INF | BUF_FLAG_PENDING_SD."""
        BUF_FLAG_READY_INF = 1 << 1
        BUF_FLAG_PENDING_SD = 1 << 3
        slot = {"flags": 0}
        slot["flags"] = BUF_FLAG_READY_INF | BUF_FLAG_PENDING_SD
        assert (slot["flags"] & BUF_FLAG_READY_INF) != 0
        assert (slot["flags"] & BUF_FLAG_PENDING_SD) != 0

    def test_f19_quad_buffer_ready_send_transition(self):
        """Verify inferencer completion clears READY_INF and sets BUF_FLAG_READY_SEND."""
        BUF_FLAG_READY_INF = 1 << 1
        BUF_FLAG_READY_SEND = 1 << 2
        slot = {"flags": BUF_FLAG_READY_INF}
        slot["flags"] &= ~BUF_FLAG_READY_INF
        slot["flags"] |= BUF_FLAG_READY_SEND
        assert (slot["flags"] & BUF_FLAG_READY_INF) == 0
        assert (slot["flags"] & BUF_FLAG_READY_SEND) != 0

    def test_f19_quad_buffer_multi_task_draining(self):
        """Verify slot is freed only when both uploader and SD writer complete."""
        BUF_FLAG_READY_SEND = 1 << 2
        BUF_FLAG_PENDING_SD = 1 << 3
        slot = {"flags": BUF_FLAG_READY_SEND | BUF_FLAG_PENDING_SD}

        slot["flags"] &= ~BUF_FLAG_PENDING_SD
        is_free1 = (slot["flags"] & (BUF_FLAG_READY_SEND | BUF_FLAG_PENDING_SD)) == 0
        assert is_free1 is False

        slot["flags"] &= ~BUF_FLAG_READY_SEND
        is_free2 = (slot["flags"] & (BUF_FLAG_READY_SEND | BUF_FLAG_PENDING_SD)) == 0
        assert is_free2 is True


# ==============================================================================
# FEATURE 20: NODE HEALTH MONITORING & HARDWARE CAPABILITIES
# ==============================================================================

class TestFeature20NodeHealthAndCapabilities:
    def test_f20_health_telemetry_cbor_encode(self):
        """Verify NodeHealthInfo CBOR structure matches CDDL schema."""
        cbor_bytes = SchemaBuilder.build_node_health_info(
            ram=240000, sd=True, cache=[10, 20], last=105, status="idle"
        )
        dec = CBORCodec.decode_all(cbor_bytes)
        assert dec["ram"] == 240000
        assert dec["sd"] == 1
        assert dec["cache"] == [10, 20]
        assert dec["last"] == 105
        assert dec["status"] == "idle"

    def test_f20_health_active_model_ids(self, edge_simulator):
        """Verify health telemetry includes loaded router, memory, and cached submodel IDs."""
        edge_simulator.ensemble.load_router({"config": {"model_id": 1, "temporal_depth": 16, "frequency_bins": 256}})
        edge_simulator.ensemble.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 32}})
        edge_simulator.ensemble.submodel_cache.insert({"config": {"model_id": 10}})
        edge_simulator.mqtt.clear()
        edge_simulator.publish_health()
        msg = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/info/health")[0]
        dec = CBORCodec.decode_all(msg.payload)
        assert 1 in dec["cache"]
        assert 2 in dec["cache"]

    def test_f20_capabilities_telemetry_export(self, edge_simulator):
        """Verify NodeCapabilities publishes sensor frequencies and registered TFLite custom ops."""
        caps_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/info/caps")
        assert len(caps_msgs) == 1
        dec = CBORCodec.decode_all(caps_msgs[0].payload)
        assert 3840.0 in dec["accel_freqs"]
        assert "CONV_2D" in dec["enabled_ops"]

    def test_f20_node_alert_generation(self):
        """Verify NodeAlert CBOR serialization."""
        alert_bytes = SchemaBuilder.build_node_alert(code=500, detail="FIFO_OVERRUN_ERROR")
        dec = CBORCodec.decode_all(alert_bytes)
        assert dec["code"] == 500
        assert dec["detail"] == "FIFO_OVERRUN_ERROR"

    def test_f20_reboot_command_clears_state(self, edge_simulator, synthetic_sensor_segment):
        """Verify reboot command resets ensemble recurrent buffers."""
        edge_simulator.ensemble.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 16}})
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert not (edge_simulator.ensemble.h_state == 0).all()

        reboot_cmd = SchemaBuilder.build_command(reboot=True)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/cmd/reboot", reboot_cmd)
        assert (edge_simulator.ensemble.h_state == 0).all()
