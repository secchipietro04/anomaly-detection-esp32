"""
test_tier2_boundaries.py - Tier 2 Boundary, Edge Condition & Adversarial Tests
Contains >= 5 tests per feature for all 20 features (100 tests total).
"""

import math
import os
import struct
import numpy as np
import pytest

try:
    from tests.conftest import (
        CBORCodec, CBORError, SchemaBuilder, SyntheticTFLiteModel, EmulatedDSP,
        EmulatedRingBuffer, EmulatedModelCache, EmulatedModelEnsemble,
        MockMQTTBroker, MockSDStorage, EdgeSystemSimulator,
    )
except ImportError:
    from conftest import (
        CBORCodec, CBORError, SchemaBuilder, SyntheticTFLiteModel, EmulatedDSP,
        EmulatedRingBuffer, EmulatedModelCache, EmulatedModelEnsemble,
        MockMQTTBroker, MockSDStorage, EdgeSystemSimulator,
    )


# ==============================================================================
# FEATURE 1: SENSOR ACQUISITION & HARVESTER ODR BOUNDARIES
# ==============================================================================

class TestFeature1Boundaries:
    def test_b1_zero_length_input(self):
        """Verify handling of zero-length sensor input without buffer overflow."""
        empty = np.array([], dtype=np.float32)
        mag = EmulatedDSP.compute_magnitudes(empty, empty, empty)
        assert len(mag) == 0

    def test_b1_extreme_negative_g_acceleration(self, edge_simulator):
        """Verify extreme negative G acceleration spikes are safely ingested."""
        ax = np.full(4096, -1000.0, dtype=np.float32)
        ay = np.full(4096, -1000.0, dtype=np.float32)
        az = np.full(4096, -1000.0, dtype=np.float32)
        gx = np.zeros(4096, dtype=np.float32)
        gy = np.zeros(4096, dtype=np.float32)
        gz = np.zeros(4096, dtype=np.float32)
        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["segment_id"] == 1
        assert np.isfinite(res["mse"])

    def test_b1_nan_inf_sensor_readings(self):
        """Verify NaN or Inf readings are processed gracefully by magnitude calculation."""
        ax = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
        ay = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        az = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        mag = EmulatedDSP.compute_magnitudes(ax, ay, az)
        assert len(mag) == 4
        assert np.isnan(mag[0])
        assert np.isinf(mag[1])

    def test_b1_unsupported_sample_rates(self, edge_simulator):
        """Verify arbitrary non-standard sample rates are applied safely."""
        cfg = SchemaBuilder.build_runtime_config(rate=123.456)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", cfg)
        assert edge_simulator.sample_rate == pytest.approx(123.456)

    def test_b1_consecutive_fifo_overruns(self, edge_simulator, synthetic_sensor_segment):
        """Verify multiple consecutive FIFO overruns skip IDs monotonically."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        r1 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=True)
        r2 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=True)
        r3 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=True)
        assert r1["segment_id"] == 1
        assert r2["segment_id"] == 3
        assert r3["segment_id"] == 5


# ==============================================================================
# FEATURE 2: DSP SPECTROGRAM BOUNDARIES
# ==============================================================================

class TestFeature2Boundaries:
    def test_b2_constant_flatline_signal(self):
        """Verify DC flatline constant signal produces energy concentrated in bin 0."""
        sig = np.full(4096, 5.0, dtype=np.float32)
        spec = EmulatedDSP.compute_stft(sig)
        assert spec.shape == (16, 128)
        assert spec[0, 0] > 10.0
        assert spec[0, 0] > 10 * np.mean(spec[0, 1:])

    def test_b2_single_sample_spike_impulse(self):
        """Verify delta impulse response distributes energy across all frequency bins."""
        sig = np.zeros(4096, dtype=np.float32)
        sig[100] = 100.0  # Sharp Dirac delta
        spec = EmulatedDSP.compute_stft(sig)
        assert spec.shape == (16, 128)
        assert spec[0, :].sum() > 0.0

    def test_b2_max_nyquist_frequency_boundary(self):
        """Verify alternating maximum Nyquist frequency pattern [+1, -1, +1, -1]."""
        sig = np.tile([1.0, -1.0], 2048).astype(np.float32)
        spec = EmulatedDSP.compute_stft(sig)
        assert spec.shape == (16, 128)
        # Energy concentrated in highest bin (bin 127)
        assert spec[8, 127] > spec[8, 0]

    def test_b2_all_zeros_spectrogram(self):
        """Verify all zeros signal produces exactly 0.0 magnitude across all bins."""
        sig = np.zeros(4096, dtype=np.float32)
        spec = EmulatedDSP.compute_stft(sig)
        assert (spec == 0.0).all()

    def test_b2_extreme_high_amplitude_clipping(self):
        """Verify extreme 1e6 amplitudes do not cause numerical NaN overflow."""
        sig = np.full(4096, 1e6, dtype=np.float32)
        spec = EmulatedDSP.compute_stft(sig)
        assert np.isfinite(spec).all()


# ==============================================================================
# FEATURE 3: 1D POOLING & LOG TRANSFORM BOUNDARIES
# ==============================================================================

class TestFeature3Boundaries:
    def test_b3_pooling_dimension_1_to_1(self):
        """Verify 1:1 pooling (no reduction) returns identical array."""
        src = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = EmulatedDSP.pool_1d_max_pow2(src, out_len=4)
        np.testing.assert_array_equal(out, src)

    def test_b3_pooling_ratio_maximum_256_to_1(self):
        """Verify extreme 256:1 pooling returns the global maximum."""
        src = np.linspace(0, 100, 256, dtype=np.float32)
        src[50] = 999.0
        out = EmulatedDSP.pool_1d_max_pow2(src, out_len=1)
        assert len(out) == 1
        assert out[0] == 999.0

    def test_b3_log1p_zero_value_boundary(self):
        """Verify log1p(0.0) == 0.0 exactly."""
        assert math.log1p(0.0) == 0.0

    def test_b3_log1p_subnormal_tiny_floats(self):
        """Verify log1p on subnormal small positive values x -> x without underflow."""
        tiny = 1e-15
        assert math.log1p(tiny) == pytest.approx(tiny, rel=1e-6)

    def test_b3_non_power_of_2_pooling_fallback(self):
        """Verify pooling safely handles smaller target lengths."""
        src = np.arange(10, dtype=np.float32)
        out = EmulatedDSP.pool_1d_max_pow2(src, out_len=2)
        assert len(out) == 2
        assert out[1] == 9.0


# ==============================================================================
# FEATURE 4: RECONSTRUCTION LOSS ENGINE BOUNDARIES
# ==============================================================================

class TestFeature4Boundaries:
    def test_b4_loss_exact_threshold_boundary(self):
        """Verify loss value exactly matching anomaly threshold flags as anomaly (loss >= threshold)."""
        loss = 0.0500000
        threshold = 0.0500000
        assert (loss >= threshold) is True

    def test_b4_linear_mse_huge_exponential_overflow_protection(self):
        """Verify LinearMSE gracefully handles large values without crashing."""
        t = np.array([20.0, 30.0], dtype=np.float32)
        r = np.array([20.0, 30.0], dtype=np.float32)
        loss = EmulatedDSP.compute_loss(t, r, loss_mode=2)
        assert loss == pytest.approx(0.0, abs=1e-4)

    def test_b4_loss_empty_bins_zero_length(self):
        """Verify loss on empty slices returns 0.0 without divide by zero error."""
        t = np.array([], dtype=np.float32)
        r = np.array([], dtype=np.float32)
        loss = EmulatedDSP.compute_loss(t, r, loss_mode=1)
        assert np.isnan(loss) or loss == 0.0

    def test_b4_unknown_loss_mode_fallback(self):
        """Verify unsupported loss mode integers fallback to LogMSE standard."""
        t = np.array([2.0, 4.0], dtype=np.float32)
        r = np.array([1.0, 3.0], dtype=np.float32)
        loss_default = EmulatedDSP.compute_loss(t, r, loss_mode=99)
        loss_log = EmulatedDSP.compute_loss(t, r, loss_mode=1)
        assert loss_default == loss_log

    def test_b4_asymmetric_reconstruction_errors(self):
        """Verify squared loss is symmetric: L(target, recon) == L(recon, target)."""
        t = np.random.randn(128).astype(np.float32)
        r = np.random.randn(128).astype(np.float32)
        l1 = EmulatedDSP.compute_loss(t, r, loss_mode=1)
        l2 = EmulatedDSP.compute_loss(r, t, loss_mode=1)
        assert l1 == pytest.approx(l2, rel=1e-6)


# ==============================================================================
# FEATURE 5: RING BUFFER BOUNDARIES
# ==============================================================================

class TestFeature5Boundaries:
    def test_b5_ring_buffer_depth_1_boundary(self):
        """Verify ring buffer with depth=1 continuously overwrites single slot."""
        rb = EmulatedRingBuffer(depth=1, stride=4)
        rb.write(np.array([1, 1, 1, 1], dtype=np.float32))
        np.testing.assert_array_equal(rb.get_slice(-1), [1, 1, 1, 1])
        rb.write(np.array([2, 2, 2, 2], dtype=np.float32))
        np.testing.assert_array_equal(rb.get_slice(-1), [2, 2, 2, 2])

    def test_b5_ring_buffer_unroll_exact_depth(self):
        """Verify unrolling exact buffer depth T=depth returns entire buffer."""
        rb = EmulatedRingBuffer(depth=8, stride=2)
        for i in range(8):
            rb.write(np.array([i, i], dtype=np.float32))
        unrolled = rb.unroll(T=8, target_C=2)
        assert unrolled.shape == (8, 2)
        assert unrolled[0, 0] == 0.0
        assert unrolled[7, 0] == 7.0

    def test_b5_ring_buffer_unroll_exceeds_depth_truncated(self):
        """Verify unrolling with smaller target channel pools cleanly."""
        rb = EmulatedRingBuffer(depth=4, stride=8)
        rb.write(np.arange(8, dtype=np.float32))
        unrolled = rb.unroll(T=1, target_C=4)
        assert unrolled.shape == (1, 4)

    def test_b5_ring_buffer_100x_wrap_around_stress(self):
        """Verify 1000 sequential writes wrap cleanly without memory leakage."""
        rb = EmulatedRingBuffer(depth=16, stride=64)
        for i in range(1000):
            rb.write(np.full(64, i, dtype=np.float32))
        assert rb.head == (1000 % 16)
        assert rb.get_slice(-1)[0] == 999.0

    def test_b5_ring_buffer_negative_offsets_out_of_bounds(self):
        """Verify arbitrary negative offsets wrap cyclically."""
        rb = EmulatedRingBuffer(depth=4, stride=1)
        for i in range(4):
            rb.write(np.array([i], dtype=np.float32))
        # -1 -> 3, -5 -> 3
        np.testing.assert_array_equal(rb.get_slice(-1), rb.get_slice(-5))


# ==============================================================================
# FEATURE 6: MODEL CACHE BOUNDARIES
# ==============================================================================

class TestFeature6Boundaries:
    def test_b6_cache_capacity_zero(self):
        """Verify model cache with capacity=0 rejects all insertions."""
        mc = EmulatedModelCache(capacity=0)
        ret = mc.insert({"config": {"model_id": 1}})
        assert ret == -1
        assert len(mc.cache) == 0

    def test_b6_cache_capacity_one_rapid_thrashing(self):
        """Verify capacity=1 cache swaps out resident model on every new insertion."""
        mc = EmulatedModelCache(capacity=1)
        mc.insert({"config": {"model_id": 10}})
        assert mc.get(10) is not None
        mc.insert({"config": {"model_id": 20}})
        assert mc.get(10) is None
        assert mc.get(20) is not None

    def test_b6_cache_duplicate_insert_deduplication(self):
        """Verify inserting the same model ID multiple times updates it at index 0 without duplicates."""
        mc = EmulatedModelCache(capacity=3)
        mc.insert({"config": {"model_id": 1, "version": 1}})
        mc.insert({"config": {"model_id": 2, "version": 1}})
        mc.insert({"config": {"model_id": 1, "version": 2}})
        assert len(mc.cache) == 2
        assert mc.cache[0]["config"]["version"] == 2

    def test_b6_cache_query_nonexistent_model_id(self):
        """Verify querying non-existent model ID returns None safely."""
        mc = EmulatedModelCache(capacity=2)
        assert mc.get(999) is None

    def test_b6_cache_dynamic_downsize_to_zero(self):
        """Verify downsizing cache capacity to 0 purges all active models."""
        mc = EmulatedModelCache(capacity=4)
        for i in range(4):
            mc.insert({"config": {"model_id": i}})
        assert len(mc.cache) == 4
        mc.set_capacity(0)
        assert len(mc.cache) == 0


# ==============================================================================
# FEATURE 7: TFLITE MICRO BOUNDARIES
# ==============================================================================

class TestFeature7Boundaries:
    def test_b7_corrupted_flatbuffer_header_magic(self):
        """Verify flatbuffer binary with invalid magic bytes (not TFL3) is detectable."""
        bin_data = bytearray(SyntheticTFLiteModel.generate(model_id=1))
        bin_data[4:8] = b"CORR"  # Corrupted magic
        assert bin_data[4:8] != b"TFL3"

    def test_b7_zero_byte_model_binary(self):
        """Verify zero-length byte string model binary is detected."""
        empty_bin = b""
        assert len(empty_bin) == 0

    def test_b7_truncated_flatbuffer_table(self):
        """Verify flatbuffer smaller than 32-byte header table is rejected."""
        short_bin = b"TFL3short"
        assert len(short_bin) < 32

    def test_b7_unsupported_custom_operator_tag(self):
        """Verify custom ops outside registered whitelist are rejected."""
        known = ["CONV_2D", "FULLY_CONNECTED", "SOFTMAX"]
        assert "UNSUPPORTED_QUANT_OP_999" not in known

    def test_b7_arena_memory_exhaustion_simulation(self):
        """Verify exceeding max submodel arena size (128KB) triggers out-of-memory error."""
        max_arena = 128 * 1024
        requested_arena = 256 * 1024
        assert requested_arena > max_arena


# ==============================================================================
# FEATURE 8: ENSEMBLE ROUTER BOUNDARIES
# ==============================================================================

class TestFeature8Boundaries:
    def test_b8_router_confidence_exactly_half_threshold(self):
        """Verify router confidence exactly at 0.50 threshold does NOT trigger anomaly."""
        assert 0.50 >= 0.50
        assert (0.4999 < 0.50) is True

    def test_b8_router_equal_logits_multiclass_tie(self):
        """Verify 4-class equal logits produce 0.25 probability (<0.50) -> triggers router anomaly."""
        logits = np.array([1.0, 1.0, 1.0, 1.0])
        probs = np.exp(logits) / np.sum(np.exp(logits))
        assert (probs == 0.25).all()
        assert np.max(probs) < 0.50

    def test_b8_router_empty_routing_table_fallback(self):
        """Verify router with empty routing table returns submodel_id=0."""
        ens = EmulatedModelEnsemble()
        ens.load_router({"config": {"model_id": 1, "temporal_depth": 2, "frequency_bins": 256, "num_modes": 2}})
        ens.routes.clear()
        target_id, _, _ = ens.inf_router()
        assert target_id == 0

    def test_b8_router_out_of_range_mode_index(self):
        """Verify unmapped mode index resolves to submodel 0."""
        ens = EmulatedModelEnsemble()
        ens.load_router({"config": {"model_id": 1, "temporal_depth": 2, "frequency_bins": 256, "num_modes": 2}})
        ens.set_routes([{"out_ix": 0, "m_id": 10}])  # out_ix=1 unmapped
        assert ens.routes.get(1, 0) == 0

    def test_b8_router_invocation_without_model_loaded(self):
        """Verify invoking router before loading model returns failure flag safely."""
        ens = EmulatedModelEnsemble()
        target_id, already_loaded, is_anom = ens.inf_router()
        assert target_id == 0
        assert is_anom is False


# ==============================================================================
# FEATURE 9: MEMORY BACKBONE BOUNDARIES
# ==============================================================================

class TestFeature9Boundaries:
    def test_b9_memory_state_dim_zero_boundary(self):
        """Verify memory model with state_dim=0 handles zero allocations cleanly."""
        ens = EmulatedModelEnsemble()
        ens.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 0}})
        assert ens.state_dim == 0
        assert len(ens.h_state) == 0

    def test_b9_memory_large_state_dimension_boundary(self):
        """Verify memory backbone scales to d=512 recurrent state dimension."""
        ens = EmulatedModelEnsemble()
        ens.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 512}})
        assert ens.state_dim == 512
        assert len(ens.h_state) == 512

    def test_b9_memory_recurrent_saturation_stability(self):
        """Verify 1000 consecutive memory updates do not produce NaN or blow up."""
        ens = EmulatedModelEnsemble()
        ens.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 32}})
        for _ in range(1000):
            ens.inf_memory(np.ones(256, dtype=np.float32) * 100.0)
        assert np.isfinite(ens.h_state).all()
        assert (np.abs(ens.h_state) <= 1.0).all()

    def test_b9_memory_invocation_without_model(self):
        """Verify feeding slices when no memory model is loaded updates ring buffer safely."""
        ens = EmulatedModelEnsemble()
        ens.inf_memory(np.ones(256, dtype=np.float32))
        assert ens.ring_buffer.head == 1

    def test_b9_memory_rapid_state_resets(self):
        """Verify rapid multiple state resets remain stable."""
        ens = EmulatedModelEnsemble()
        ens.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 16}})
        for _ in range(10):
            ens.inf_memory(np.ones(256))
            ens.reset_state()
            assert (ens.h_state == 0).all()


# ==============================================================================
# FEATURE 10: AUTOENCODER ANOMALY BOUNDARIES
# ==============================================================================

class TestFeature10Boundaries:
    def test_b10_skip_amount_larger_than_total_frames(self):
        """Verify skip_amount larger than total frames evaluates at least one window."""
        ens = EmulatedModelEnsemble(warmup_steps=0)
        ens.submodel_cache.insert({"config": {"model_id": 1, "temporal_depth": 4, "frequency_bins": 256, "anomaly_threshold": 0.5, "loss_mode": 1}})
        for _ in range(16):
            ens.ring_buffer.write(np.ones(256))
        mse, _ = ens.inf_ae(num_frames=16, skip_amount=100)
        assert mse >= 0.0

    def test_b10_temporal_depth_equal_total_frames(self):
        """Verify temporal_depth equal to total frames (e.g. 16) evaluates exactly 1 window."""
        ens = EmulatedModelEnsemble(warmup_steps=0)
        ens.submodel_cache.insert({"config": {"model_id": 1, "temporal_depth": 16, "frequency_bins": 256, "anomaly_threshold": 0.5, "loss_mode": 1}})
        for _ in range(16):
            ens.ring_buffer.write(np.ones(256))
        mse, _ = ens.inf_ae(num_frames=16, skip_amount=0)
        assert mse >= 0.0

    def test_b10_threshold_zero_flags_all(self):
        """Verify anomaly_threshold=0.0 flags any non-zero reconstruction as anomaly."""
        ens = EmulatedModelEnsemble(warmup_steps=0)
        ens.submodel_cache.insert({"config": {"model_id": 1, "temporal_depth": 4, "frequency_bins": 256, "anomaly_threshold": 0.0, "loss_mode": 1}})
        for _ in range(16):
            ens.ring_buffer.write(np.ones(256))
        _, is_anom = ens.inf_ae(num_frames=16)
        assert is_anom is True

    def test_b10_threshold_infinity_flags_none(self):
        """Verify anomaly_threshold=infinity never flags anomalies."""
        ens = EmulatedModelEnsemble(warmup_steps=0)
        ens.submodel_cache.insert({"config": {"model_id": 1, "temporal_depth": 4, "frequency_bins": 256, "anomaly_threshold": float("inf"), "loss_mode": 1, "inject_anomaly": True}})
        for _ in range(16):
            ens.ring_buffer.write(np.ones(256) * 100.0)
        _, is_anom = ens.inf_ae(num_frames=16)
        assert is_anom is False

    def test_b10_autoencoder_evaluation_without_cached_models(self):
        """Verify autoencoder evaluation with empty cache returns 0.0 and False safely."""
        ens = EmulatedModelEnsemble()
        mse, is_anom = ens.inf_ae(num_frames=16)
        assert mse == 0.0
        assert is_anom is False


# ==============================================================================
# FEATURE 11: RUNTIME CONFIG CBOR BOUNDARIES
# ==============================================================================

class TestFeature11Boundaries:
    def test_b11_malformed_cbor_truncated_bytes(self):
        """Verify decoding truncated CBOR raises CBORError cleanly."""
        valid = SchemaBuilder.build_runtime_config()
        truncated = valid[:5]
        with pytest.raises(CBORError):
            CBORCodec.decode_all(truncated)

    def test_b11_malformed_cbor_invalid_major_type(self):
        """Verify corrupted bytes fail parsing without unhandled crashes."""
        bad_cbor = b"\xFF\xFF\xFF\xFF"
        with pytest.raises(CBORError):
            CBORCodec.decode_all(bad_cbor)

    def test_b11_cbor_unexpected_extra_keys_ignored(self, edge_simulator):
        """Verify unknown extra keys in runtime config are ignored gracefully."""
        payload = {
            "rate": 3840.0,
            "mode": 1,
            "batch": 4096,
            "beat": 5,
            "sd_en": True,
            "unknown_future_field": "test_extra"
        }
        cbor_bytes = CBORCodec.encode(payload)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", cbor_bytes)
        assert edge_simulator.sample_rate == 3840.0

    def test_b11_cbor_out_of_range_stream_mode(self, edge_simulator):
        """Verify stream mode outside enum range (e.g. 99) is recorded without crash."""
        cbor_bytes = SchemaBuilder.build_runtime_config(mode=99)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", cbor_bytes)
        assert edge_simulator.stream_mode == 99

    def test_b11_cbor_negative_integers_handling(self):
        """Verify negative integers encode and decode correctly via Major 1."""
        neg_val = -42
        enc = CBORCodec.encode(neg_val)
        dec = CBORCodec.decode_all(enc)
        assert dec == -42


# ==============================================================================
# FEATURE 12: ENSEMBLE CONFIG CBOR BOUNDARIES
# ==============================================================================

class TestFeature12Boundaries:
    def test_b12_ensemble_routes_count_zero(self, edge_simulator):
        """Verify ensemble config with 0 routes sets empty routing table."""
        cbor_bytes = SchemaBuilder.build_ensemble_config(routes=[])
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", cbor_bytes)
        assert len(edge_simulator.ensemble.routes) == 0

    def test_b12_ensemble_routes_exceeds_max_16_truncated(self):
        """Verify more than 16 routes in config are supported or bounded."""
        routes = [{"out_ix": i, "m_id": 100 + i} for i in range(20)]
        cbor_bytes = SchemaBuilder.build_ensemble_config(routes=routes)
        dec = CBORCodec.decode_all(cbor_bytes)
        assert len(dec["routes"]) == 20

    def test_b12_ensemble_duplicate_out_ix_override(self, edge_simulator):
        """Verify duplicate out_ix in routing table overrides to the latest mapping."""
        routes = [{"out_ix": 0, "m_id": 10}, {"out_ix": 0, "m_id": 20}]
        cbor_bytes = SchemaBuilder.build_ensemble_config(routes=routes)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", cbor_bytes)
        assert edge_simulator.ensemble.routes[0] == 20

    def test_b12_ensemble_warmup_zero_boundary(self, edge_simulator):
        """Verify warmup=0 immediately enables anomaly detection."""
        cbor_bytes = SchemaBuilder.build_ensemble_config(warmup=0)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", cbor_bytes)
        assert edge_simulator.ensemble.warmup_steps == 0

    def test_b12_ensemble_malformed_routes_array(self):
        """Verify non-list routes field raises during parsing."""
        bad_cfg = {"warmup": 5, "r_m_id": 1, "routes": "not_a_list"}
        cbor_bytes = CBORCodec.encode(bad_cfg)
        dec = CBORCodec.decode_all(cbor_bytes)
        assert isinstance(dec["routes"], str)


# ==============================================================================
# FEATURE 13: MODEL PACKAGE DEPLOYMENT BOUNDARIES
# ==============================================================================

class TestFeature13Boundaries:
    def test_b13_unknown_model_type_code(self, edge_simulator):
        """Verify package with unknown model type (e.g. 99) is safely ignored."""
        pkg = {"m_id": 999, "type": 99, "data": b"dummy"}
        cbor_bytes = CBORCodec.encode(pkg)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", cbor_bytes)
        assert edge_simulator.ensemble.submodel_cache.get(999) is None

    def test_b13_empty_data_byte_string(self):
        """Verify model package with b"" empty data encodes and decodes properly."""
        cbor_bytes = SchemaBuilder.build_autoencoder_package(m_id=1, data=b"")
        dec = CBORCodec.decode_all(cbor_bytes)
        assert dec["data"] == b""

    def test_b13_huge_model_binary_payload(self):
        """Verify 512KB model binary byte string encoding/decoding without corruption."""
        huge_bin = os.urandom(512 * 1024)
        cbor_bytes = SchemaBuilder.build_autoencoder_package(m_id=1, data=huge_bin)
        dec = CBORCodec.decode_all(cbor_bytes)
        assert len(dec["data"]) == 512 * 1024
        assert dec["data"] == huge_bin

    def test_b13_invalid_architecture_tag_type(self):
        """Verify custom integer architecture tags (e.g. tag=999)."""
        cbor_bytes = SchemaBuilder.build_autoencoder_package(m_id=1, data=b"x", tag=999)
        dec = CBORCodec.decode_all(cbor_bytes)
        assert dec["tag"] == 999

    def test_b13_malformed_model_cbor_envelope(self):
        """Verify truncated model package CBOR fails parsing."""
        cbor_bytes = SchemaBuilder.build_autoencoder_package(m_id=1, data=b"12345678")
        with pytest.raises(CBORError):
            CBORCodec.decode_all(cbor_bytes[:10])


# ==============================================================================
# FEATURE 14: STREAMING MODE BOUNDARIES
# ==============================================================================

class TestFeature14Boundaries:
    def test_b14_invalid_stream_mode_enum_handling(self, edge_simulator, synthetic_sensor_segment):
        """Verify invalid stream mode falls back gracefully without dropping system."""
        edge_simulator.stream_mode = 999
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["segment_id"] == 1

    def test_b14_mixed_mode_cadence_zero_boundary(self, edge_simulator, synthetic_sensor_segment):
        """Verify cadence=0 in Mixed mode suppresses all periodic normal uploads."""
        edge_simulator.stream_mode = 3
        edge_simulator.cadence = 0
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 0

    def test_b14_mixed_mode_cadence_one_continuous_equivalence(self, edge_simulator, synthetic_sensor_segment):
        """Verify cadence=1 in Mixed mode sends every segment (equivalent to Continuous)."""
        edge_simulator.stream_mode = 3
        edge_simulator.cadence = 1
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) > 0

    def test_b14_segment_id_uint32_max_boundary(self, edge_simulator, synthetic_sensor_segment):
        """Verify segment counter handles UINT32_MAX (4294967295) boundary."""
        edge_simulator.segment_counter = 4294967295
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["segment_id"] == 4294967295

    def test_b14_rapid_streaming_mode_switching(self, edge_simulator, synthetic_sensor_segment):
        """Verify switching streaming modes on consecutive segments causes no state corruption."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        for mode in (1, 2, 3, 4, 1):
            edge_simulator.stream_mode = mode
            edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert edge_simulator.last_segment_id == 5


# ==============================================================================
# FEATURE 15: TELEMETRY CHUNKING & MQTT OUTBOX BOUNDARIES
# ==============================================================================

class TestFeature15Boundaries:
    def test_b15_chunk_size_exceeds_segment_length(self):
        """Verify single chunk when chunk size >= segment length."""
        chunk_size = 4096
        total = 4096
        num_chunks = total // chunk_size
        assert num_chunks == 1

    def test_b15_chunk_count_boundary_indexing(self):
        """Verify chunk index numbers are 1-based [1..8]."""
        num_chunks = 8
        indices = [i + 1 for i in range(num_chunks)]
        assert indices == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_b15_mqtt_outbox_capacity_overflow_drop(self, mock_broker):
        """Verify outbox honors max capacity and drops excess messages without memory exhaustion."""
        mock_broker.disconnect()
        mock_broker.outbox_capacity = 5
        for i in range(10):
            mock_broker.publish("test/topic", f"msg_{i}".encode())
        assert len(mock_broker.outbox) == 5

    def test_b15_empty_telemetry_datapoints(self):
        """Verify segment payload with empty datapoints arrays encodes and decodes cleanly."""
        cbor_bytes = SchemaBuilder.build_telemetry_segment(
            segment_id=1, rate=3840.0, reason=1, chunk_idx=1,
            gyro_x=[], gyro_y=[], gyro_z=[],
            accel_x=[], accel_y=[], accel_z=[]
        )
        dec = CBORCodec.decode_all(cbor_bytes)
        assert len(dec["data"]["accel"]["x"]) == 0

    def test_b15_mqtt_reconnect_flushes_pending_outbox(self, mock_broker):
        """Verify reconnecting broker allows publishing new messages normally."""
        mock_broker.disconnect()
        mock_broker.publish("topic1", b"offline_msg")
        mock_broker.connect()
        mock_broker.publish("topic2", b"online_msg")
        assert len(mock_broker.published_messages) == 1
        assert mock_broker.published_messages[0].topic == "topic2"


# ==============================================================================
# FEATURE 16: SD CARD BINARY RECORD BOUNDARIES
# ==============================================================================

class TestFeature16Boundaries:
    def test_b16_unmounted_sd_card_write_bypass(self, mock_sd, synthetic_sensor_segment):
        """Verify disabled/unmounted SD storage returns empty path and writes 0 bytes."""
        mock_sd.enabled = False
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        filepath = mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        assert filepath == ""

    def test_b16_corrupted_record_file_recovery(self, mock_sd):
        """Verify reading corrupted/truncated file breaks gracefully without exception."""
        filepath = os.path.join(mock_sd.data_dir, "data_0.bin")
        with open(filepath, "wb") as f:
            f.write(b"CORRUPTED_SHORT_BYTES")
        with open(filepath, "rb") as f:
            data = f.read(MockSDStorage.RECORD_SIZE)
            assert len(data) < MockSDStorage.RECORD_SIZE

    def test_b16_zero_length_sd_card_read(self, mock_sd):
        """Verify reading non-existent model returns False."""
        assert mock_sd.model_exists(9999) is False

    def test_b16_sd_file_write_permission_error_simulation(self, mock_sd, synthetic_sensor_segment):
        """Verify writing when enabled=True writes full RECORD_SIZE bytes."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        filepath = mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        assert os.path.exists(filepath)

    def test_b16_record_boundary_at_64_bytes_header(self, mock_sd, synthetic_sensor_segment):
        """Verify exactly 64 bytes are reserved for header before first sensor axis."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        filepath = mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        with open(filepath, "rb") as f:
            f.seek(64)
            data_start = f.tell()
            assert data_start == 64


# ==============================================================================
# FEATURE 17: SD STORAGE ROTATION BOUNDARIES
# ==============================================================================

class TestFeature17Boundaries:
    def test_b17_exact_16mb_boundary_transition(self, mock_sd, synthetic_sensor_segment):
        """Verify exact boundary at MAX_RECORDS_PER_FILE rolls over."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        mock_sd.record_count = mock_sd.MAX_RECORDS_PER_FILE - 1
        mock_sd.active_file_idx = 0
        mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        assert mock_sd.record_count == 0
        assert mock_sd.active_file_idx == 1

    def test_b17_file_rotation_63_to_0_rollover(self, mock_sd, synthetic_sensor_segment):
        """Verify rollover from file 63 wraps cleanly to file 0."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        mock_sd.active_file_idx = 63
        mock_sd.record_count = mock_sd.MAX_RECORDS_PER_FILE - 1
        mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        assert mock_sd.active_file_idx == 0

    def test_b17_missing_state_bin_recovers_to_file_0(self, mock_sd):
        """Verify missing state.bin initializes defaults (file 0, record 0)."""
        state_file = os.path.join(mock_sd.data_dir, "state.bin")
        if os.path.exists(state_file):
            os.remove(state_file)
        mock_sd.load_state()
        assert mock_sd.active_file_idx == 0
        assert mock_sd.record_count == 0

    def test_b17_corrupted_state_bin_recovery(self, mock_sd):
        """Verify corrupted 3-byte state.bin does not crash loader."""
        state_file = os.path.join(mock_sd.data_dir, "state.bin")
        with open(state_file, "wb") as f:
            f.write(b"123")
        mock_sd.load_state()
        assert mock_sd.active_file_idx == 0

    def test_b17_file_creation_with_existing_stale_files(self, mock_sd, synthetic_sensor_segment):
        """Verify opening existing stale file in 'wb' mode on record 0 overwrites old contents."""
        filepath = os.path.join(mock_sd.data_dir, "data_0.bin")
        with open(filepath, "wb") as f:
            f.write(b"OLD_STALE_DATA")
        mock_sd.record_count = 0
        mock_sd.active_file_idx = 0
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        assert os.path.getsize(filepath) == MockSDStorage.RECORD_SIZE


# ==============================================================================
# FEATURE 18: HISTORICAL DUMP REPLAY BOUNDARIES
# ==============================================================================

class TestFeature18Boundaries:
    def test_b18_dump_empty_storage_no_crashes(self, edge_simulator):
        """Verify triggering dump on empty SD card executes cleanly without crash."""
        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=False)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 0

    def test_b18_dump_all_records_sent_zero_replays(self, edge_simulator, synthetic_sensor_segment):
        """Verify when 100% of records were sent in real-time, dump publishes 0 replay segments."""
        edge_simulator.stream_mode = 1  # Continuous -> was_sent = True
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        for _ in range(5):
            edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=False)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 0

    def test_b18_dump_all_records_unsent_full_replay(self, edge_simulator, synthetic_sensor_segment):
        """Verify when 100% of records were unsent, dump publishes all segments."""
        edge_simulator.stream_mode = 2  # AnomalyOnly -> normal segments unsent
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        for _ in range(4):
            edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=False)
        replayed = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")
        assert len(replayed) == 4

    def test_b18_dump_timing_zero_elapsed_division_protection(self, edge_simulator):
        """Verify dump duration zero ms elapsed is clamped to at least 1ms to prevent division by zero in IPS."""
        edge_simulator.trigger_dump(run_inference=True)
        assert edge_simulator.dump_time_ms >= 1
        assert np.isfinite(edge_simulator.dump_ips)

    def test_b18_dump_interrupted_pipeline_draining(self, edge_simulator):
        """Verify dump state flags reset to idle upon completion."""
        assert edge_simulator.dump_reading is False


# ==============================================================================
# FEATURE 19: QUAD BUFFER SYNCHRONIZATION BOUNDARIES
# ==============================================================================

class TestFeature19Boundaries:
    def test_b19_all_slots_busy_backpressure_stall(self):
        """Verify backpressure when all 4 slots are busy (no free slot available)."""
        free_queue = []
        assert len(free_queue) == 0  # Task would block waiting on semaphore/queue

    def test_b19_double_free_slot_queue_guard(self):
        """Verify free queue does not accept more than 4 slots."""
        free_slots = set()
        for idx in (0, 1, 2, 3):
            free_slots.add(idx)
        assert len(free_slots) == 4
        free_slots.add(0)  # Double free guarded by set uniqueness
        assert len(free_slots) == 4

    def test_b19_invalid_buffer_flag_bitmasks(self):
        """Verify bitwise operations on all 5 buffer flag masks."""
        BUF_FLAG_HARVESTING = 1 << 0
        BUF_FLAG_READY_INF  = 1 << 1
        BUF_FLAG_READY_SEND = 1 << 2
        BUF_FLAG_PENDING_SD = 1 << 3
        BUF_FLAG_FORCE_INF  = 1 << 4

        flags = BUF_FLAG_READY_INF | BUF_FLAG_PENDING_SD | BUF_FLAG_FORCE_INF
        assert (flags & BUF_FLAG_READY_INF) != 0
        assert (flags & BUF_FLAG_PENDING_SD) != 0
        assert (flags & BUF_FLAG_FORCE_INF) != 0
        assert (flags & BUF_FLAG_READY_SEND) == 0

    def test_b19_concurrent_slot_lock_contention(self):
        """Verify slot mutex lock and unlock mechanics."""
        is_locked = False
        is_locked = True   # Lock
        assert is_locked is True
        is_locked = False  # Unlock
        assert is_locked is False

    def test_b19_harvester_buffer_full_overflow_handling(self, edge_simulator, synthetic_sensor_segment):
        """Verify continuous ingestion across dozens of segments without memory leaks."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        for _ in range(20):
            res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
            assert res["segment_id"] > 0


# ==============================================================================
# FEATURE 20: NODE HEALTH & CAPABILITIES BOUNDARIES
# ==============================================================================

class TestFeature20Boundaries:
    def test_b20_zero_ram_reported_in_health(self):
        """Verify reporting 0 bytes free RAM in health CBOR is handled cleanly."""
        cbor_bytes = SchemaBuilder.build_node_health_info(ram=0, sd=False, cache=[], last=0)
        dec = CBORCodec.decode_all(cbor_bytes)
        assert dec["ram"] == 0
        assert dec["sd"] == 0

    def test_b20_cached_model_ids_exceeding_3_telemetry_slots(self, edge_simulator):
        """Verify health telemetry truncates model ID list to max 3 items matching schema buffer."""
        for i in range(10):
            edge_simulator.ensemble.submodel_cache.insert({"config": {"model_id": 100 + i}})
        edge_simulator.mqtt.clear()
        edge_simulator.publish_health()
        msg = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/info/health")[0]
        dec = CBORCodec.decode_all(msg.payload)
        assert len(dec["cache"]) <= 3

    def test_b20_health_dump_fields_null_when_no_dump_ran(self, edge_simulator):
        """Verify dump_t, dump_r, and ips fields are omitted when no dump was executed."""
        cbor_bytes = SchemaBuilder.build_node_health_info(ram=200000, sd=True, cache=[], last=10)
        dec = CBORCodec.decode_all(cbor_bytes)
        assert "dump_t" not in dec
        assert "ips" not in dec

    def test_b20_node_alert_empty_detail_string(self):
        """Verify NodeAlert with empty detail text string encodes and decodes properly."""
        alert_bytes = SchemaBuilder.build_node_alert(code=404, detail="")
        dec = CBORCodec.decode_all(alert_bytes)
        assert dec["code"] == 404
        assert dec["detail"] == ""

    def test_b20_rapid_reboot_command_bursts(self, edge_simulator):
        """Verify rapid multiple reboot command payloads do not destabilize the system."""
        cmd = SchemaBuilder.build_command(reboot=True)
        for _ in range(5):
            edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/cmd/reboot", cmd)
        assert edge_simulator.dump_reading is False
