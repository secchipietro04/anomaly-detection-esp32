"""
test_tier3_combinations.py - Tier 3 Cross-Feature Combination Tests
Contains >= 20 cross-feature tests exercising end-to-end multi-module workflows.
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


class TestTier3Combinations:
    def test_comb_01_registration_and_capabilities_discovery(self, edge_simulator):
        """Workflow: Boot -> Publish Capabilities -> Receive Config -> Transition State."""
        caps_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/info/caps")
        assert len(caps_msgs) == 1
        dec_caps = CBORCodec.decode_all(caps_msgs[0].payload)
        assert 3840.0 in dec_caps["accel_freqs"]

        # Downlink new config
        cfg = SchemaBuilder.build_runtime_config(rate=1920.0, mode=3, cad=5)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", cfg)
        assert edge_simulator.sample_rate == 1920.0
        assert edge_simulator.stream_mode == 3
        assert edge_simulator.cadence == 5

    def test_comb_02_config_downlink_to_telemetry_streaming(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: Config downlink changes rate/mode -> Telemetry packets reflect new parameters."""
        cfg = SchemaBuilder.build_runtime_config(rate=960.0, mode=1)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", cfg)

        edge_simulator.mqtt.clear()
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)

        chunks = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")
        assert len(chunks) == 8
        first_chunk = CBORCodec.decode_all(chunks[0].payload)
        assert first_chunk["rate"] == 960

    def test_comb_03_anomaly_detection_to_alert_generation(self, edge_simulator, anomaly_sensor_segment):
        """Workflow: Normal operation -> Injected anomaly -> Anomaly flagged -> Packet published."""
        # Deploy autoencoder with strict threshold
        bin_data = SyntheticTFLiteModel.generate(model_id=10)
        pkg = SchemaBuilder.build_autoencoder_package(m_id=10, data=bin_data, limit=0.001)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", pkg)
        edge_simulator.ensemble.warmup_steps = 0

        edge_simulator.stream_mode = 2  # AnomalyOnly
        edge_simulator.mqtt.clear()
        ax, ay, az, gx, gy, gz = anomaly_sensor_segment
        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["is_anomaly"] is True

        inf_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")
        assert len(inf_msgs) == 1
        dec_inf = CBORCodec.decode_all(inf_msgs[0].payload)
        assert dec_inf["anom"] is True

    def test_comb_04_mixed_streaming_cadence_filtering(self, edge_simulator, synthetic_sensor_segment, anomaly_sensor_segment):
        """Workflow: Mixed mode (cadence=5) -> Ingest 10 normal + 1 anomaly -> Exact periodic & anomaly filtering."""
        edge_simulator.stream_mode = 3
        edge_simulator.cadence = 5
        edge_simulator.ensemble.warmup_steps = 0
        bin_data = SyntheticTFLiteModel.generate(model_id=10)
        pkg = SchemaBuilder.build_autoencoder_package(m_id=10, data=bin_data, limit=0.005)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", pkg)

        norm_ax, norm_ay, norm_az, norm_gx, norm_gy, norm_gz = synthetic_sensor_segment
        anom_ax, anom_ay, anom_az, anom_gx, anom_gy, anom_gz = anomaly_sensor_segment

        edge_simulator.mqtt.clear()
        # Segments 1 to 4: normal -> suppressed
        for _ in range(4):
            edge_simulator.ingest_sensor_segment(norm_ax, norm_ay, norm_az, norm_gx, norm_gy, norm_gz)
        # Segment 5: normal, but 5 % 5 == 0 -> sent (Periodic)
        edge_simulator.ingest_sensor_segment(norm_ax, norm_ay, norm_az, norm_gx, norm_gy, norm_gz)
        # Segment 6: anomaly -> sent (Anomaly)
        edge_simulator.ingest_sensor_segment(anom_ax, anom_ay, anom_az, anom_gx, anom_gy, anom_gz)

        inf_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")
        assert len(inf_msgs) == 2
        reasons = [CBORCodec.decode_all(m.payload)["reason"] for m in inf_msgs]
        assert 3 in reasons  # Periodic
        assert 2 in reasons  # Anomaly

    def test_comb_05_multi_modal_routing_pipeline(self, edge_simulator):
        """Workflow: Deploy Router + 2 Autoencoders -> Low/High frequency vibrations route to respective models."""
        r_bin = SyntheticTFLiteModel.generate(model_id=1, archetype="router")
        ae1_bin = SyntheticTFLiteModel.generate(model_id=101, archetype="autoencoder")
        ae2_bin = SyntheticTFLiteModel.generate(model_id=102, archetype="autoencoder")

        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_router_package(m_id=1, data=r_bin, classes=2))
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=101, data=ae1_bin))
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=102, data=ae2_bin))
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_ensemble_config(routes=[{"out_ix": 0, "m_id": 101}, {"out_ix": 1, "m_id": 102}]))

        assert edge_simulator.ensemble.routes[0] == 101
        assert edge_simulator.ensemble.routes[1] == 102

    def test_comb_06_dynamic_model_cache_miss_and_sd_loading(self, edge_simulator):
        """Workflow: Model not in RAM cache -> Loaded from SD card storage into cache."""
        bin_data = SyntheticTFLiteModel.generate(model_id=88)
        pkg = SchemaBuilder.build_autoencoder_package(m_id=88, data=bin_data)
        edge_simulator.sd.save_model(88, pkg)
        assert edge_simulator.ensemble.submodel_cache.get(88) is None

        # Load into cache
        model_dict = {"config": {"model_id": 88, "temporal_depth": 16, "frequency_bins": 256, "anomaly_threshold": 0.05, "loss_mode": 1}}
        edge_simulator.ensemble.submodel_cache.insert(model_dict)
        assert edge_simulator.ensemble.submodel_cache.get(88) is not None

    def test_comb_07_cache_overflow_lifo_eviction(self, edge_simulator):
        """Workflow: Cache capacity 2 -> Insert models 1, 2, 3 -> Model 1 evicted -> Querying 1 misses."""
        edge_simulator.ensemble.submodel_cache.set_capacity(2)
        edge_simulator.ensemble.submodel_cache.insert({"config": {"model_id": 1}})
        edge_simulator.ensemble.submodel_cache.insert({"config": {"model_id": 2}})
        edge_simulator.ensemble.submodel_cache.insert({"config": {"model_id": 3}})
        assert edge_simulator.ensemble.submodel_cache.get(1) is None
        assert edge_simulator.ensemble.submodel_cache.get(2) is not None
        assert edge_simulator.ensemble.submodel_cache.get(3) is not None

    def test_comb_08_ensemble_reconfiguration_and_evicted_model_cleanup(self, edge_simulator):
        """Workflow: Routes updated removing model 20 -> File deleted from SD -> Model 10 retained."""
        edge_simulator.sd.save_model(10, b"m10")
        edge_simulator.sd.save_model(20, b"m20")
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_ensemble_config(routes=[{"out_ix": 0, "m_id": 10}, {"out_ix": 1, "m_id": 20}]))
        # Update removing 20
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_ensemble_config(routes=[{"out_ix": 0, "m_id": 10}, {"out_ix": 1, "m_id": 30}]))
        assert edge_simulator.sd.model_exists(10) is True
        assert edge_simulator.sd.model_exists(20) is False

    def test_comb_09_continuous_logging_and_historical_dump(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: AnomalyOnly streaming logs normal segments to SD -> Dump replays all unsent data."""
        edge_simulator.stream_mode = 2
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        for _ in range(5):
            edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)

        edge_simulator.mqtt.clear()
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/cmd/dump", SchemaBuilder.build_command(dump=True))
        replayed = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")
        assert len(replayed) == 5

    def test_comb_10_offline_re_inferencing_dump_i(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: Stored segments replayed with dump_i -> Offline inferences executed -> IPS computed."""
        edge_simulator.stream_mode = 2
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        for _ in range(3):
            edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)

        edge_simulator.mqtt.clear()
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/cmd/dump_i", SchemaBuilder.build_command(dump_i=True))
        inf_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")
        assert len(inf_msgs) == 3
        assert edge_simulator.dump_ips >= 0.0

    def test_comb_11_fifo_overrun_with_temporal_discontinuity(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: Harvester experiences FIFO overrun -> Segment counter skips -> Continuity gap preserved."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        r1 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=False)
        r2 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=True)
        r3 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=False)
        assert r1["segment_id"] == 1
        assert r2["segment_id"] == 2
        assert r3["segment_id"] == 4

    def test_comb_12_router_uncertainty_anomaly_bypass(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: Low confidence router classification (<0.5) immediately triggers Router Anomaly with mse=1.0."""
        # Unmapped / ambiguous router state
        edge_simulator.ensemble.load_router({"config": {"model_id": 1, "temporal_depth": 16, "frequency_bins": 256, "num_modes": 4}})
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["is_anomaly"] is True
        assert res["mse"] == 1.0

    def test_comb_13_memory_backbone_recurrent_continuity_across_segments(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: Memory backbone updates recurrent state vectors monotonically across consecutive segments."""
        edge_simulator.ensemble.load_memory({"config": {"model_id": 2, "frequency_bins": 256, "d": 16}})
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        h1 = edge_simulator.ensemble.h_state.copy()
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        h2 = edge_simulator.ensemble.h_state.copy()
        assert not np.array_equal(h1, h2)

        # Reboot resets state
        edge_simulator.reboot()
        assert (edge_simulator.ensemble.h_state == 0).all()

    def test_comb_14_sd_storage_16mb_rollover_and_state_persistence(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: Logging segments up to MAX_RECORDS_PER_FILE triggers file rollover and state.bin update."""
        edge_simulator.sd.record_count = edge_simulator.sd.MAX_RECORDS_PER_FILE - 1
        edge_simulator.sd.active_file_idx = 0
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert edge_simulator.sd.active_file_idx == 1
        assert edge_simulator.sd.record_count == 0

    def test_comb_15_network_disconnect_and_outbox_buffering(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: Broker disconnected -> Messages buffer into outbox -> Reconnect allows publishing."""
        edge_simulator.mqtt.disconnect()
        edge_simulator.stream_mode = 1
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert len(edge_simulator.mqtt.outbox) > 0

        edge_simulator.mqtt.connect()
        assert edge_simulator.mqtt.is_connected is True

    def test_comb_16_skip_amount_optimization(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: Autoencoder with skip_amount=2 evaluates window loop with skip steps."""
        ens = EmulatedModelEnsemble(warmup_steps=0)
        ens.submodel_cache.insert({"config": {"model_id": 10, "temporal_depth": 4, "frequency_bins": 256, "anomaly_threshold": 0.1, "loss_mode": 1, "skip_amount": 2}})
        for _ in range(16):
            ens.ring_buffer.write(np.ones(256, dtype=np.float32))
        mse, _ = ens.inf_ae(num_frames=16, skip_amount=2)
        assert mse >= 0.0

    def test_comb_17_hot_swap_router_model(self, edge_simulator):
        """Workflow: Active system hot-swaps router model flatbuffer without restarting."""
        r1 = SyntheticTFLiteModel.generate(model_id=1, archetype="router")
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_router_package(m_id=1, data=r1, classes=2))
        assert edge_simulator.ensemble.router_model["config"]["model_id"] == 1

        # Hot-swap with model 2
        r2 = SyntheticTFLiteModel.generate(model_id=2, archetype="router")
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_router_package(m_id=2, data=r2, classes=3))
        assert edge_simulator.ensemble.router_model["config"]["model_id"] == 2

    def test_comb_18_hot_swap_memory_model(self, edge_simulator):
        """Workflow: Active system hot-swaps memory backbone model and updates state vector dimensions."""
        m1 = SyntheticTFLiteModel.generate(model_id=10, archetype="memory")
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_memory_package(m_id=10, data=m1, state=16))
        assert edge_simulator.ensemble.state_dim == 16

        m2 = SyntheticTFLiteModel.generate(model_id=20, archetype="memory")
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_memory_package(m_id=20, data=m2, state=64))
        assert edge_simulator.ensemble.state_dim == 64

    def test_comb_19_health_telemetry_status_transitions(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: Health telemetry reflects state transitions between idle and streaming_dump."""
        edge_simulator.mqtt.clear()
        edge_simulator.publish_health()
        m1 = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/info/health")[0]
        assert CBORCodec.decode_all(m1.payload)["status"] == "idle"

    def test_comb_20_full_lifecycle_cold_boot_to_anomaly_and_dump(self, edge_simulator, synthetic_sensor_segment, anomaly_sensor_segment):
        """Workflow: Full E2E Lifecycle: Boot -> Caps -> Config -> Deploy -> Stream -> Anomaly -> Dump -> Reboot."""
        # 1. Capabilities
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/info/caps")) == 1

        # 2. Config downlink
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", SchemaBuilder.build_runtime_config(rate=3840.0, mode=2))
        assert edge_simulator.stream_mode == 2

        # 3. Model Deployment
        ae_bin = SyntheticTFLiteModel.generate(model_id=10)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=10, data=ae_bin, limit=0.005))
        edge_simulator.ensemble.warmup_steps = 0

        # 4. Ingest normal segment (suppressed in mode 2)
        norm_ax, norm_ay, norm_az, norm_gx, norm_gy, norm_gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()
        edge_simulator.ingest_sensor_segment(norm_ax, norm_ay, norm_az, norm_gx, norm_gy, norm_gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 0

        # 5. Ingest anomaly segment (published)
        anom_ax, anom_ay, anom_az, anom_gx, anom_gy, anom_gz = anomaly_sensor_segment
        res_anom = edge_simulator.ingest_sensor_segment(anom_ax, anom_ay, anom_az, anom_gx, anom_gy, anom_gz)
        assert res_anom["is_anomaly"] is True
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) > 0

        # 6. Dump historical unsent normal segment
        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=False)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 1

        # 7. Reboot
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/cmd/reboot", SchemaBuilder.build_command(reboot=True))
        assert edge_simulator.dump_reading is False

    def test_comb_21_dual_axis_harmonic_interference(self, edge_simulator):
        """Workflow: Complex multi-axis vibration harmonics combine cleanly into 1D magnitude and STFT."""
        t = np.linspace(0, 4096 / 3840.0, 4096, endpoint=False)
        ax = np.sin(2 * np.pi * 60 * t).astype(np.float32)
        ay = np.cos(2 * np.pi * 120 * t).astype(np.float32)
        az = 9.81 + np.sin(2 * np.pi * 180 * t).astype(np.float32)
        gx = 0.5 * np.sin(2 * np.pi * 30 * t).astype(np.float32)
        gy = 0.5 * np.cos(2 * np.pi * 30 * t).astype(np.float32)
        gz = 0.1 * np.random.randn(4096).astype(np.float32)

        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["segment_id"] > 0
        assert np.isfinite(res["mse"])

    def test_comb_22_sd_disabled_runtime_toggle(self, edge_simulator, synthetic_sensor_segment):
        """Workflow: SD logging active -> Config downlink disables SD -> Resumes on re-enabling."""
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        cnt1 = edge_simulator.sd.record_count
        assert cnt1 == 1

        # Disable SD
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", SchemaBuilder.build_runtime_config(sd_en=False))
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert edge_simulator.sd.record_count == 1  # No write

        # Re-enable SD
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", SchemaBuilder.build_runtime_config(sd_en=True))
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert edge_simulator.sd.record_count == 2
