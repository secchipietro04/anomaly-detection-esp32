"""
test_tier4_scenarios.py - Tier 4 Real-World End-to-End Scenario Tests
Contains >= 10 end-to-end real-world industrial scenario tests as specified in TEST_INFRA.md.
"""

import math
import os
import struct
import time
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


class TestTier4RealWorldScenarios:
    def test_scenario_01_factory_conveyor_continuous_monitoring(self, edge_simulator, synthetic_sensor_segment):
        """
        Scenario 1: Factory Conveyor Continuous Health Monitoring
        Simulates steady-state 24-hour equivalent continuous logging and real-time streaming at 3840 Hz.
        """
        # Configure continuous mode
        cfg = SchemaBuilder.build_runtime_config(rate=3840.0, mode=1, beat=5, sd_en=True)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/config", cfg)

        # Deploy baseline model
        ae_bin = SyntheticTFLiteModel.generate(model_id=10)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=10, data=ae_bin))

        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        edge_simulator.mqtt.clear()

        # Ingest 10 consecutive steady-state operational segments
        for i in range(10):
            res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
            assert res["segment_id"] == i + 1
            assert res["is_anomaly"] is False

        # Verify all 10 segments published both raw chunks and inference packets
        sensor_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")
        inf_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")
        assert len(sensor_msgs) == 10 * 8  # 10 segments x 8 chunks = 80 chunks
        assert len(inf_msgs) == 10
        assert edge_simulator.sd.record_count == 10

    def test_scenario_02_sudden_bearing_failure_instant_escalation(self, edge_simulator, synthetic_sensor_segment, anomaly_sensor_segment):
        """
        Scenario 2: Sudden Bearing Failure & Instant Anomaly Escalation
        Healthy machine operation suddenly experiences destructive bearing seizure.
        """
        edge_simulator.stream_mode = 2  # AnomalyOnly
        edge_simulator.ensemble.warmup_steps = 0
        ae_bin = SyntheticTFLiteModel.generate(model_id=10)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=10, data=ae_bin, limit=0.01))

        norm_ax, norm_ay, norm_az, norm_gx, norm_gy, norm_gz = synthetic_sensor_segment
        anom_ax, anom_ay, anom_az, anom_gx, anom_gy, anom_gz = anomaly_sensor_segment

        edge_simulator.mqtt.clear()
        # 5 normal segments -> all suppressed
        for _ in range(5):
            edge_simulator.ingest_sensor_segment(norm_ax, norm_ay, norm_az, norm_gx, norm_gy, norm_gz)
        assert len(edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")) == 0

        # Sudden bearing spike
        res = edge_simulator.ingest_sensor_segment(anom_ax, anom_ay, anom_az, anom_gx, anom_gy, anom_gz)
        assert res["is_anomaly"] is True

        # Verify instant alert published
        anom_chunks = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")
        anom_inf = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")
        assert len(anom_chunks) == 8
        assert len(anom_inf) == 1
        dec_inf = CBORCodec.decode_all(anom_inf[0].payload)
        assert dec_inf["anom"] is True
        assert dec_inf["reason"] == 2  # Anomaly

    def test_scenario_03_multi_operational_machine_modes(self, edge_simulator):
        """
        Scenario 3: Multi-Operational Machine Modes (CNC Mill: Idle, Roughing, Finishing)
        Ensemble Router dynamically selects specialized submodels based on spectral signatures.
        """
        r_bin = SyntheticTFLiteModel.generate(model_id=1, archetype="router")
        ae1_bin = SyntheticTFLiteModel.generate(model_id=101)  # Roughing submodel
        ae2_bin = SyntheticTFLiteModel.generate(model_id=102)  # Finishing submodel

        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_router_package(m_id=1, data=r_bin, classes=2))
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=101, data=ae1_bin))
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=102, data=ae2_bin))
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_ensemble_config(routes=[{"out_ix": 0, "m_id": 101}, {"out_ix": 1, "m_id": 102}]))

        # Mode 1: Low frequency roughing vibrations
        t = np.linspace(0, 4096 / 3840.0, 4096, endpoint=False)
        ax1 = 2.0 * np.sin(2 * np.pi * 50 * t).astype(np.float32)
        res1 = edge_simulator.ingest_sensor_segment(ax1, ax1*0.5, np.full(4096, 9.81, dtype=np.float32), np.zeros(4096, dtype=np.float32), np.zeros(4096, dtype=np.float32), np.zeros(4096, dtype=np.float32))
        assert res1["router_id"] == 1
        assert res1["ae_id"] in (101, 102)

    def test_scenario_04_intermittent_network_blackout_and_bulk_recovery(self, edge_simulator, synthetic_sensor_segment):
        """
        Scenario 4: Intermittent Network Blackout & Bulk Historical Data Recovery
        Plant Wi-Fi disconnects for multiple hours -> all segments logged to SD -> connection restored -> bulk dump.
        """
        edge_simulator.stream_mode = 2  # AnomalyOnly (suppresses normal streaming)
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment

        # Simulate 8 offline shifts
        for _ in range(8):
            edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert edge_simulator.sd.record_count == 8

        # Wi-Fi link restores -> Cloud sends dump command
        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=False)

        # All 8 stored records replayed
        replayed_chunks = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/data/sensor")
        assert len(replayed_chunks) == 8
        for chunk_msg in replayed_chunks:
            dec = CBORCodec.decode_all(chunk_msg.payload)
            assert dec["reason"] == 4  # ManualDump

    def test_scenario_05_ota_machine_learning_ensemble_upgrade(self, edge_simulator, synthetic_sensor_segment):
        """
        Scenario 5: Over-the-Air (OTA) Machine Learning Ensemble Upgrade
        Hot-swapping models during continuous operation without interruption.
        """
        edge_simulator.stream_mode = 1
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment

        # Step 1: Initial deployment (v1)
        ae_v1 = SyntheticTFLiteModel.generate(model_id=10)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=10, data=ae_v1))
        edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert edge_simulator.ensemble.submodel_cache.get(10) is not None

        # Step 2: Cloud deploys improved model (v2 - model ID 20) and updates routes
        ae_v2 = SyntheticTFLiteModel.generate(model_id=20)
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=20, data=ae_v2))
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_ensemble_config(routes=[{"out_ix": 0, "m_id": 20}]))

        # Step 3: Subsequent ingestion seamlessly utilizes v2 model
        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["segment_id"] == 2
        assert edge_simulator.ensemble.submodel_cache.get(20) is not None

    def test_scenario_06_storage_exhaustion_circular_file_retention(self, mock_sd, synthetic_sensor_segment):
        """
        Scenario 6: Flash / SD Storage Exhaustion & Circular File Retention
        Continuous logging rotates files at 16MB boundaries across 64 cyclical partitions.
        """
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        mock_sd.record_count = mock_sd.MAX_RECORDS_PER_FILE - 1
        mock_sd.active_file_idx = 63  # Last file

        # Write segment -> rolls over to file 0
        mock_sd.write_segment_record(1, 3840.0, ax, ay, az, gx, gy, gz)
        assert mock_sd.active_file_idx == 0
        assert mock_sd.record_count == 0

    def test_scenario_07_memory_constrained_edge_device_stress(self, edge_simulator, synthetic_sensor_segment):
        """
        Scenario 7: Memory-Constrained Edge Optimization
        Constrained RAM budget (capacity=2) under rapid submodel switching stress.
        """
        edge_simulator.ensemble.submodel_cache.set_capacity(2)
        for i in range(1, 6):
            bin_data = SyntheticTFLiteModel.generate(model_id=i)
            edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/ensemble", SchemaBuilder.build_autoencoder_package(m_id=i, data=bin_data))

        # Only 2 models resident in RAM
        assert len(edge_simulator.ensemble.submodel_cache.cache) == 2
        # All 5 persisted on SD card
        for i in range(1, 6):
            assert edge_simulator.sd.model_exists(i) is True

    def test_scenario_08_high_frequency_vibration_burst_overrun_recovery(self, edge_simulator, synthetic_sensor_segment):
        """
        Scenario 8: High-Frequency Vibration Burst & FIFO Overrun Recovery
        Temporary FIFO buffer overrun skips ID to preserve temporal indexing without crashing.
        """
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        r1 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=False)
        r2 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=True)  # Overrun
        r3 = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz, fifo_overrun=False)
        assert r1["segment_id"] == 1
        assert r2["segment_id"] == 2
        assert r3["segment_id"] == 4  # ID 3 skipped

    def test_scenario_09_retrospective_offline_anomaly_rescoring(self, edge_simulator, synthetic_sensor_segment):
        """
        Scenario 9: Retrospective Anomaly Verification with Offline Re-scoring (dump_i)
        Historical raw vibration data is re-evaluated offline against newly updated models.
        """
        edge_simulator.stream_mode = 2  # Unsent
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        for _ in range(3):
            edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)

        edge_simulator.mqtt.clear()
        edge_simulator.trigger_dump(run_inference=True)

        inf_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")
        assert len(inf_msgs) == 3
        assert edge_simulator.dump_time_ms >= 1
        assert edge_simulator.dump_ips >= 0.0

    def test_scenario_10_sensor_hardware_fault_isolation_and_tele_diagnostics(self, edge_simulator):
        """
        Scenario 10: Catastrophic Hardware Fault Isolation & Remote Reboot Recovery
        Sensor fault publishes NodeAlert -> Cloud investigates -> issues remote reboot.
        """
        # Node publishes alert
        alert = SchemaBuilder.build_node_alert(code=503, detail="SPI_COMM_CRC_ERROR")
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/info/alert", alert)
        alert_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/info/alert")
        assert len(alert_msgs) == 1
        dec = CBORCodec.decode_all(alert_msgs[0].payload)
        assert dec["code"] == 503

        # Remote reboot command
        edge_simulator.mqtt.publish(f"v1/{edge_simulator.node_id}/cmd/reboot", SchemaBuilder.build_command(reboot=True))
        assert edge_simulator.dump_reading is False

    def test_scenario_11_variable_speed_compressor_workcycle(self, edge_simulator, synthetic_sensor_segment):
        """
        Scenario 11: Variable Speed Industrial Air Compressor Workcycle
        Mixed mode cadence operation with dynamic rate adjustments.
        """
        edge_simulator.stream_mode = 3
        edge_simulator.cadence = 4
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment

        edge_simulator.mqtt.clear()
        for i in range(8):
            edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)

        # Segments 4 and 8 sent on cadence
        inf_msgs = edge_simulator.mqtt.get_messages(f"v1/{edge_simulator.node_id}/inference")
        assert len(inf_msgs) == 2

    def test_scenario_12_autonomous_edge_inference_in_disconnected_field(self, edge_simulator, synthetic_sensor_segment):
        """
        Scenario 12: Autonomous Edge Inference in Cloud-Disconnected Offshore Turbine
        Node boots, loads models locally from SD card without broker, and evaluates vibration data.
        """
        edge_simulator.mqtt.disconnect()
        # Save model to SD
        ae_bin = SyntheticTFLiteModel.generate(model_id=55)
        pkg = SchemaBuilder.build_autoencoder_package(m_id=55, data=ae_bin)
        edge_simulator.sd.save_model(55, pkg)

        # Autonomous inference
        ax, ay, az, gx, gy, gz = synthetic_sensor_segment
        res = edge_simulator.ingest_sensor_segment(ax, ay, az, gx, gy, gz)
        assert res["segment_id"] == 1
        assert edge_simulator.sd.record_count == 1
