"""
conftest.py - Comprehensive E2E Test Fixtures, Synthetic CBOR Codecs,
Mock MQTT Brokers, TFLite Model Generators, and Emulated Edge Subsystems.
"""

import math
import os
import struct
import tempfile
import time
import pytest
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field


# ==============================================================================
# 1. PURE-PYTHON CBOR ENCODER & DECODER (RFC 8949) WITH CDDL SCHEMA CONFORMANCE
# ==============================================================================

class CBORError(Exception):
    """Base exception for CBOR encode/decode errors."""
    pass


class CBORCodec:
    """
    RFC 8949 compliant CBOR encoder and decoder supporting all major types:
    - Major 0: Unsigned integer
    - Major 1: Negative integer
    - Major 2: Byte string
    - Major 3: Text string (UTF-8)
    - Major 4: Array of data items
    - Major 5: Map of pairs of data items
    - Major 6: Semantic tags
    - Major 7: Floating-point numbers and simple values (bool, null)
    """

    @classmethod
    def encode(cls, obj: Any) -> bytes:
        """Encode a Python object into CBOR bytes."""
        if obj is None:
            return bytes([0xF6])  # null
        elif isinstance(obj, bool):
            return bytes([0xF5 if obj else 0xF4])
        elif isinstance(obj, int):
            if obj >= 0:
                return cls._encode_type_and_val(0, obj)
            else:
                return cls._encode_type_and_val(1, -1 - obj)
        elif isinstance(obj, float):
            # Encode as IEEE 754 double (64-bit) float
            return bytes([0xFB]) + struct.pack(">d", obj)
        elif isinstance(obj, bytes):
            return cls._encode_type_and_val(2, len(obj)) + obj
        elif isinstance(obj, str):
            utf8 = obj.encode("utf-8")
            return cls._encode_type_and_val(3, len(utf8)) + utf8
        elif isinstance(obj, (list, tuple)):
            header = cls._encode_type_and_val(4, len(obj))
            return header + b"".join(cls.encode(item) for item in obj)
        elif isinstance(obj, dict):
            header = cls._encode_type_and_val(5, len(obj))
            body = bytearray()
            for k, v in obj.items():
                body.extend(cls.encode(k))
                body.extend(cls.encode(v))
            return header + bytes(body)
        else:
            raise CBORError(f"Unsupported type for CBOR encoding: {type(obj)}")

    @classmethod
    def _encode_type_and_val(cls, major_type: int, val: int) -> bytes:
        major = major_type << 5
        if val < 24:
            return bytes([major | val])
        elif val <= 0xFF:
            return bytes([major | 24, val])
        elif val <= 0xFFFF:
            return bytes([major | 25]) + struct.pack(">H", val)
        elif val <= 0xFFFFFFFF:
            return bytes([major | 26]) + struct.pack(">I", val)
        else:
            return bytes([major | 27]) + struct.pack(">Q", val)

    @classmethod
    def decode(cls, data: bytes) -> Tuple[Any, int]:
        """
        Decode CBOR bytes to a Python object.
        Returns (decoded_object, bytes_consumed).
        """
        if not data:
            raise CBORError("Empty CBOR byte stream")
        return cls._decode_item(data, 0)

    @classmethod
    def decode_all(cls, data: bytes) -> Any:
        val, consumed = cls.decode(data)
        return val

    @classmethod
    def _decode_item(cls, data: bytes, offset: int) -> Tuple[Any, int]:
        if offset >= len(data):
            raise CBORError(f"Unexpected end of CBOR stream at offset {offset}")

        initial_byte = data[offset]
        major_type = initial_byte >> 5
        additional_info = initial_byte & 0x1F
        curr = offset + 1

        val, curr = cls._read_val(data, additional_info, curr)

        if major_type == 0:  # Unsigned integer
            return val, curr
        elif major_type == 1:  # Negative integer
            return -1 - val, curr
        elif major_type == 2:  # Byte string
            if curr + val > len(data):
                raise CBORError("Byte string exceeds data length")
            b = data[curr:curr + val]
            return b, curr + val
        elif major_type == 3:  # Text string
            if curr + val > len(data):
                raise CBORError("Text string exceeds data length")
            s = data[curr:curr + val].decode("utf-8", errors="replace")
            return s, curr + val
        elif major_type == 4:  # Array
            items = []
            for _ in range(val):
                item, curr = cls._decode_item(data, curr)
                items.append(item)
            return items, curr
        elif major_type == 5:  # Map
            mapping = {}
            for _ in range(val):
                k, curr = cls._decode_item(data, curr)
                v, curr = cls._decode_item(data, curr)
                mapping[k] = v
            return mapping, curr
        elif major_type == 6:  # Tag
            tag_val = val
            tagged_item, curr = cls._decode_item(data, curr)
            return {"_cbor_tag": tag_val, "value": tagged_item}, curr
        elif major_type == 7:  # Float / Simple
            if additional_info == 20:
                return False, curr
            elif additional_info == 21:
                return True, curr
            elif additional_info == 22:
                return None, curr
            elif additional_info == 25:  # IEEE 754 Half-Precision Float (16-bit)
                half_bits = struct.unpack(">H", data[offset + 1:offset + 3])[0]
                val = float(np.frombuffer(struct.pack(">H", half_bits), dtype=np.float16)[0])
                return val, offset + 3
            elif additional_info == 26:  # Single Precision Float (32-bit)
                val = struct.unpack(">f", data[offset + 1:offset + 5])[0]
                return val, offset + 5
            elif additional_info == 27:  # Double Precision Float (64-bit)
                val = struct.unpack(">d", data[offset + 1:offset + 9])[0]
                return val, offset + 9
            else:
                return val, curr
        else:
            raise CBORError(f"Unknown major type {major_type}")

    @classmethod
    def _read_val(cls, data: bytes, info: int, offset: int) -> Tuple[int, int]:
        if info < 24:
            return info, offset
        elif info == 24:
            if offset >= len(data):
                raise CBORError("Truncated 1-byte integer")
            return data[offset], offset + 1
        elif info == 25:
            if offset + 2 > len(data):
                raise CBORError("Truncated 2-byte integer")
            return struct.unpack(">H", data[offset:offset + 2])[0], offset + 2
        elif info == 26:
            if offset + 4 > len(data):
                raise CBORError("Truncated 4-byte integer")
            return struct.unpack(">I", data[offset:offset + 4])[0], offset + 4
        elif info == 27:
            if offset + 8 > len(data):
                raise CBORError("Truncated 8-byte integer")
            return struct.unpack(">Q", data[offset:offset + 8])[0], offset + 8
        else:
            raise CBORError(f"Unsupported additional info {info}")


# ==============================================================================
# 2. CDDL SCHEMA PAYLOAD BUILDERS & VALIDATORS
# ==============================================================================

class SchemaBuilder:
    """Helper methods to build CBOR-encoded messages compliant with CDDL schemas."""

    @staticmethod
    def build_runtime_config(
        rate: float = 3840.0,
        mode: int = 1,
        batch: int = 4096,
        beat: int = 5,
        sd_en: bool = True,
        cad: Optional[int] = None
    ) -> bytes:
        payload = {
            "rate": float(rate),
            "mode": int(mode),
            "batch": int(batch),
            "beat": int(beat),
            "sd_en": bool(sd_en),
        }
        if cad is not None:
            payload["cad"] = int(cad)
        return CBORCodec.encode(payload)

    @staticmethod
    def build_ensemble_config(
        warmup: int = 5,
        r_m_id: int = 1,
        routes: Optional[List[Dict[str, int]]] = None,
        mem_id: Optional[int] = None
    ) -> bytes:
        if routes is None:
            routes = [{"out_ix": 0, "m_id": 10}, {"out_ix": 1, "m_id": 20}]
        cfg = {
            "warmup": int(warmup),
            "r_m_id": int(r_m_id),
            "routes": [{"out_ix": int(r.get("out_ix", r.get("out_idx", 0))), "m_id": int(r["m_id"])} for r in routes]
        }
        if mem_id is not None:
            cfg["mem_id"] = int(mem_id)
        return CBORCodec.encode(cfg)

    @staticmethod
    def build_autoencoder_package(
        m_id: int,
        data: bytes,
        accel_bins: int = 128,
        gyro_bins: int = 128,
        tsteps: int = 16,
        limit: float = 0.05,
        loss: int = 1,
        tag: Optional[int] = 1,
        mem_d: Optional[int] = None,
        skip: Optional[int] = None
    ) -> bytes:
        pkg = {
            "m_id": int(m_id),
            "type": 1,
            "accel_bins": int(accel_bins),
            "gyro_bins": int(gyro_bins),
            "tsteps": int(tsteps),
            "limit": float(limit),
            "loss": int(loss),
            "data": bytes(data)
        }
        if tag is not None:
            pkg["tag"] = int(tag)
        if mem_d is not None:
            pkg["mem_d"] = int(mem_d)
        if skip is not None:
            pkg["skip"] = int(skip)
        return CBORCodec.encode(pkg)

    @staticmethod
    def build_router_package(
        m_id: int,
        data: bytes,
        accel_bins: int = 128,
        gyro_bins: int = 128,
        tsteps: int = 16,
        classes: int = 2,
        tag: Optional[int] = 10,
        mem_d: Optional[int] = None
    ) -> bytes:
        pkg = {
            "m_id": int(m_id),
            "type": 2,
            "accel_bins": int(accel_bins),
            "gyro_bins": int(gyro_bins),
            "tsteps": int(tsteps),
            "class": int(classes),
            "data": bytes(data)
        }
        if tag is not None:
            pkg["tag"] = int(tag)
        if mem_d is not None:
            pkg["mem_d"] = int(mem_d)
        return CBORCodec.encode(pkg)

    @staticmethod
    def build_memory_package(
        m_id: int,
        data: bytes,
        accel_bins: int = 128,
        gyro_bins: int = 128,
        outdim: int = 32,
        state: int = 32,
        tag: Optional[int] = 20
    ) -> bytes:
        pkg = {
            "m_id": int(m_id),
            "type": 3,
            "accel_bins": int(accel_bins),
            "gyro_bins": int(gyro_bins),
            "outdim": int(outdim),
            "state": int(state),
            "data": bytes(data)
        }
        if tag is not None:
            pkg["tag"] = int(tag)
        return CBORCodec.encode(pkg)

    @staticmethod
    def build_telemetry_segment(
        segment_id: int,
        rate: float,
        reason: int,
        chunk_idx: int,
        gyro_x: List[float],
        gyro_y: List[float],
        gyro_z: List[float],
        accel_x: List[float],
        accel_y: List[float],
        accel_z: List[float]
    ) -> bytes:
        seg = {
            "start": int(segment_id),
            "end": int(segment_id),
            "rate": int(rate),
            "reason": int(reason),
            "chunk": int(chunk_idx),
            "data": {
                "gyro": {"x": gyro_x, "y": gyro_y, "z": gyro_z},
                "accel": {"x": accel_x, "y": accel_y, "z": accel_z}
            }
        }
        return CBORCodec.encode(seg)

    @staticmethod
    def build_inference_packet(
        segment_id: int,
        reason: int,
        r_m_id: int,
        ae_id: int,
        mse: float,
        anom: bool
    ) -> bytes:
        pkt = {
            "start": int(segment_id),
            "end": int(segment_id),
            "reason": int(reason),
            "r_m_id": int(r_m_id),
            "ae_id": int(ae_id),
            "mse": float(mse),
            "anom": bool(anom)
        }
        return CBORCodec.encode(pkt)

    @staticmethod
    def build_node_health_info(
        ram: int,
        sd: bool,
        cache: List[int],
        last: int,
        status: str = "idle",
        dump_t: Optional[int] = None,
        dump_r: Optional[bool] = None,
        ips: Optional[float] = None
    ) -> bytes:
        health = {
            "ram": int(ram),
            "sd": int(1 if sd else 0),
            "cache": [int(c) for c in cache],
            "last": int(last),
            "status": str(status),
            "caps": {
                "accel_freqs": [120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0],
                "gyro_freqs": [120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0],
                "enabled_ops": ["CONV_2D", "FULLY_CONNECTED", "RESHAPE"]
            }
        }
        if dump_t is not None:
            health["dump_t"] = int(dump_t)
        if dump_r is not None:
            health["dump_r"] = bool(dump_r)
        if ips is not None:
            health["ips"] = float(ips)
        return CBORCodec.encode(health)

    @staticmethod
    def build_node_capabilities(
        accel_freqs: List[float],
        gyro_freqs: List[float],
        enabled_ops: List[str]
    ) -> bytes:
        caps = {
            "accel_freqs": [float(f) for f in accel_freqs],
            "gyro_freqs": [float(f) for f in gyro_freqs],
            "enabled_ops": [str(op) for op in enabled_ops]
        }
        return CBORCodec.encode(caps)

    @staticmethod
    def build_node_alert(code: int, detail: str) -> bytes:
        alert = {
            "code": int(code),
            "detail": str(detail)
        }
        return CBORCodec.encode(alert)

    @staticmethod
    def build_command(dump: bool = False, dump_i: bool = False, reboot: bool = False) -> bytes:
        cmd = {
            "dump": bool(dump),
            "dump_i": bool(dump_i),
            "reboot": bool(reboot)
        }
        return CBORCodec.encode(cmd)


# ==============================================================================
# 3. SYNTHETIC TFLITE FLATBUFFER MODEL BINARY GENERATOR
# ==============================================================================

class SyntheticTFLiteModel:
    @staticmethod
    def generate(
        model_id: int,
        archetype: str = "autoencoder",
        input_shape: Tuple[int, ...] = (1, 16, 256),
        output_shape: Tuple[int, ...] = (1, 256),
        weights_seed: int = 42
    ) -> bytes:
        rng = np.random.default_rng(weights_seed + model_id)
        header = bytearray(32)
        struct.pack_into("<I", header, 0, 16)
        header[4:8] = b"TFL3"
        struct.pack_into("<I", header, 8, model_id)
        struct.pack_into("<I", header, 12, len(input_shape))
        for i, dim in enumerate(input_shape[:3]):
            struct.pack_into("<H", header, 16 + i * 2, dim)

        weight_data = rng.standard_normal(size=256, dtype=np.float32).tobytes()
        metadata = f"archetype={archetype};id={model_id}".encode("ascii")
        return bytes(header) + weight_data + metadata


# ==============================================================================
# 4. EMULATED EDGE SUBSYSTEMS FOR DETERMINISTIC E2E TESTING
# ==============================================================================

class EmulatedDSP:
    @staticmethod
    def compute_magnitudes(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
        return np.sqrt(ax * ax + ay * ay + az * az).astype(np.float32)

    @staticmethod
    def pool_1d_max_pow2(in_vec: np.ndarray, out_len: int) -> np.ndarray:
        in_len = len(in_vec)
        if in_len < out_len or out_len == 0:
            return in_vec[:out_len]
        K = in_len // out_len
        out = np.zeros(out_len, dtype=np.float32)
        for i in range(out_len):
            out[i] = np.max(in_vec[i * K : (i + 1) * K])
        return out

    @staticmethod
    def compute_stft(
        signal: np.ndarray,
        window_size: int = 256,
        hop_size: int = 240,
        n_fft: int = 256
    ) -> np.ndarray:
        num_frames = 16  # Fixed 16 frames in firmware
        num_bins = n_fft // 2  # 128 bins
        spectrogram = np.zeros((num_frames, num_bins), dtype=np.float32)

        window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1)))

        for i in range(num_frames):
            start = i * hop_size
            if start + window_size <= len(signal):
                frame = signal[start : start + window_size] * window
            else:
                frame = np.pad(signal[start:], (0, window_size - len(signal[start:]))) * window
            fft_res = np.fft.rfft(frame, n=n_fft)
            mag = np.abs(fft_res[:num_bins])
            spectrogram[i, :] = mag.astype(np.float32)

        return spectrogram

    @staticmethod
    def compute_loss(target: np.ndarray, reconstruction: np.ndarray, loss_mode: int) -> float:
        if loss_mode == 2:
            t = np.expm1(target)
            r = np.expm1(reconstruction)
            diff = t - r
        else:
            diff = target - reconstruction
        return float(np.mean(diff * diff))


class EmulatedRingBuffer:
    def __init__(self, depth: int = 256, stride: int = 256):
        self.depth = depth
        self.stride = stride
        self.data = np.zeros((depth, stride), dtype=np.float32)
        self.model_ids = np.zeros(depth, dtype=np.uint32)
        self.head = 0

    def clear(self):
        self.data.fill(0.0)
        self.model_ids.fill(0)
        self.head = 0

    def write(self, slice_data: np.ndarray, model_id: int = 0):
        self.data[self.head, :] = slice_data[:self.stride]
        self.model_ids[self.head] = model_id
        self.head = (self.head + 1) % self.depth

    def set_last_model_id(self, model_id: int):
        last_idx = (self.head - 1 + self.depth) % self.depth
        self.model_ids[last_idx] = model_id

    def get_slice(self, offset: int) -> np.ndarray:
        idx = (self.head + offset) % self.depth
        return self.data[idx, :]

    def unroll(self, T: int, target_C: int) -> np.ndarray:
        out = np.zeros((T, target_C), dtype=np.float32)
        for t in range(T):
            offset = -T + t
            src = self.get_slice(offset)
            if target_C < self.stride:
                out[t, :] = EmulatedDSP.pool_1d_max_pow2(src, target_C)
            else:
                out[t, :len(src)] = src[:target_C]
        return out

    def verify_sequence(self, target_id: int, T: int) -> bool:
        for t in range(T):
            offset = -T + t
            idx = (self.head + offset) % self.depth
            if self.model_ids[idx] != target_id:
                return False
        return True


class EmulatedModelCache:
    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.cache: List[Dict[str, Any]] = []

    def clear(self):
        self.cache.clear()

    def get(self, model_id: int) -> Optional[Dict[str, Any]]:
        for i, item in enumerate(self.cache):
            if item["config"]["model_id"] == model_id:
                mru = self.cache.pop(i)
                self.cache.insert(0, mru)
                return mru
        return None

    def insert(self, model_dict: Dict[str, Any]) -> int:
        if self.capacity == 0:
            return -1
        for i, item in enumerate(self.cache):
            if item["config"]["model_id"] == model_dict["config"]["model_id"]:
                self.cache.pop(i)
                self.cache.insert(0, model_dict)
                return 0
        if len(self.cache) >= self.capacity:
            self.cache.pop()
        self.cache.insert(0, model_dict)
        return 0

    def set_capacity(self, new_cap: int):
        self.capacity = new_cap
        while len(self.cache) > self.capacity:
            self.cache.pop()


class EmulatedModelEnsemble:
    def __init__(self, raw_bins: int = 256, history_depth: int = 256, warmup_steps: int = 5):
        self.raw_bins = raw_bins
        self.history_depth = history_depth
        self.warmup_steps = warmup_steps
        self.warmup_steps_done = 0
        self.ring_buffer = EmulatedRingBuffer(history_depth, raw_bins)
        self.submodel_cache = EmulatedModelCache(4)
        self.routes: Dict[int, int] = {}
        self.router_model: Optional[Dict[str, Any]] = None
        self.memory_model: Optional[Dict[str, Any]] = None
        self.h_state: Optional[np.ndarray] = None
        self.c_state: Optional[np.ndarray] = None
        self.state_dim = 0

    def reset_state(self):
        self.ring_buffer.clear()
        if self.h_state is not None:
            self.h_state.fill(0.0)
        if self.c_state is not None:
            self.c_state.fill(0.0)
        self.warmup_steps_done = 0

    def load_router(self, model_dict: Dict[str, Any]):
        self.router_model = model_dict

    def load_memory(self, model_dict: Dict[str, Any]):
        self.memory_model = model_dict
        self.state_dim = model_dict["config"].get("d", 32)
        self.h_state = np.zeros(self.state_dim, dtype=np.float32)
        self.c_state = np.zeros(self.state_dim, dtype=np.float32)

    def set_routes(self, routes: List[Dict[str, int]]):
        self.routes = {int(r.get("out_ix", r.get("out_idx", 0))): int(r["m_id"]) for r in routes}

    def inf_memory(self, fft_raw: np.ndarray):
        x_log = np.log1p(fft_raw[:self.raw_bins])
        self.ring_buffer.write(x_log, 0)
        if self.memory_model is not None and self.h_state is not None:
            mean_sig = np.mean(x_log)
            self.h_state = np.tanh(0.1 * mean_sig + 0.9 * self.h_state)
            self.c_state = self.c_state * 0.95 + 0.05 * self.h_state

    def inf_router(self) -> Tuple[int, bool, bool]:
        if self.router_model is None:
            return 0, False, False
        cfg = self.router_model["config"]
        depth = cfg["temporal_depth"]
        bins = cfg["frequency_bins"]
        num_classes = cfg.get("num_modes", 2)

        unrolled = self.ring_buffer.unroll(depth, bins)
        e_low = np.mean(unrolled[:, :bins//2])
        e_high = np.mean(unrolled[:, bins//2:])

        logits = np.array([e_low, e_high], dtype=np.float32)
        if len(logits) < num_classes:
            logits = np.pad(logits, (0, num_classes - len(logits)), "constant")
        elif len(logits) > num_classes:
            logits = logits[:num_classes]

        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)
        best_mode = int(np.argmax(probs))
        best_prob = float(probs[best_mode])

        if best_prob < 0.5:
            return 0, False, True

        target_m_id = self.routes.get(best_mode, 0)
        self.ring_buffer.set_last_model_id(target_m_id)
        cached = self.submodel_cache.get(target_m_id)
        return target_m_id, (cached is not None), False

    def inf_ae(self, num_frames: int = 16, skip_amount: int = 0) -> Tuple[float, bool]:
        if not self.submodel_cache.cache:
            return 0.0, False
        active = self.submodel_cache.cache[0]
        cfg = active["config"]
        depth = cfg["temporal_depth"]
        bins = cfg["frequency_bins"]
        threshold = cfg["anomaly_threshold"]
        loss_mode = cfg["loss_mode"]

        is_memory = self.memory_model is not None
        start_t = (num_frames - depth) if is_memory else 0
        end_t = num_frames - depth

        total_loss = 0.0
        evals = 0
        any_anom = False

        t = start_t
        while t <= end_t:
            target_slice = self.ring_buffer.get_slice(-num_frames + t + depth - 1)
            target_processed = EmulatedDSP.pool_1d_max_pow2(target_slice, bins) if bins < self.raw_bins else target_slice[:bins]

            mock_recon = target_processed * (0.95 if not cfg.get("inject_anomaly", False) else 0.4)
            loss = EmulatedDSP.compute_loss(target_processed, mock_recon, loss_mode)

            total_loss += loss
            if loss >= threshold:
                any_anom = True
            evals += 1

            if is_memory:
                break
            t += depth + skip_amount

        avg_loss = (total_loss / evals) if evals > 0 else 0.0

        self.warmup_steps_done += 1
        if self.warmup_steps_done <= self.warmup_steps:
            any_anom = False

        return avg_loss, any_anom


# ==============================================================================
# 5. MOCK MQTT BROKER & IN-MEMORY MESSAGE BUS
# ==============================================================================

@dataclass
class MQTTMessage:
    topic: str
    payload: bytes
    qos: int
    retain: bool
    timestamp: float = field(default_factory=time.time)


class MockMQTTBroker:
    def __init__(self):
        self.subscriptions: Dict[str, List[Tuple[Callable[[str, bytes], None], Optional[Any]]]] = {}
        self.published_messages: List[MQTTMessage] = []
        self.retained_messages: Dict[str, MQTTMessage] = {}
        self.is_connected = True
        self.outbox_capacity = 1000
        self.outbox: List[MQTTMessage] = []

    def connect(self):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def subscribe(self, topic: str, callback: Callable[[str, bytes], None], user_ctx: Optional[Any] = None):
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append((callback, user_ctx))

    def unsubscribe(self, topic: str):
        if topic in self.subscriptions:
            del self.subscriptions[topic]

    def publish(self, topic: str, payload: bytes, qos: int = 1, retain: bool = False) -> int:
        if not self.is_connected:
            if len(self.outbox) < self.outbox_capacity:
                self.outbox.append(MQTTMessage(topic, payload, qos, retain))
            return -1

        msg = MQTTMessage(topic=topic, payload=payload, qos=qos, retain=retain)
        self.published_messages.append(msg)

        if retain:
            self.retained_messages[topic] = msg

        for sub_topic, subs in self.subscriptions.items():
            if self._match_topic(sub_topic, topic):
                for cb, ctx in subs:
                    cb(topic, payload)

        return len(self.published_messages)

    def get_messages(self, topic_filter: Optional[str] = None) -> List[MQTTMessage]:
        if topic_filter is None:
            return list(self.published_messages)
        return [m for m in self.published_messages if self._match_topic(topic_filter, m.topic)]

    def clear(self):
        self.published_messages.clear()
        self.outbox.clear()

    @staticmethod
    def _match_topic(pattern: str, topic: str) -> bool:
        if pattern == topic or pattern == "#":
            return True
        p_parts = pattern.split("/")
        t_parts = topic.split("/")
        if len(p_parts) != len(t_parts) and not ("#" in p_parts):
            return False
        for p, t in zip(p_parts, t_parts):
            if p == "#":
                return True
            if p != "+" and p != t:
                return False
        return len(p_parts) == len(t_parts)


# ==============================================================================
# 6. SD STORAGE & RECORD FILE SYSTEM EMULATOR
# ==============================================================================

class MockSDStorage:
    RECORD_SIZE = (4096 * 6 * 4) + 64
    MAX_RECORDS_PER_FILE = (16 * 1024 * 1024) // RECORD_SIZE
    MAX_FILES = 64

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "data")
        self.models_dir = os.path.join(root_dir, "models")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.active_file_idx = 0
        self.record_count = 0
        self.enabled = True

    def save_state(self):
        state_file = os.path.join(self.data_dir, "state.bin")
        with open(state_file, "wb") as f:
            f.write(struct.pack("<II", self.active_file_idx, self.record_count))

    def load_state(self):
        state_file = os.path.join(self.data_dir, "state.bin")
        if os.path.exists(state_file):
            with open(state_file, "rb") as f:
                data = f.read(8)
                if len(data) == 8:
                    self.active_file_idx, self.record_count = struct.unpack("<II", data)

    def write_segment_record(
        self,
        segment_id: int,
        sample_rate: float,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
        gyro_x: np.ndarray,
        gyro_y: np.ndarray,
        gyro_z: np.ndarray,
        has_inf: bool = True,
        router_id: int = 1,
        mem_id: int = 0,
        submodel_id: int = 10,
        mse: float = 0.01,
        is_anom: bool = False,
        was_sent: bool = False
    ) -> str:
        if not self.enabled:
            return ""
        filepath = os.path.join(self.data_dir, f"data_{self.active_file_idx}.bin")
        mode = "wb" if self.record_count == 0 else "ab"

        record_bytes = bytearray(self.RECORD_SIZE)
        struct.pack_into("<I", record_bytes, 0, segment_id)
        struct.pack_into("<f", record_bytes, 4, sample_rate)
        struct.pack_into("<B", record_bytes, 8, 1 if has_inf else 0)
        struct.pack_into("<I", record_bytes, 9, router_id)
        struct.pack_into("<I", record_bytes, 13, mem_id)
        struct.pack_into("<I", record_bytes, 17, submodel_id)
        struct.pack_into("<f", record_bytes, 21, mse)
        struct.pack_into("<B", record_bytes, 25, 1 if is_anom else 0)
        struct.pack_into("<B", record_bytes, 26, 1 if was_sent else 0)

        offset = 64
        for arr in (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z):
            b = arr.astype(np.float32).tobytes()
            record_bytes[offset : offset + len(b)] = b
            offset += len(b)

        with open(filepath, mode) as f:
            f.write(record_bytes)

        self.record_count += 1
        if self.record_count >= self.MAX_RECORDS_PER_FILE:
            self.record_count = 0
            self.active_file_idx = (self.active_file_idx + 1) % self.MAX_FILES

        self.save_state()
        return filepath

    def save_model(self, model_id: int, binary_cbor: bytes):
        path = os.path.join(self.models_dir, f"model_{model_id}.bin")
        with open(path, "wb") as f:
            f.write(binary_cbor)

    def delete_model(self, model_id: int) -> bool:
        path = os.path.join(self.models_dir, f"model_{model_id}.bin")
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    def model_exists(self, model_id: int) -> bool:
        path = os.path.join(self.models_dir, f"model_{model_id}.bin")
        return os.path.exists(path)


# ==============================================================================
# 7. COMPLETE SYSTEM PIPELINE SIMULATOR FOR FULL-STACK E2E TESTING
# ==============================================================================

class EdgeSystemSimulator:
    def __init__(self, node_id: str = "node_esp32_test", storage_dir: Optional[str] = None):
        self.node_id = node_id
        self.storage_dir = storage_dir or tempfile.mkdtemp(prefix="esp32_sd_")
        self.sd = MockSDStorage(self.storage_dir)
        self.mqtt = MockMQTTBroker()

        self.sample_rate = 3840.0
        self.stream_mode = 1
        self.cadence = 10
        self.heartbeat_s = 5
        self.sd_enabled = True

        self.segment_counter = 1
        self.last_segment_id = 0
        self.dump_time_ms = 0
        self.dump_reading = False
        self.dump_ips = 0.0

        self.ensemble = EmulatedModelEnsemble(raw_bins=256, history_depth=256, warmup_steps=5)
        self._setup_mqtt_handlers()

    def _setup_mqtt_handlers(self):
        self.mqtt.subscribe(f"v1/{self.node_id}/config", self._on_config)
        self.mqtt.subscribe(f"v1/{self.node_id}/ensemble", self._on_ensemble)
        self.mqtt.subscribe(f"v1/{self.node_id}/cmd/dump", lambda t, p: self.trigger_dump(run_inference=False))
        self.mqtt.subscribe(f"v1/{self.node_id}/cmd/dump_i", lambda t, p: self.trigger_dump(run_inference=True))
        self.mqtt.subscribe(f"v1/{self.node_id}/cmd/reboot", lambda t, p: self.reboot())
        self.mqtt.subscribe(f"v1/{self.node_id}/cmd/updt_status", lambda t, p: self.publish_health())

    def _on_config(self, topic: str, payload: bytes):
        try:
            cfg = CBORCodec.decode_all(payload)
            if "rate" in cfg:
                self.sample_rate = float(cfg["rate"])
            if "mode" in cfg:
                self.stream_mode = int(cfg["mode"])
            if "sd_en" in cfg:
                self.sd_enabled = bool(cfg["sd_en"])
                self.sd.enabled = self.sd_enabled
            if "beat" in cfg:
                self.heartbeat_s = int(cfg["beat"])
            if "cad" in cfg:
                self.cadence = int(cfg["cad"])
        except Exception:
            pass

    def _on_ensemble(self, topic: str, payload: bytes):
        try:
            pkg = CBORCodec.decode_all(payload)
            if "routes" in pkg:
                old_routes = set(self.ensemble.routes.values())
                new_routes = set(r["m_id"] for r in pkg["routes"])
                evicted = old_routes - new_routes
                for m_id in evicted:
                    self.sd.delete_model(m_id)
                self.ensemble.set_routes(pkg["routes"])
                if "warmup" in pkg:
                    self.ensemble.warmup_steps = int(pkg["warmup"])
            elif "type" in pkg:
                m_type = pkg["type"]
                m_id = pkg["m_id"]
                if m_type == 1:
                    model_dict = {
                        "config": {
                            "model_id": m_id,
                            "archetype": 1,
                            "temporal_depth": pkg.get("tsteps", 16),
                            "frequency_bins": pkg.get("accel_bins", 128) + pkg.get("gyro_bins", 128),
                            "anomaly_threshold": float(pkg.get("limit", 0.05)),
                            "loss_mode": int(pkg.get("loss", 1)),
                            "skip_amount": int(pkg.get("skip", 0))
                        },
                        "data": pkg["data"]
                    }
                    self.ensemble.submodel_cache.insert(model_dict)
                    self.sd.save_model(m_id, payload)
                elif m_type == 2:
                    model_dict = {
                        "config": {
                            "model_id": m_id,
                            "archetype": 2,
                            "temporal_depth": pkg.get("tsteps", 16),
                            "frequency_bins": pkg.get("accel_bins", 128) + pkg.get("gyro_bins", 128),
                            "num_modes": pkg.get("class", 2)
                        },
                        "data": pkg["data"]
                    }
                    self.ensemble.load_router(model_dict)
                    self.sd.save_model(m_id, payload)
                elif m_type == 3:
                    model_dict = {
                        "config": {
                            "model_id": m_id,
                            "archetype": 3,
                            "frequency_bins": pkg.get("accel_bins", 128) + pkg.get("gyro_bins", 128),
                            "d": pkg.get("state", 32)
                        },
                        "data": pkg["data"]
                    }
                    self.ensemble.load_memory(model_dict)
                    self.sd.save_model(m_id, payload)
        except Exception:
            pass

    def ingest_sensor_segment(
        self,
        accel_x: np.ndarray,
        accel_y: np.ndarray,
        accel_z: np.ndarray,
        gyro_x: np.ndarray,
        gyro_y: np.ndarray,
        gyro_z: np.ndarray,
        fifo_overrun: bool = False
    ) -> Dict[str, Any]:
        seg_id = self.segment_counter
        if fifo_overrun:
            self.segment_counter += 2
        else:
            self.segment_counter += 1

        self.last_segment_id = seg_id

        accel_mag = EmulatedDSP.compute_magnitudes(accel_x, accel_y, accel_z)
        gyro_mag = EmulatedDSP.compute_magnitudes(gyro_x, gyro_y, gyro_z)
        accel_spec = EmulatedDSP.compute_stft(accel_mag)
        gyro_spec = EmulatedDSP.compute_stft(gyro_mag)

        num_frames = accel_spec.shape[0]
        for t in range(num_frames):
            acc_p = EmulatedDSP.pool_1d_max_pow2(accel_spec[t], 128)
            gyr_p = EmulatedDSP.pool_1d_max_pow2(gyro_spec[t], 128)
            slice_256 = np.concatenate([acc_p, gyr_p])
            self.ensemble.inf_memory(slice_256)

        r_m_id = self.ensemble.router_model["config"]["model_id"] if self.ensemble.router_model else 0
        mem_id = self.ensemble.memory_model["config"]["model_id"] if self.ensemble.memory_model else 0
        target_submodel, already_loaded, router_anom = self.ensemble.inf_router()

        if router_anom:
            is_anomaly = True
            mse = 1.0
            ae_id = 0
        else:
            ae_id = target_submodel
            mse, is_anomaly = self.ensemble.inf_ae(num_frames=16, skip_amount=0)

        should_send_raw = False
        should_send_inf = True
        reason = 1

        if self.stream_mode == 1:
            should_send_raw = True
            reason = 1
        elif self.stream_mode == 2:
            if is_anomaly:
                should_send_raw = True
                should_send_inf = True
                reason = 2
            else:
                should_send_raw = False
                should_send_inf = False
        elif self.stream_mode == 3:
            if is_anomaly:
                should_send_raw = True
                should_send_inf = True
                reason = 2
            elif self.cadence > 0 and (seg_id % self.cadence == 0):
                should_send_raw = True
                should_send_inf = True
                reason = 3
            else:
                should_send_raw = False
                should_send_inf = False
        elif self.stream_mode == 4:
            should_send_inf = True
            if is_anomaly:
                should_send_raw = True
                reason = 2
            else:
                should_send_raw = False
                reason = 1

        was_sent = should_send_raw or should_send_inf
        if self.sd_enabled:
            self.sd.write_segment_record(
                segment_id=seg_id,
                sample_rate=self.sample_rate,
                accel_x=accel_x, accel_y=accel_y, accel_z=accel_z,
                gyro_x=gyro_x, gyro_y=gyro_y, gyro_z=gyro_z,
                has_inf=True, router_id=r_m_id, mem_id=mem_id,
                submodel_id=ae_id, mse=mse, is_anom=is_anomaly,
                was_sent=was_sent
            )

        if should_send_raw:
            chunk_size = 512
            num_chunks = len(accel_x) // chunk_size
            for c in range(num_chunks):
                start = c * chunk_size
                end = start + chunk_size
                payload = SchemaBuilder.build_telemetry_segment(
                    segment_id=seg_id,
                    rate=self.sample_rate,
                    reason=reason,
                    chunk_idx=c + 1,
                    gyro_x=gyro_x[start:end].tolist(),
                    gyro_y=gyro_y[start:end].tolist(),
                    gyro_z=gyro_z[start:end].tolist(),
                    accel_x=accel_x[start:end].tolist(),
                    accel_y=accel_y[start:end].tolist(),
                    accel_z=accel_z[start:end].tolist()
                )
                self.mqtt.publish(f"v1/{self.node_id}/data/sensor", payload)

        if should_send_inf:
            inf_payload = SchemaBuilder.build_inference_packet(
                segment_id=seg_id,
                reason=reason,
                r_m_id=r_m_id,
                ae_id=ae_id,
                mse=mse,
                anom=is_anomaly
            )
            self.mqtt.publish(f"v1/{self.node_id}/inference", inf_payload)

        return {
            "segment_id": seg_id,
            "mse": mse,
            "is_anomaly": is_anomaly,
            "router_id": r_m_id,
            "ae_id": ae_id,
            "was_sent": was_sent
        }

    def trigger_dump(self, run_inference: bool = False):
        self.dump_reading = True
        t0 = time.time()
        replayed_count = 0

        for file_idx in range(MockSDStorage.MAX_FILES):
            filepath = os.path.join(self.sd.data_dir, f"data_{file_idx}.bin")
            if not os.path.exists(filepath):
                continue
            with open(filepath, "rb") as f:
                while True:
                    rec_bytes = f.read(MockSDStorage.RECORD_SIZE)
                    if len(rec_bytes) < MockSDStorage.RECORD_SIZE:
                        break
                    seg_id, rate, has_inf, r_id, m_id, ae_id, mse, is_anom, was_sent = struct.unpack("<IfBIIIfBB", rec_bytes[:27])
                    if was_sent:
                        continue

                    offset = 64
                    floats_per_axis = 4096
                    ax = np.frombuffer(rec_bytes[offset : offset + floats_per_axis * 4], dtype=np.float32)
                    offset += floats_per_axis * 4
                    ay = np.frombuffer(rec_bytes[offset : offset + floats_per_axis * 4], dtype=np.float32)
                    offset += floats_per_axis * 4
                    az = np.frombuffer(rec_bytes[offset : offset + floats_per_axis * 4], dtype=np.float32)
                    offset += floats_per_axis * 4
                    gx = np.frombuffer(rec_bytes[offset : offset + floats_per_axis * 4], dtype=np.float32)
                    offset += floats_per_axis * 4
                    gy = np.frombuffer(rec_bytes[offset : offset + floats_per_axis * 4], dtype=np.float32)
                    offset += floats_per_axis * 4
                    gz = np.frombuffer(rec_bytes[offset : offset + floats_per_axis * 4], dtype=np.float32)

                    payload = SchemaBuilder.build_telemetry_segment(
                        segment_id=seg_id,
                        rate=rate,
                        reason=4,
                        chunk_idx=1,
                        gyro_x=gx[:512].tolist(),
                        gyro_y=gy[:512].tolist(),
                        gyro_z=gz[:512].tolist(),
                        accel_x=ax[:512].tolist(),
                        accel_y=ay[:512].tolist(),
                        accel_z=az[:512].tolist()
                    )
                    self.mqtt.publish(f"v1/{self.node_id}/data/sensor", payload)

                    if run_inference:
                        inf_payload = SchemaBuilder.build_inference_packet(
                            segment_id=seg_id,
                            reason=4,
                            r_m_id=r_id,
                            ae_id=ae_id,
                            mse=mse,
                            anom=bool(is_anom)
                        )
                        self.mqtt.publish(f"v1/{self.node_id}/inference", inf_payload)

                    replayed_count += 1

        t1 = time.time()
        elapsed_ms = max(int((t1 - t0) * 1000), 1)
        self.dump_time_ms = elapsed_ms
        self.dump_reading = False
        self.dump_ips = (float(replayed_count) / (elapsed_ms / 1000.0)) if run_inference else 0.0

        self.publish_health()

    def publish_health(self):
        cached_ids = [m["config"]["model_id"] for m in self.ensemble.submodel_cache.cache]
        if self.ensemble.router_model:
            cached_ids.insert(0, self.ensemble.router_model["config"]["model_id"])
        if self.ensemble.memory_model:
            cached_ids.insert(0, self.ensemble.memory_model["config"]["model_id"])

        health_payload = SchemaBuilder.build_node_health_info(
            ram=240000,
            sd=self.sd_enabled,
            cache=cached_ids[:3],
            last=self.last_segment_id,
            status="streaming_dump" if self.dump_reading else "idle",
            dump_t=self.dump_time_ms if self.dump_time_ms > 0 else None,
            dump_r=self.dump_reading if self.dump_time_ms > 0 else None,
            ips=self.dump_ips if self.dump_time_ms > 0 else None
        )
        self.mqtt.publish(f"v1/{self.node_id}/info/health", health_payload)

    def publish_capabilities(self):
        caps_payload = SchemaBuilder.build_node_capabilities(
            accel_freqs=[120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0],
            gyro_freqs=[120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0],
            enabled_ops=["CONV_2D", "FULLY_CONNECTED", "RESHAPE", "SOFTMAX", "TANH"]
        )
        self.mqtt.publish(f"v1/{self.node_id}/info/caps", caps_payload, qos=1, retain=True)

    def reboot(self):
        self.ensemble.reset_state()
        self.publish_health()


# ==============================================================================
# 8. PYTEST FIXTURES
# ==============================================================================

@pytest.fixture
def mock_broker() -> MockMQTTBroker:
    return MockMQTTBroker()


@pytest.fixture
def mock_sd(tmp_path) -> MockSDStorage:
    return MockSDStorage(str(tmp_path))


@pytest.fixture
def edge_simulator(tmp_path) -> EdgeSystemSimulator:
    sim = EdgeSystemSimulator(node_id="test_node_01", storage_dir=str(tmp_path))
    sim.publish_capabilities()
    return sim


@pytest.fixture
def synthetic_sensor_segment() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0, 4096 / 3840.0, 4096, endpoint=False)
    ax = np.sin(2 * np.pi * 50 * t).astype(np.float32) + 0.05 * np.random.randn(4096).astype(np.float32)
    ay = 0.5 * np.cos(2 * np.pi * 50 * t).astype(np.float32) + 0.05 * np.random.randn(4096).astype(np.float32)
    az = 9.81 + 0.1 * np.sin(2 * np.pi * 120 * t).astype(np.float32)

    gx = 0.1 * np.sin(2 * np.pi * 25 * t).astype(np.float32)
    gy = 0.1 * np.cos(2 * np.pi * 25 * t).astype(np.float32)
    gz = 0.05 * np.random.randn(4096).astype(np.float32)
    return ax, ay, az, gx, gy, gz


@pytest.fixture
def anomaly_sensor_segment() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0, 4096 / 3840.0, 4096, endpoint=False)
    ax = 10.0 * np.sin(2 * np.pi * 400 * t).astype(np.float32) + 2.0 * np.random.randn(4096).astype(np.float32)
    ay = 8.0 * np.cos(2 * np.pi * 400 * t).astype(np.float32) + 2.0 * np.random.randn(4096).astype(np.float32)
    az = 9.81 + 5.0 * np.sin(2 * np.pi * 800 * t).astype(np.float32)

    gx = 5.0 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
    gy = 4.0 * np.cos(2 * np.pi * 200 * t).astype(np.float32)
    gz = 3.0 * np.random.randn(4096).astype(np.float32)
    return ax, ay, az, gx, gy, gz
