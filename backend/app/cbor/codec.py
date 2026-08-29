# CBOR binary codec implementation
import struct
from typing import Any, Dict, List, Optional, Type, Union
import cbor2
from pydantic import ValidationError

from app.cbor.schemas import (
    EmitReason,
    StreamMode,
    ModelType,
    LossMode,
    ArchitectureTag,
    Datapoints,
    SegmentData,
    Segment,
    InferencePacket,
    NodeCapabilities,
    NodeHealthInfo,
    NodeAlert,
    Command,
    RuntimeConfig,
    RouteEntry,
    EnsembleConfig,
    BaseModelPackage,
    MemoryModelPackage,
    AutoencoderModelPackage,
    RouterModelPackage,
    ModelPackageUnion,
    PayloadUnion,
)

# base exception class
class CBORError(Exception):
    # base error for cbor codec
    pass

class CBOREncodeError(CBORError):
    # raised when encoding fails
    pass

class CBORDecodeError(CBORError):
    # raised when decoding fails
    pass

class CBORValidationError(CBORError):
    # raised when payload fails schema validation
    pass

# Helper to clean None values from dictionary
def _strip_nones(d: Dict[str, Any]) -> Dict[str, Any]:
    # remove keys with None values for CDDL conformance
    out = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            out[k] = _strip_nones(v)
        elif isinstance(v, list):
            out[k] = [_strip_nones(item) if isinstance(item, dict) else item for item in v]
        else:
            out[k] = v
    return out

# Segment encode / decode
def encode_segment(segment: Segment) -> bytes:
    # encode Segment to cbor bytes
    try:
        # validate emission reason
        if segment.reason not in (1, 2, 3, 4):
            raise CBORValidationError(f"Invalid emit_reason: {segment.reason}")
        
        payload = {
            "id": int(segment.id),
            "rate": float(segment.rate),
            "reason": int(segment.reason),
            "chunk": int(segment.chunk),
            "data": {
                "gyro": {
                    "x": [float(v) for v in segment.data.gyro.x],
                    "y": [float(v) for v in segment.data.gyro.y],
                    "z": [float(v) for v in segment.data.gyro.z],
                },
                "accel": {
                    "x": [float(v) for v in segment.data.accel.x],
                    "y": [float(v) for v in segment.data.accel.y],
                    "z": [float(v) for v in segment.data.accel.z],
                },
            }
        }
        return cbor2.dumps(payload)
    except CBORValidationError:
        raise
    except Exception as e:
        raise CBOREncodeError(f"Failed to encode Segment: {e}") from e

def decode_segment(data: bytes) -> Segment:
    # decode cbor bytes to Segment
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
    except Exception as e:
        raise CBORDecodeError(f"Corrupt CBOR byte stream: {e}") from e
    
    if not isinstance(obj, dict):
        raise CBORValidationError("Decoded CBOR root is not a map")
    
    try:
        seg = Segment.model_validate(obj)
        if seg.reason not in (1, 2, 3, 4):
            raise CBORValidationError(f"Invalid emit_reason: {seg.reason}")
        return seg
    except ValidationError as e:
        raise CBORValidationError(f"Schema validation error for Segment: {e}") from e

# InferencePacket encode / decode
def encode_inference_packet(packet: InferencePacket) -> bytes:
    # encode InferencePacket to cbor bytes
    try:
        if packet.reason not in (1, 2, 3, 4):
            raise CBORValidationError(f"Invalid emit_reason: {packet.reason}")
        
        payload = {
            "id": int(packet.id),
            "reason": int(packet.reason),
            "r_m_id": int(packet.r_m_id),
            "ae_id": int(packet.ae_id),
            "mse": float(packet.mse),
            "anom": bool(packet.anom),
        }
        return cbor2.dumps(payload)
    except CBORValidationError:
        raise
    except Exception as e:
        raise CBOREncodeError(f"Failed to encode InferencePacket: {e}") from e

def decode_inference_packet(data: bytes) -> InferencePacket:
    # decode cbor bytes to InferencePacket
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
    except Exception as e:
        raise CBORDecodeError(f"Corrupt CBOR byte stream: {e}") from e
    
    if not isinstance(obj, dict):
        raise CBORValidationError("Decoded CBOR root is not a map")
    
    try:
        pkt = InferencePacket.model_validate(obj)
        if pkt.reason not in (1, 2, 3, 4):
            raise CBORValidationError(f"Invalid emit_reason: {pkt.reason}")
        return pkt
    except ValidationError as e:
        raise CBORValidationError(f"Schema validation error for InferencePacket: {e}") from e

# NodeCapabilities encode / decode
def encode_node_capabilities(caps: NodeCapabilities) -> bytes:
    # encode NodeCapabilities to cbor bytes
    try:
        payload = {
            "accel_freqs": [float(f) for f in caps.accel_freqs],
            "gyro_freqs": [float(f) for f in caps.gyro_freqs],
            "enabled_ops": [str(op) for op in caps.enabled_ops],
        }
        return cbor2.dumps(payload)
    except Exception as e:
        raise CBOREncodeError(f"Failed to encode NodeCapabilities: {e}") from e

def decode_node_capabilities(data: bytes) -> NodeCapabilities:
    # decode cbor bytes to NodeCapabilities
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
    except Exception as e:
        raise CBORDecodeError(f"Corrupt CBOR byte stream: {e}") from e
    
    if not isinstance(obj, dict):
        raise CBORValidationError("Decoded CBOR root is not a map")
    
    try:
        return NodeCapabilities.model_validate(obj)
    except ValidationError as e:
        raise CBORValidationError(f"Schema validation error for NodeCapabilities: {e}") from e

# NodeHealthInfo encode / decode
def encode_node_health(health: NodeHealthInfo) -> bytes:
    # encode NodeHealthInfo to cbor bytes
    try:
        payload: Dict[str, Any] = {
            "ram": int(health.ram),
            "sd": int(health.sd),
            "cache": [int(c) for c in health.cache],
            "last": int(health.last),
        }
        if health.status is not None:
            payload["status"] = str(health.status)
        if health.caps is not None:
            payload["caps"] = {
                "accel_freqs": [float(f) for f in health.caps.accel_freqs],
                "gyro_freqs": [float(f) for f in health.caps.gyro_freqs],
                "enabled_ops": [str(op) for op in health.caps.enabled_ops],
            }
        if health.dump_t is not None:
            payload["dump_t"] = int(health.dump_t)
        if health.dump_r is not None:
            payload["dump_r"] = bool(health.dump_r)
        if health.ips is not None:
            payload["ips"] = float(health.ips)
            
        return cbor2.dumps(payload)
    except Exception as e:
        raise CBOREncodeError(f"Failed to encode NodeHealthInfo: {e}") from e

def decode_node_health(data: bytes) -> NodeHealthInfo:
    # decode cbor bytes to NodeHealthInfo
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
    except Exception as e:
        raise CBORDecodeError(f"Corrupt CBOR byte stream: {e}") from e
    
    if not isinstance(obj, dict):
        raise CBORValidationError("Decoded CBOR root is not a map")
    
    try:
        return NodeHealthInfo.model_validate(obj)
    except ValidationError as e:
        raise CBORValidationError(f"Schema validation error for NodeHealthInfo: {e}") from e

# RuntimeConfig encode / decode
def encode_runtime_config(config: RuntimeConfig) -> bytes:
    # encode RuntimeConfig to cbor bytes
    try:
        if config.mode not in (1, 2, 3, 4):
            raise CBORValidationError(f"Invalid StreamMode: {config.mode}")
        
        payload: Dict[str, Any] = {
            "rate": float(config.rate),
            "mode": int(config.mode),
            "batch": int(config.batch),
            "beat": int(config.beat),
            "sd_en": bool(config.sd_en),
        }
        if config.cad is not None:
            payload["cad"] = int(config.cad)
        return cbor2.dumps(payload)
    except CBORValidationError:
        raise
    except Exception as e:
        raise CBOREncodeError(f"Failed to encode RuntimeConfig: {e}") from e

def decode_runtime_config(data: bytes) -> RuntimeConfig:
    # decode cbor bytes to RuntimeConfig
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
    except Exception as e:
        raise CBORDecodeError(f"Corrupt CBOR byte stream: {e}") from e
    
    if not isinstance(obj, dict):
        raise CBORValidationError("Decoded CBOR root is not a map")
    
    try:
        cfg = RuntimeConfig.model_validate(obj)
        if cfg.mode not in (1, 2, 3, 4):
            raise CBORValidationError(f"Invalid StreamMode: {cfg.mode}")
        return cfg
    except ValidationError as e:
        raise CBORValidationError(f"Schema validation error for RuntimeConfig: {e}") from e

# EnsembleConfig encode / decode
def encode_ensemble_config(ensemble: EnsembleConfig) -> bytes:
    # encode EnsembleConfig to cbor bytes
    try:
        if not ensemble.routes:
            raise CBORValidationError("EnsembleConfig must contain at least 1 route entry")
        if len(ensemble.routes) > 16:
            raise CBORValidationError("EnsembleConfig cannot exceed 16 route entries")
        
        payload: Dict[str, Any] = {
            "warmup": int(ensemble.warmup),
            "r_m_id": int(ensemble.r_m_id),
            "routes": [{"out_ix": int(r.out_ix), "m_id": int(r.m_id)} for r in ensemble.routes]
        }
        if ensemble.mem_id is not None:
            payload["mem_id"] = int(ensemble.mem_id)
        return cbor2.dumps(payload)
    except CBORValidationError:
        raise
    except Exception as e:
        raise CBOREncodeError(f"Failed to encode EnsembleConfig: {e}") from e

def decode_ensemble_config(data: bytes) -> EnsembleConfig:
    # decode cbor bytes to EnsembleConfig
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
    except Exception as e:
        raise CBORDecodeError(f"Corrupt CBOR byte stream: {e}") from e
    
    if not isinstance(obj, dict):
        raise CBORValidationError("Decoded CBOR root is not a map")
    
    try:
        ens = EnsembleConfig.model_validate(obj)
        if not ens.routes:
            raise CBORValidationError("EnsembleConfig must contain at least 1 route entry")
        if len(ens.routes) > 16:
            raise CBORValidationError("EnsembleConfig cannot exceed 16 route entries")
        return ens
    except ValidationError as e:
        raise CBORValidationError(f"Schema validation error for EnsembleConfig: {e}") from e

# ModelPackage encode / decode
def encode_model_package(model: ModelPackageUnion) -> bytes:
    # encode ModelPackage to cbor bytes
    try:
        if not isinstance(model.data, (bytes, bytearray)):
            raise CBORValidationError("ModelPackage data must be raw binary bytes")
        
        payload: Dict[str, Any] = {
            "m_id": int(model.m_id),
            "type": int(model.type),
            "data": bytes(model.data),
        }
        if model.tag is not None:
            payload["tag"] = int(model.tag)
            
        if isinstance(model, AutoencoderModelPackage):
            payload["accel_bins"] = int(model.accel_bins)
            payload["gyro_bins"] = int(model.gyro_bins)
            payload["tsteps"] = int(model.tsteps)
            payload["limit"] = float(model.limit)
            payload["loss"] = int(model.loss)
            if model.mem_d is not None:
                payload["mem_d"] = int(model.mem_d)
            if model.skip is not None:
                payload["skip"] = int(model.skip)
        elif isinstance(model, RouterModelPackage):
            payload["accel_bins"] = int(model.accel_bins)
            payload["gyro_bins"] = int(model.gyro_bins)
            payload["tsteps"] = int(model.tsteps)
            payload["class"] = int(model.class_count)
            if model.mem_d is not None:
                payload["mem_d"] = int(model.mem_d)
        elif isinstance(model, MemoryModelPackage):
            payload["accel_bins"] = int(model.accel_bins)
            payload["gyro_bins"] = int(model.gyro_bins)
            payload["outdim"] = int(model.outdim)
            payload["state"] = int(model.state)
        else:
            raise CBORValidationError(f"Unknown model package type: {type(model)}")
            
        return cbor2.dumps(payload)
    except CBORValidationError:
        raise
    except Exception as e:
        raise CBOREncodeError(f"Failed to encode ModelPackage: {e}") from e

def decode_model_package(data: bytes) -> ModelPackageUnion:
    # decode cbor bytes to specific ModelPackage
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
    except Exception as e:
        raise CBORDecodeError(f"Corrupt CBOR byte stream: {e}") from e
    
    if not isinstance(obj, dict):
        raise CBORValidationError("Decoded CBOR root is not a map")
    
    m_type = obj.get("type")
    if m_type is None:
        raise CBORValidationError("Missing required 'type' field in ModelPackage")
        
    try:
        if m_type == int(ModelType.AUTOENCODER):
            return AutoencoderModelPackage.model_validate(obj)
        elif m_type == int(ModelType.ROUTER):
            return RouterModelPackage.model_validate(obj)
        elif m_type == int(ModelType.MEMORY):
            return MemoryModelPackage.model_validate(obj)
        else:
            raise CBORValidationError(f"Unsupported model type tag: {m_type}")
    except ValidationError as e:
        raise CBORValidationError(f"Schema validation error for ModelPackage (type {m_type}): {e}") from e

# Generic Payload encode / decode
def encode_payload(payload: PayloadUnion) -> bytes:
    # encode Payload union
    if isinstance(payload, EnsembleConfig):
        return encode_ensemble_config(payload)
    elif isinstance(payload, (AutoencoderModelPackage, RouterModelPackage, MemoryModelPackage)):
        return encode_model_package(payload)
    else:
        raise CBOREncodeError(f"Unsupported payload union type: {type(payload)}")

def decode_payload(data: bytes) -> PayloadUnion:
    # decode Payload union
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
    except Exception as e:
        raise CBORDecodeError(f"Corrupt CBOR byte stream: {e}") from e
    
    if not isinstance(obj, dict):
        raise CBORValidationError("Decoded CBOR root is not a map")
    
    if "routes" in obj or "warmup" in obj:
        return decode_ensemble_config(data)
    elif "type" in obj:
        return decode_model_package(data)
    else:
        raise CBORValidationError("Payload is neither EnsembleConfig nor ModelPackage")

# Alert and Command helpers
def encode_node_alert(alert: NodeAlert) -> bytes:
    # encode NodeAlert
    try:
        payload = {
            "code": int(alert.code),
            "detail": str(alert.detail),
        }
        return cbor2.dumps(payload)
    except Exception as e:
        raise CBOREncodeError(f"Failed to encode NodeAlert: {e}") from e

def decode_node_alert(data: bytes) -> NodeAlert:
    # decode NodeAlert
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
        return NodeAlert.model_validate(obj)
    except Exception as e:
        raise CBORDecodeError(f"Failed to decode NodeAlert: {e}") from e

def encode_command(cmd: Command) -> bytes:
    # encode Command
    try:
        payload = {
            "dump": bool(cmd.dump),
            "dump_i": bool(cmd.dump_i),
            "reboot": bool(cmd.reboot),
        }
        return cbor2.dumps(payload)
    except Exception as e:
        raise CBOREncodeError(f"Failed to encode Command: {e}") from e

def decode_command(data: bytes) -> Command:
    # decode Command
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise CBORDecodeError("Input data must be bytes-like")
    try:
        obj = cbor2.loads(data)
        return Command.model_validate(obj)
    except Exception as e:
        raise CBORDecodeError(f"Failed to decode Command: {e}") from e

# Generic Top-Level Functions
def encode(obj: Any) -> bytes:
    # generic cbor encode dispatcher
    if isinstance(obj, Segment):
        return encode_segment(obj)
    elif isinstance(obj, InferencePacket):
        return encode_inference_packet(obj)
    elif isinstance(obj, NodeCapabilities):
        return encode_node_capabilities(obj)
    elif isinstance(obj, NodeHealthInfo):
        return encode_node_health(obj)
    elif isinstance(obj, RuntimeConfig):
        return encode_runtime_config(obj)
    elif isinstance(obj, EnsembleConfig):
        return encode_ensemble_config(obj)
    elif isinstance(obj, (AutoencoderModelPackage, RouterModelPackage, MemoryModelPackage)):
        return encode_model_package(obj)
    elif isinstance(obj, NodeAlert):
        return encode_node_alert(obj)
    elif isinstance(obj, Command):
        return encode_command(obj)
    else:
        # Fallback to standard cbor dumps
        try:
            return cbor2.dumps(obj)
        except Exception as e:
            raise CBOREncodeError(f"Failed to encode object: {e}") from e

def decode(data: bytes, target_type: Optional[Type] = None) -> Any:
    # generic cbor decode dispatcher
    if target_type is Segment:
        return decode_segment(data)
    elif target_type is InferencePacket:
        return decode_inference_packet(data)
    elif target_type is NodeCapabilities:
        return decode_node_capabilities(data)
    elif target_type is NodeHealthInfo:
        return decode_node_health(data)
    elif target_type is RuntimeConfig:
        return decode_runtime_config(data)
    elif target_type is EnsembleConfig:
        return decode_ensemble_config(data)
    elif target_type in (ModelPackageUnion, AutoencoderModelPackage, RouterModelPackage, MemoryModelPackage):
        return decode_model_package(data)
    elif target_type is NodeAlert:
        return decode_node_alert(data)
    elif target_type is Command:
        return decode_command(data)
    else:
        # decode raw python object
        try:
            return cbor2.loads(data)
        except Exception as e:
            raise CBORDecodeError(f"Corrupt CBOR byte stream: {e}") from e

# aliases for explicit cbor encoding
encode_cbor = encode
decode_cbor = decode

