# CBOR data models and CDDL schemas
from enum import IntEnum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, model_validator

# Emission reason enumeration
class EmitReason(IntEnum):
    CONTINUOUS = 1
    ANOMALY = 2
    PERIODIC = 3
    MANUAL_DUMP = 4

# stream mode enumeration
class StreamMode(IntEnum):
    CONTINUOUS = 1
    ANOMALY_ONLY = 2
    ANOMALY_OR_PERIODIC = 3
    CONTINUOUS_SCORE_RAW_ANOMALY = 4

# model archetype enumeration
class ModelType(IntEnum):
    AUTOENCODER = 1
    ROUTER = 2
    MEMORY = 3

# architecture tags
class ArchitectureTag(IntEnum):
    VA = 1
    CA_1D = 2
    CA_2D = 3
    CLSTM = 4
    DA = 5
    FITS = 6
    TN = 7
    DENSE_R = 10
    STFT_MCNN = 12
    ONE_D_CNN_R = 13
    LSTM_MEM = 20

# Loss mode enumeration
class LossMode(IntEnum):
    LOG_MSE = 1
    LINEAR_MSE = 2

# 3-axis motion datapoints
class Datapoints(BaseModel):
    x: List[float] = Field(default_factory=list)
    y: List[float] = Field(default_factory=list)
    z: List[float] = Field(default_factory=list)

# segment data container
class SegmentData(BaseModel):
    gyro: Datapoints = Field(default_factory=Datapoints)
    accel: Datapoints = Field(default_factory=Datapoints)

# raw telemetry segment
class Segment(BaseModel):
    id: int
    rate: float
    reason: int
    chunk: int = 1
    data: SegmentData

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        # handle alias keys like start/end for id
        if isinstance(data, dict):
            if "id" not in data and "start" in data:
                data["id"] = data["start"]
            # handle flat accel/gyro passed directly
            if "data" not in data and ("accel" in data or "gyro" in data):
                data["data"] = {
                    "accel": data.get("accel", {}),
                    "gyro": data.get("gyro", {})
                }
        return data

# edge inference packet
class InferencePacket(BaseModel):
    id: int
    reason: int
    r_m_id: int
    ae_id: int
    mse: float
    anom: bool

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        # normalize aliases for id and model ids
        if isinstance(data, dict):
            if "id" not in data and "start" in data:
                data["id"] = data["start"]
            if "r_m_id" not in data and "router_model_id" in data:
                data["r_m_id"] = data["router_model_id"]
            if "ae_id" not in data and "autoencoder_model_id" in data:
                data["ae_id"] = data["autoencoder_model_id"]
        return data

# node capabilities
class NodeCapabilities(BaseModel):
    accel_freqs: List[float] = Field(default_factory=list)
    gyro_freqs: List[float] = Field(default_factory=list)
    enabled_ops: List[str] = Field(default_factory=list)

# node health telemetry
class NodeHealthInfo(BaseModel):
    ram: int
    sd: int
    cache: List[int] = Field(default_factory=list)
    last: int = 0
    status: Optional[str] = "idle"
    caps: Optional[NodeCapabilities] = None
    dump_t: Optional[int] = None
    dump_r: Optional[bool] = None
    ips: Optional[float] = 0.0

# Alert packet
class NodeAlert(BaseModel):
    code: int
    detail: str

# downlink command
class Command(BaseModel):
    dump: bool = False
    dump_i: bool = False
    reboot: bool = False

# runtime configuration
class RuntimeConfig(BaseModel):
    rate: float
    mode: int
    batch: int
    cad: Optional[int] = None
    beat: int
    sd_en: bool

# routing entry
class RouteEntry(BaseModel):
    out_ix: int
    m_id: int

    @model_validator(mode="before")
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        # support out_idx alias
        if isinstance(data, dict):
            if "out_ix" not in data and "out_idx" in data:
                data["out_ix"] = data["out_idx"]
        return data

# ensemble configuration
class EnsembleConfig(BaseModel):
    warmup: int
    r_m_id: int
    routes: List[RouteEntry] = Field(default_factory=list)
    mem_id: Optional[int] = None

# base model package
class BaseModelPackage(BaseModel):
    m_id: int
    tag: Optional[int] = None
    data: bytes

# memory model package
class MemoryModelPackage(BaseModelPackage):
    type: int = int(ModelType.MEMORY)
    accel_bins: int
    gyro_bins: int
    outdim: int
    state: int

# autoencoder model package
class AutoencoderModelPackage(BaseModelPackage):
    type: int = int(ModelType.AUTOENCODER)
    accel_bins: int
    gyro_bins: int
    tsteps: int
    mem_d: Optional[int] = None
    limit: float
    loss: int
    skip: Optional[int] = None

# router model package
class RouterModelPackage(BaseModelPackage):
    type: int = int(ModelType.ROUTER)
    accel_bins: int
    gyro_bins: int
    tsteps: int
    mem_d: Optional[int] = None
    class_count: int = Field(alias="class")

    model_config = {
        "populate_by_name": True
    }

# Union types for generic parsing
ModelPackageUnion = Union[AutoencoderModelPackage, RouterModelPackage, MemoryModelPackage]
PayloadUnion = Union[AutoencoderModelPackage, RouterModelPackage, MemoryModelPackage, EnsembleConfig]
