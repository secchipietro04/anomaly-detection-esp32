# pydantic v2 schemas matching cddl definitions
from enum import IntEnum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

class StreamMode(IntEnum):
    # stream modes defined in telemetry.cddl
    CONTINUOUS = 1
    ANOMALY_ONLY = 2
    ANOMALY_OR_PERIODIC = 3
    CONTINUOUS_SCORE_RAW_ANOMALY = 4

class EmitReason(IntEnum):
    # telemetry emission reasons
    CONTINUOUS = 1
    CADENCE = 2
    ANOMALY = 3
    ANOMALY_AFTERMATH = 4
    BURST = 5
    DUMP = 6

class ModelType(IntEnum):
    # model types from model.cddl
    AUTOENCODER = 1
    ROUTER = 2
    MEMORY = 3

class ArchitectureTag(IntEnum):
    # architecture tags
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

class LossMode(IntEnum):
    # reconstruction loss mode
    LOG_MSE = 1
    LINEAR_MSE = 2

class RuntimeConfig(BaseModel):
    # runtime configuration sent to sensor
    model_config = ConfigDict(extra="ignore")

    mode: int = Field(ge=1, le=4)
    rate: float = Field(gt=0)
    beat: int = Field(gt=0)
    batch: int = Field(gt=0)
    sd_en: bool = False
    cad: Optional[int] = Field(default=None, gt=0)

class NodeCapabilities(BaseModel):
    # sensor capabilities payload
    model_config = ConfigDict(extra="ignore")

    accel_freqs: List[float] = Field(default_factory=list)
    gyro_freqs: List[float] = Field(default_factory=list)
    enabled_ops: List[str] = Field(default_factory=list)

class NodeHealthInfo(BaseModel):
    # periodic health heartbeat
    model_config = ConfigDict(extra="ignore")

    ram: int = 0
    sd_ok: bool = False
    cache: List[int] = Field(default_factory=list)
    seg_id: int = 0
    stat: str = "idle"
    ips: float = 0.0

class Segment(BaseModel):
    # raw telemetry segment payload
    model_config = ConfigDict(extra="ignore")

    id: int
    chunk: int = 1
    rate: float
    reason: int
    ax: List[float] = Field(default_factory=list)
    ay: List[float] = Field(default_factory=list)
    az: List[float] = Field(default_factory=list)
    gx: List[float] = Field(default_factory=list)
    gy: List[float] = Field(default_factory=list)
    gz: List[float] = Field(default_factory=list)

class InferencePacket(BaseModel):
    # inference packet from edge
    model_config = ConfigDict(extra="ignore")

    seg_id: int
    reason: int
    r_m_id: Optional[int] = None
    ae_m_id: Optional[int] = None
    mse: float
    anomaly: bool

class BaseModelPackage(BaseModel):
    # base model package
    model_config = ConfigDict(extra="ignore")

    m_id: int
    tag: Optional[int] = None
    data: bytes

class AutoencoderModelPackage(BaseModelPackage):
    # autoencoder package definition
    type: int = int(ModelType.AUTOENCODER)
    accel_bins: int
    gyro_bins: int
    tsteps: int
    mem_d: Optional[int] = None
    limit: float
    loss: int = int(LossMode.LOG_MSE)
    skip: Optional[int] = None

class RouterModelPackage(BaseModelPackage):
    # router model package
    type: int = int(ModelType.ROUTER)
    accel_bins: int
    gyro_bins: int
    tsteps: int
    mem_d: Optional[int] = None
    class_count: int = Field(alias="class")

class MemoryModelPackage(BaseModelPackage):
    # recurrent memory model package
    type: int = int(ModelType.MEMORY)
    accel_bins: int
    gyro_bins: int
    outdim: int
    state: int

class RouteEntry(BaseModel):
    # single routing entry
    model_config = ConfigDict(extra="ignore")

    out_ix: int
    m_id: int

class EnsembleConfig(BaseModel):
    # lightweight ensemble routing table payload
    model_config = ConfigDict(extra="ignore")

    warmup: int = 0
    r_m_id: Optional[int] = None
    mem_id: Optional[int] = None
    routes: List[RouteEntry] = Field(default_factory=list)
