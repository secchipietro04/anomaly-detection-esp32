# pydantic v2 schemas backed by generated_schemas.py (compiled from cbor_schema/*.cddl)
from enum import IntEnum
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, ConfigDict

# import models compiled from CDDL
from app.cbor.generated_schemas import (
    Datapoints,
    Segment as CDDLSegment,
    InferencePacket,
    NodeCapabilities,
    NodeHealthInfo,
    NodeAlert,
    Command,
    BaseModelPackage,
    MemoryModelPackage,
    AutoencoderModelPackage,
    RouterModelPackage,
    ModelPackage,
    EmitReason,
    ModelType as CDDLModelType,
    ArchitectureTag as CDDLArchitectureTag,
    LossMode as CDDLLossMode,
)

# int enums for type safety in backend
class StreamMode(IntEnum):
    CONTINUOUS = 1
    ANOMALY_ONLY = 2
    ANOMALY_OR_PERIODIC = 3
    CONTINUOUS_SCORE_RAW_ANOMALY = 4

class EmitReasonEnum(IntEnum):
    CONTINUOUS = 1
    ANOMALY = 2
    PERIODIC = 3
    MANUAL_DUMP = 4

class ModelType(IntEnum):
    AUTOENCODER = 1
    ROUTER = 2
    MEMORY = 3

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

class LossMode(IntEnum):
    LOG_MSE = 1
    LINEAR_MSE = 2

# runtime configuration (downlink)
class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: int = Field(ge=1, le=4)
    rate: float = Field(gt=0)
    beat: int = Field(gt=0)
    batch: int = Field(gt=0)
    sd_en: bool = False
    cad: Optional[int] = Field(default=None, gt=0)

# segment schema helper with flattened acceleration/gyro convenience fields
class Segment(CDDLSegment):
    model_config = ConfigDict(extra="ignore")

    ax: List[float] = Field(default_factory=list)
    ay: List[float] = Field(default_factory=list)
    az: List[float] = Field(default_factory=list)
    gx: List[float] = Field(default_factory=list)
    gy: List[float] = Field(default_factory=list)
    gz: List[float] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        # extract flat arrays from nested data dict if present
        if self.data and isinstance(self.data, dict):
            accel = self.data.get("accel", {})
            gyro = self.data.get("gyro", {})
            if isinstance(accel, dict):
                self.ax = self.ax or accel.get("x", [])
                self.ay = self.ay or accel.get("y", [])
                self.az = self.az or accel.get("z", [])
            if isinstance(gyro, dict):
                self.gx = self.gx or gyro.get("x", [])
                self.gy = self.gy or gyro.get("y", [])
                self.gz = self.gz or gyro.get("z", [])

# ensemble routing config (downlink)
class RouteEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    out_ix: int
    m_id: int

class EnsembleConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    warmup: int = 0
    r_m_id: Optional[int] = None
    mem_id: Optional[int] = None
    routes: List[RouteEntry] = Field(default_factory=list)
