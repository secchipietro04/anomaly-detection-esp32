# Auto-generated from cbor_schema/*.cddl via cddl2py. DO NOT EDIT DIRECTLY.
from __future__ import annotations
from typing import Any, Literal, Optional, Union, List, Dict
from pydantic import BaseModel, Field, ConfigDict

# --- device.cddl ---



StreamModeContinuous = Literal[1]

StreamModeAnomalyOnly = Literal[2]

StreamModeAnomalyOrPeriodic = Literal[3]

StreamModeContinuousScoreRawAnomaly = Literal[4]

StreamMode = Union[StreamModeContinuous, StreamModeAnomalyOnly, StreamModeAnomalyOrPeriodic, StreamModeContinuousScoreRawAnomaly]

class RuntimeConfig(BaseModel):
    rate: float  # sample rate
    mode: StreamMode  # stream mode
    batch: int  # batch size (MQTT points)
    cad: Optional[int] = None  # mixed cadence (N reads)
    beat: int  # heartbeat (s)
    sd_en: bool  # enable SD


# --- telemetry.cddl ---



EmitReasonContinuous = Literal[1]

EmitReasonAnomaly = Literal[2]

EmitReasonPeriodic = Literal[3]

EmitReasonManualDump = Literal[4]

EmitReason = Union[EmitReasonContinuous, EmitReasonAnomaly, EmitReasonPeriodic, EmitReasonManualDump]

class Datapoints(BaseModel):
    x: list[float]
    y: list[float]
    z: list[float]

class Segment(BaseModel):
    id: int  # segment_id
    rate: float  # sample rate in Hz
    reason: EmitReason  # Why this batch was emitted
    chunk: int  # subsegment index
    data: dict[str, Any]

class InferencePacket(BaseModel):
    id: int  # segment_id
    reason: EmitReason  # Why this packet was emitted
    r_m_id: int  # router model ID
    ae_id: int  # ae_m_id: ae model ID
    mse: float  # mse_loss: Reconstruction loss
    anom: bool  # is_anomaly
    time: Optional[int] = None  # inference execution time in ms (optional)

class NodeCapabilities(BaseModel):
    accel_freqs: list[float]
    gyro_freqs: list[float]
    enabled_ops: list[str]

class NodeHealthInfo(BaseModel):
    ram: int  # free_ram
    sd: int  # free_sd
    cache: list[int]  # cached_models: m_ids stored locally
    last: int  # last_point_id: last read point id
    status: Optional[str] = None  # cmd_status: Execution status ("idle", "streaming_dump")
    caps: Optional[NodeCapabilities] = None  # model capabilities
    dump_t: Optional[int] = None  # time took in ms
    dump_r: Optional[bool] = None  # is still reading SD card
    ips: Optional[float] = None  # inferences per second

class NodeAlert(BaseModel):
    code: int  # Error code
    detail: str  # Description

class Command(BaseModel):
    dump: bool  # dump_all: Trigger a full data dump to the cloud
    dump_i: bool  # dump_all_inf: Trigger info dump + inference
    reboot: bool  # Trigger a device reboot


# --- model.cddl ---



ModelTypeAutoencoder = Literal[1]

ModelTypeRouter = Literal[2]

ModelTypeMemory = Literal[3]

ModelType = Union[ModelTypeAutoencoder, ModelTypeRouter, ModelTypeMemory]

ArchTagVa = Literal[1]

ArchTagCa1D = Literal[2]

ArchTagCa2D = Literal[3]

ArchTagClstm = Literal[4]

ArchTagDa = Literal[5]

ArchTagDenseR = Literal[10]

ArchTagStftMcnn = Literal[12]

ArchTag1dCnnR = Literal[13]

ArchTagLstmMem = Literal[20]

ArchitectureTag = Union[ArchTagVa, ArchTagCa1D, ArchTagCa2D, ArchTagClstm, ArchTagDa, ArchTagDenseR, ArchTagStftMcnn, ArchTag1dCnnR, ArchTagLstmMem, int]

LossModeLogMse = Literal[1]

LossModeLinearMse = Literal[2]

LossMode = Union[LossModeLogMse, LossModeLinearMse]

class BaseModelPackage(BaseModel):
    m_id: int  # Unique Model ID
    tag: Optional[ArchitectureTag] = None  # arch type tag
    data: bytes  # Raw .tflite binary bytes

class MemoryModelPackage(BaseModelPackage):
    type: ModelTypeMemory
    accel_bins: int  # input bins for accelerometer
    gyro_bins: int  # input bins for gyroscope
    outdim: int  # output dimension
    state: int  # state dimension size

class AutoencoderModelPackage(BaseModelPackage):
    type: ModelTypeAutoencoder
    accel_bins: int  # input bins for accelerometer
    gyro_bins: int  # input bins for gyroscope
    tsteps: int  # input temporal depth
    mem_d: Optional[int] = None  # memory context size consumed
    limit: float  # anomaly threshold
    loss: LossMode  # loss mode
    skip: Optional[int] = None  # evaluation window skip amount

class RouterModelPackage(BaseModelPackage):
    type: ModelTypeRouter
    accel_bins: int  # input bins for accelerometer
    gyro_bins: int  # input bins for gyroscope
    tsteps: int  # input temporal depth
    mem_d: Optional[int] = None  # memory context size consumed
    class_count: int = Field(alias="class")  # number of output classes

ModelPackage = Union[MemoryModelPackage, AutoencoderModelPackage, RouterModelPackage]


# --- ensemble.cddl ---



class RouteEntry(BaseModel):
    out_ix: int  # Router output class idx
    m_id: int  # Target submodel

class EnsembleConfig(BaseModel):
    warmup: int  # Steps to warm up before alerting
    mem_id: Optional[int] = None  # Memory Model ID
    r_m_id: int  # router ID
    routes: list[RouteEntry]  # Mapping table

Payload = Union["ModelPackage", EnsembleConfig]


# --- Typed IntEnums for Python ---

from enum import IntEnum

class StreamMode(IntEnum):
    CONTINUOUS = 1
    ANOMALY_ONLY = 2
    ANOMALY_OR_PERIODIC = 3
    CONTINUOUS_SCORE_RAW_ANOMALY = 4

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
    DENSE_R = 10
    STFT_MCNN = 12
    ONE_D_CNN_R = 13
    LSTM_MEM = 20

class LossMode(IntEnum):
    LOG_MSE = 1
    LINEAR_MSE = 2

