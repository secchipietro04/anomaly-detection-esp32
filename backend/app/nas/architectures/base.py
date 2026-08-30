# base class for modular model architectures in nas
from abc import ABC, abstractmethod
from typing import Set, Dict, Any, Optional
import optuna
from app.cbor.schemas import ModelType, ArchitectureTag

class BaseArchitecture(ABC):
    # base class defining operator requirements and hyperparameter space
    tag: ArchitectureTag
    model_type: ModelType
    required_ops: Set[str] = set()

    @classmethod
    def is_supported(cls, enabled_ops: Set[str]) -> bool:
        # check if all required operators are compiled in sensor firmware
        return cls.required_ops.issubset(enabled_ops)

    @abstractmethod
    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "") -> Dict[str, Any]:
        # sample model hyperparameters from optuna
        pass

    @abstractmethod
    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        # estimate model byte size
        pass
