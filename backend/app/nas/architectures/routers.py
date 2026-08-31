# modular router architectures
from typing import Set, Dict, Any
import optuna
from app.cbor.schemas import ModelType, ArchitectureTag
from app.nas.architectures.base import BaseArchitecture

class DenseRouter(BaseArchitecture):
    # dense multi-class routing network
    tag = ArchitectureTag.DENSE_R
    model_type = ModelType.ROUTER
    required_ops: Set[str] = {"FULLY_CONNECTED", "RELU", "SOFTMAX"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "router") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "hidden_units": trial.suggest_categorical(f"{p}hidden_units", [16, 32, 64, 128]),
            "num_layers": trial.suggest_int(f"{p}num_layers", 1, 3),
            "learning_rate": trial.suggest_float(f"{p}lr", 1e-4, 1e-2, log=True),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        in_dim = input_bins * tsteps
        hidden = params.get("hidden_units", 32)
        layers = params.get("num_layers", 1)
        classes = 2
        param_count = in_dim * hidden + (layers - 1) * (hidden * hidden) + hidden * classes
        return 1024 + param_count * 4

class Conv1DRouter(BaseArchitecture):
    # 1d cnn routing network
    tag = ArchitectureTag.ONE_D_CNN_R
    model_type = ModelType.ROUTER
    required_ops: Set[str] = {"CONV_2D", "MAX_POOL_2D", "FULLY_CONNECTED", "RELU", "SOFTMAX", "RESHAPE"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "router") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "filters": trial.suggest_categorical(f"{p}filters", [8, 16, 32]),
            "hidden_units": trial.suggest_categorical(f"{p}hidden_units", [16, 32, 64]),
            "learning_rate": trial.suggest_float(f"{p}lr", 1e-4, 1e-2, log=True),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        in_dim = input_bins * tsteps
        filters = params.get("filters", 16)
        hidden = params.get("hidden_units", 32)
        classes = 2
        param_count = filters * 3 * (in_dim // tsteps) + hidden * classes
        return 1024 + param_count * 4

class STFTMCNNRouter(BaseArchitecture):
    # multi-branch cnn router with parallel convolution branches
    tag = ArchitectureTag.STFT_MCNN
    model_type = ModelType.ROUTER
    required_ops: Set[str] = {"CONV_2D", "FULLY_CONNECTED", "RELU", "SOFTMAX", "MAX_POOL_2D"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "router") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "branch_filters": trial.suggest_categorical(f"{p}branch_filters", [8, 16, 32]),
            "hidden_units": trial.suggest_categorical(f"{p}hidden_units", [16, 32, 64]),
            "learning_rate": trial.suggest_float(f"{p}lr", 1e-4, 1e-2, log=True),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        in_dim = input_bins * tsteps
        filters = params.get("branch_filters", 16)
        hidden = params.get("hidden_units", 32)
        classes = 2
        # 3 parallel branches with kernels 3, 5, 7
        param_count = (filters * (3 + 5 + 7) + hidden * classes) * 2
        return 1024 + param_count * 4

ALL_ROUTER_ARCHS = [DenseRouter(), Conv1DRouter(), STFTMCNNRouter()]
