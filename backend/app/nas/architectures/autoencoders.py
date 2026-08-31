# modular autoencoder architectures
from typing import Set, Dict, Any, List
import optuna
from app.cbor.schemas import ModelType, ArchitectureTag, LossMode
from app.nas.architectures.base import BaseArchitecture

class DenseAutoencoder(BaseArchitecture):
    # dense multi-layer autoencoder
    tag = ArchitectureTag.DA
    model_type = ModelType.AUTOENCODER
    required_ops: Set[str] = {"FULLY_CONNECTED", "RELU", "SUB", "SQUARE", "MEAN"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "latent_dim": trial.suggest_categorical(f"{p}latent_dim", [8, 16, 32, 64]),
            "num_layers": trial.suggest_int(f"{p}num_layers", 1, 3),
            "loss_mode": trial.suggest_categorical(f"{p}loss_mode", [int(LossMode.LOG_MSE), int(LossMode.LINEAR_MSE)]),
            "limit": trial.suggest_float(f"{p}limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}skip", [1, 2, 4]),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        in_dim = input_bins * tsteps
        latent = params.get("latent_dim", 16)
        layers = params.get("num_layers", 2)
        h1 = latent * 4
        param_count = (in_dim * h1 + h1 * latent + latent * h1 + h1 * in_dim) * layers
        return 1024 + param_count * 4

class VariationalAutoencoder(BaseArchitecture):
    # variational autoencoder with reparameterization
    tag = ArchitectureTag.VA
    model_type = ModelType.AUTOENCODER
    required_ops: Set[str] = {"FULLY_CONNECTED", "RELU", "EXP", "MUL", "ADD", "SUB", "SQUARE", "MEAN"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "latent_dim": trial.suggest_categorical(f"{p}latent_dim", [8, 16, 32]),
            "num_layers": trial.suggest_int(f"{p}num_layers", 1, 2),
            "loss_mode": trial.suggest_categorical(f"{p}loss_mode", [int(LossMode.LOG_MSE), int(LossMode.LINEAR_MSE)]),
            "limit": trial.suggest_float(f"{p}limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}skip", [1, 2, 4]),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        in_dim = input_bins * tsteps
        latent = params.get("latent_dim", 16)
        layers = params.get("num_layers", 2)
        h1 = latent * 4
        param_count = (in_dim * h1 + h1 * (latent * 2) + latent * h1 + h1 * in_dim) * layers
        return 1024 + param_count * 4

class Conv1DAutoencoder(BaseArchitecture):
    # 1d convolutional autoencoder
    tag = ArchitectureTag.CA_1D
    model_type = ModelType.AUTOENCODER
    required_ops: Set[str] = {"CONV_2D", "TRANSPOSE_CONV", "MAX_POOL_2D", "RESHAPE", "RELU", "SUB", "SQUARE", "MEAN"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "filters": trial.suggest_categorical(f"{p}filters", [16, 32, 64]),
            "latent_dim": trial.suggest_categorical(f"{p}latent_dim", [16, 32]),
            "num_layers": trial.suggest_int(f"{p}num_layers", 1, 2),
            "loss_mode": trial.suggest_categorical(f"{p}loss_mode", [int(LossMode.LOG_MSE), int(LossMode.LINEAR_MSE)]),
            "limit": trial.suggest_float(f"{p}limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}skip", [1, 2, 4]),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        in_dim = input_bins * tsteps
        filters = params.get("filters", 16)
        latent = params.get("latent_dim", 16)
        layers = params.get("num_layers", 2)
        param_count = (filters * 9 + latent * (in_dim // 4)) * layers * 2
        return 1024 + param_count * 4

class Conv2DAutoencoder(BaseArchitecture):
    # 2d convolutional autoencoder
    tag = ArchitectureTag.CA_2D
    model_type = ModelType.AUTOENCODER
    required_ops: Set[str] = {"CONV_2D", "TRANSPOSE_CONV", "MAX_POOL_2D", "RESHAPE", "RELU", "SUB", "SQUARE", "MEAN"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "filters": trial.suggest_categorical(f"{p}filters", [16, 32, 64]),
            "latent_dim": trial.suggest_categorical(f"{p}latent_dim", [16, 32]),
            "num_layers": trial.suggest_int(f"{p}num_layers", 1, 2),
            "loss_mode": trial.suggest_categorical(f"{p}loss_mode", [int(LossMode.LOG_MSE), int(LossMode.LINEAR_MSE)]),
            "limit": trial.suggest_float(f"{p}limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}skip", [1, 2, 4]),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        in_dim = input_bins * tsteps
        filters = params.get("filters", 16)
        latent = params.get("latent_dim", 16)
        layers = params.get("num_layers", 2)
        param_count = (filters * 9 + latent * (in_dim // 4)) * layers * 2
        return 1024 + param_count * 4

class ConvLSTMAutoencoder(BaseArchitecture):
    # spatio-temporal convolutional lstm autoencoder
    tag = ArchitectureTag.CLSTM
    model_type = ModelType.AUTOENCODER
    required_ops: Set[str] = {
        "CONV_2D", "TRANSPOSE_CONV", "UNIDIRECTIONAL_SEQUENCE_LSTM", "RESHAPE",
        "FULLY_CONNECTED", "RELU", "TANH", "LOGISTIC", "MUL", "ADD", "SUB", "SQUARE", "MEAN"
    }

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "filters": trial.suggest_categorical(f"{p}filters", [16, 32]),
            "hidden_dim": trial.suggest_categorical(f"{p}hidden_dim", [16, 32]),
            "latent_dim": trial.suggest_categorical(f"{p}latent_dim", [16, 32]),
            "loss_mode": trial.suggest_categorical(f"{p}loss_mode", [int(LossMode.LOG_MSE), int(LossMode.LINEAR_MSE)]),
            "limit": trial.suggest_float(f"{p}limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}skip", [1, 2, 4]),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        in_dim = input_bins * tsteps
        filters = params.get("filters", 16)
        hidden = params.get("hidden_dim", 16)
        param_count = (filters * 9 + hidden * hidden * 4) * 2
        return 1024 + param_count * 4

ALL_AUTOENCODER_ARCHS = [
    DenseAutoencoder(),
    VariationalAutoencoder(),
    Conv1DAutoencoder(),
    Conv2DAutoencoder(),
    ConvLSTMAutoencoder()
]
