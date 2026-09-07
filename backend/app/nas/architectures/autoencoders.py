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
            "latent_dim": trial.suggest_categorical(f"{p}da_latent_dim", [8, 16, 32, 64]),
            "num_layers": trial.suggest_int(f"{p}da_num_layers", 1, 3),
            "loss_mode": 1,
            "limit": trial.suggest_float(f"{p}da_limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}da_skip", [1, 2, 4]),
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
            "latent_dim": trial.suggest_categorical(f"{p}va_latent_dim", [8, 16, 32]),
            "num_layers": trial.suggest_int(f"{p}va_num_layers", 1, 2),
            "loss_mode": 1,
            "limit": trial.suggest_float(f"{p}va_limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}va_skip", [1, 2, 4]),
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
    required_ops: Set[str] = {"CONV_2D", "CONV_2D_TRANSPOSE", "RELU", "SUB", "SQUARE", "MEAN"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "filters": trial.suggest_categorical(f"{p}ca1d_filters", [16, 32, 64]),
            "latent_dim": trial.suggest_categorical(f"{p}ca1d_latent_dim", [16, 32]),
            "num_layers": trial.suggest_int(f"{p}ca1d_num_layers", 1, 2),
            "loss_mode": 1,
            "limit": trial.suggest_float(f"{p}ca1d_limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}ca1d_skip", [1, 2, 4]),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        filters = params.get("filters", 32)
        latent = params.get("latent_dim", 16)
        layers = params.get("num_layers", 2)
        param_count = (filters * 3 * 2 + filters * filters * 3 + filters * latent + latent * filters) * layers
        return 1024 + param_count * 4

class Conv2DAutoencoder(BaseArchitecture):
    # 2d spatial-temporal convolutional autoencoder
    tag = ArchitectureTag.CA_2D
    model_type = ModelType.AUTOENCODER
    required_ops: Set[str] = {"CONV_2D", "CONV_2D_TRANSPOSE", "RELU", "SUB", "SQUARE", "MEAN"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "filters": trial.suggest_categorical(f"{p}ca2d_filters", [16, 32, 64]),
            "latent_dim": trial.suggest_categorical(f"{p}ca2d_latent_dim", [16, 32]),
            "num_layers": trial.suggest_int(f"{p}ca2d_num_layers", 1, 2),
            "loss_mode": 1,
            "limit": trial.suggest_float(f"{p}ca2d_limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}ca2d_skip", [1, 2, 4]),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        filters = params.get("filters", 32)
        latent = params.get("latent_dim", 16)
        layers = params.get("num_layers", 2)
        param_count = (filters * 3 * 3 * 2 + filters * filters * 9 + filters * latent + latent * filters) * layers
        return 1024 + param_count * 4

class ConvLSTMAutoencoder(BaseArchitecture):
    # convolutional autoencoder coupled with recurrent lstm cells
    tag = ArchitectureTag.CLSTM
    model_type = ModelType.AUTOENCODER
    required_ops: Set[str] = {"CONV_2D", "UNIDIRECTIONAL_SEQUENCE_LSTM", "FULLY_CONNECTED", "RELU", "TANH", "SUB", "SQUARE", "MEAN"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "filters": trial.suggest_categorical(f"{p}clstm_filters", [16, 32]),
            "hidden_dim": trial.suggest_categorical(f"{p}clstm_hidden_dim", [16, 32]),
            "latent_dim": trial.suggest_categorical(f"{p}clstm_latent_dim", [16, 32]),
            "loss_mode": 1,
            "limit": trial.suggest_float(f"{p}clstm_limit", 0.05, 0.5),
            "skip": trial.suggest_categorical(f"{p}clstm_skip", [1, 2, 4]),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        filters = params.get("filters", 16)
        hidden = params.get("hidden_dim", 16)
        latent = params.get("latent_dim", 16)
        param_count = filters * 9 + (hidden * hidden * 4 + hidden * latent) + latent * filters
        return 2048 + param_count * 4

ALL_AUTOENCODER_ARCHS = [
    DenseAutoencoder(),
    VariationalAutoencoder(),
    Conv1DAutoencoder(),
    Conv2DAutoencoder(),
    ConvLSTMAutoencoder(),
]
