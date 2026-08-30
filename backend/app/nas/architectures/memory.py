# modular recurrent memory architecture
from typing import Set, Dict, Any
import optuna
from app.cbor.schemas import ModelType, ArchitectureTag
from app.nas.architectures.base import BaseArchitecture

class LSTMBackbone(BaseArchitecture):
    # unidirectional sequence lstm recurrent backbone
    tag = ArchitectureTag.LSTM_MEM
    model_type = ModelType.MEMORY
    required_ops: Set[str] = {"UNIDIRECTIONAL_SEQUENCE_LSTM", "FULLY_CONNECTED", "TANH", "LOGISTIC", "ADD", "MUL"}

    def sample_hyperparameters(self, trial: optuna.Trial, prefix: str = "memory") -> Dict[str, Any]:
        p = f"{prefix}_" if prefix else ""
        return {
            "arch_tag": int(self.tag),
            "hidden_dim": trial.suggest_categorical(f"{p}hidden_dim", [16, 32, 64, 128]),
            "state_dim": trial.suggest_categorical(f"{p}state_dim", [16, 32, 64]),
            "num_layers": trial.suggest_int(f"{p}num_layers", 1, 2),
            "learning_rate": trial.suggest_float(f"{p}lr", 1e-4, 1e-2, log=True),
        }

    def estimate_size(self, params: Dict[str, Any], input_bins: int, tsteps: int) -> int:
        hidden = params.get("hidden_dim", 32)
        layers = params.get("num_layers", 1)
        param_count = layers * 4 * (input_bins * hidden + hidden * hidden + hidden)
        return 1024 + param_count * 4

ALL_MEMORY_ARCHS = [LSTMBackbone()]
