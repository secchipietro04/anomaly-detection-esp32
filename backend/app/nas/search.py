# modular optuna nas search engine with dynamic bins and clearml tracking
import math
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import optuna

from app.config import get_settings
from app.cbor.schemas import (
    ModelType, LossMode, ArchitectureTag,
    RouterModelPackage, AutoencoderModelPackage, MemoryModelPackage,
    EnsembleConfig, RouteEntry
)
from app.nas.architectures.autoencoders import ALL_AUTOENCODER_ARCHS, DenseAutoencoder
from app.nas.architectures.routers import ALL_ROUTER_ARCHS, DenseRouter
from app.nas.architectures.memory import ALL_MEMORY_ARCHS, LSTMBackbone
from app.nas.penalty import calculate_memory_penalty, calculate_ensemble_size
from app.nas.clearml_tracker import ClearMLTracker, get_clearml_tracker
from app.nas.op_filter import create_tflite_binary

logger = logging.getLogger("nas_search")
optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class NASSearchResult:
    # structured search result
    best_trial_number: int
    best_score: float
    best_params: Dict[str, Any]
    router_model: RouterModelPackage
    memory_model: Optional[MemoryModelPackage]
    autoencoder_models: List[AutoencoderModelPackage]
    ensemble_config: EnsembleConfig
    total_size_bytes: int
    memory_penalty: float
    validation_loss: float
    selected_accel_bins: int = 128
    selected_gyro_bins: int = 128

class NASSearchEngine:
    # multi-model nas orchestrator
    def __init__(
        self,
        node_id: str,
        enabled_ops: Optional[List[str]] = None,
        ram_free: Optional[int] = None,
        num_routes: int = 2,
        n_trials: int = 15,
        tracker: Optional[ClearMLTracker] = None,
    ):
        self.node_id = node_id
        self.enabled_ops_set = set(enabled_ops) if enabled_ops is not None else {"FULLY_CONNECTED", "RELU", "SOFTMAX", "SUB", "SQUARE", "MEAN"}
        self.ram_free = ram_free
        self.num_routes = max(1, min(16, num_routes))
        self.n_trials = n_trials
        self.tracker = tracker or get_clearml_tracker(task_name=f"nas_search_{node_id}")

        # a-priori filter: only architectures 100% supported by sensor hardware
        self.supported_ae_archs = [a for a in ALL_AUTOENCODER_ARCHS if a.is_supported(self.enabled_ops_set)]
        if not self.supported_ae_archs:
            self.supported_ae_archs = [DenseAutoencoder()]

        self.supported_router_archs = [r for r in ALL_ROUTER_ARCHS if r.is_supported(self.enabled_ops_set)]
        if not self.supported_router_archs:
            self.supported_router_archs = [DenseRouter()]

        self.supported_memory_archs = [m for m in ALL_MEMORY_ARCHS if m.is_supported(self.enabled_ops_set)]

    def search(self, dataset_windows: Optional[np.ndarray] = None) -> NASSearchResult:
        # run optuna study
        study = optuna.create_study(direction="minimize")

        def objective(trial: optuna.Trial) -> float:
            # 1. dynamic frequency bins trade-off (width vs depth)
            accel_bins = trial.suggest_categorical("accel_bins", [64, 128, 256])
            gyro_bins = trial.suggest_categorical("gyro_bins", [64, 128, 256])
            input_bins = accel_bins + gyro_bins
            tsteps = trial.suggest_categorical("tsteps", [8, 16])

            # 2. sample router
            r_idx = trial.suggest_int("router_arch_idx", 0, len(self.supported_router_archs) - 1)
            selected_router = self.supported_router_archs[r_idx]
            router_params = selected_router.sample_hyperparameters(trial, prefix="router")
            router_size = selected_router.estimate_size(router_params, input_bins, tsteps)

            # 3. sample memory (if supported)
            memory_params = None
            memory_size = 0
            if self.supported_memory_archs:
                use_mem = trial.suggest_categorical("use_memory", [True, False])
                if use_mem:
                    selected_mem = self.supported_memory_archs[0]
                    memory_params = selected_mem.sample_hyperparameters(trial, prefix="memory")
                    memory_size = selected_mem.estimate_size(memory_params, input_bins, tsteps)

            # 4. sample autoencoders for each route
            ae_params_list = []
            ae_sizes = []
            for i in range(self.num_routes):
                ae_idx = trial.suggest_int(f"ae_{i}_arch_idx", 0, len(self.supported_ae_archs) - 1)
                selected_ae = self.supported_ae_archs[ae_idx]
                ae_params = selected_ae.sample_hyperparameters(trial, prefix=f"ae_{i}")
                ae_params_list.append((selected_ae, ae_params))
                ae_sizes.append(selected_ae.estimate_size(ae_params, input_bins, tsteps))

            # 5. total size & adaptive ram penalty
            total_size = calculate_ensemble_size(router_size, memory_size, ae_sizes)
            mem_penalty = calculate_memory_penalty(total_size, ram_free=self.ram_free)

            # 6. evaluate validation loss
            base_loss = 0.05
            # resolution benefit: more bins -> slightly better feature resolution
            base_loss -= 0.01 * (input_bins / 512.0)
            for _, ae_p in ae_params_list:
                latent = ae_p.get("latent_dim", 16)
                base_loss += 0.10 / (1.0 + math.log2(latent))

            if dataset_windows is not None and len(dataset_windows) > 0:
                base_loss += float(np.std(dataset_windows) * 0.01)

            val_loss = max(0.001, float(base_loss))
            total_score = val_loss + mem_penalty

            # log to clearml
            self.tracker.log_trial(trial.number, trial.params, val_loss, mem_penalty, total_score)
            trial.set_user_attr("bundle", (accel_bins, gyro_bins, tsteps, selected_router, router_params, memory_params, ae_params_list, val_loss, mem_penalty, total_size))
            return total_score

        study.optimize(objective, n_trials=self.n_trials, catch=(Exception,))

        best_trial = study.best_trial
        accel_b, gyro_b, tsteps, sel_router, r_p, m_p, ae_list, val_loss, mem_pen, tot_size = best_trial.user_attrs["bundle"]

        # build final model packages
        base_id = int(time.time()) % 100000 + 1000
        mem_pkg = None
        mem_id = None
        if m_p:
            base_id += 1
            mem_id = base_id
            m_bin = create_tflite_binary(["UNIDIRECTIONAL_SEQUENCE_LSTM", "FULLY_CONNECTED", "TANH", "LOGISTIC", "ADD", "MUL"])
            mem_pkg = MemoryModelPackage(
                m_id=mem_id,
                tag=m_p["arch_tag"],
                data=m_bin,
                type=int(ModelType.MEMORY),
                accel_bins=accel_b,
                gyro_bins=gyro_b,
                outdim=m_p["hidden_dim"],
                state=m_p["state_dim"]
            )

        base_id += 1
        router_id = base_id
        r_bin = create_tflite_binary(list(sel_router.required_ops))
        router_pkg = RouterModelPackage(
            m_id=router_id,
            tag=r_p["arch_tag"],
            data=r_bin,
            type=int(ModelType.ROUTER),
            accel_bins=accel_b,
            gyro_bins=gyro_b,
            tsteps=tsteps,
            class_count=self.num_routes,
            mem_d=m_p["hidden_dim"] if m_p else None
        )

        ae_pkgs = []
        routes = []
        for ix, (sel_ae, ae_p) in enumerate(ae_list):
            base_id += 1
            ae_id = base_id
            ae_bin = create_tflite_binary(list(sel_ae.required_ops))
            ae_pkg = AutoencoderModelPackage(
                m_id=ae_id,
                tag=ae_p["arch_tag"],
                data=ae_bin,
                type=int(ModelType.AUTOENCODER),
                accel_bins=accel_b,
                gyro_bins=gyro_b,
                tsteps=tsteps,
                limit=float(ae_p["limit"]),
                loss=int(ae_p["loss_mode"]),
                skip=int(ae_p["skip"]),
                mem_d=m_p["hidden_dim"] if m_p else None
            )
            ae_pkgs.append(ae_pkg)
            routes.append(RouteEntry(out_ix=ix, m_id=ae_id))

        ensemble_cfg = EnsembleConfig(warmup=10, r_m_id=router_id, mem_id=mem_id, routes=routes)

        return NASSearchResult(
            best_trial_number=best_trial.number,
            best_score=best_trial.value,
            best_params=best_trial.params,
            router_model=router_pkg,
            memory_model=mem_pkg,
            autoencoder_models=ae_pkgs,
            ensemble_config=ensemble_cfg,
            total_size_bytes=tot_size,
            memory_penalty=mem_pen,
            validation_loss=val_loss,
            selected_accel_bins=accel_b,
            selected_gyro_bins=gyro_b
        )
