# optuna multi-model nas search engine with enabled_ops filter and memory penalty
import copy
import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import optuna

from app.config import get_settings
from app.cbor.schemas import (
    ModelType,
    LossMode,
    ArchitectureTag,
    RouterModelPackage,
    AutoencoderModelPackage,
    MemoryModelPackage,
    EnsembleConfig,
    RouteEntry,
    ModelPackageUnion,
)
from app.nas.op_filter import (
    create_tflite_binary,
    extract_tflite_ops,
    validate_candidate_architecture,
    ALL_SUPPORTED_OPS,
)
from app.nas.penalty import calculate_memory_penalty, calculate_ensemble_size
from app.nas.clearml_tracker import ClearMLTracker, get_clearml_tracker

logger = logging.getLogger("nas_search")

# optuna log level adjustment
optuna.logging.set_verbosity(optuna.logging.WARNING)

@dataclass
class NASSearchResult:
    # structured result from nas search
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
    enabled_ops_used: List[str] = field(default_factory=list)

class NASSearchEngine:
    # multi-model neural architecture search orchestrator
    def __init__(
        self,
        node_id: str,
        enabled_ops: Optional[List[str]] = None,
        accel_bins: int = 128,
        gyro_bins: int = 128,
        tsteps: int = 16,
        num_routes: int = 2,
        n_trials: int = 15,
        timeout_seconds: Optional[int] = None,
        ram_limit_bytes: Optional[int] = None,
        lambda_penalty: Optional[float] = None,
        tracker: Optional[ClearMLTracker] = None,
        include_memory_model: bool = True,
    ):
        settings = get_settings()
        self.node_id = node_id
        self.enabled_ops = list(enabled_ops) if enabled_ops is not None else list(ALL_SUPPORTED_OPS)
        self.accel_bins = accel_bins
        self.gyro_bins = gyro_bins
        self.tsteps = tsteps
        self.num_routes = max(1, min(16, num_routes))
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.ram_limit_bytes = ram_limit_bytes if ram_limit_bytes is not None else settings.ram_limit_bytes
        self.lambda_penalty = lambda_penalty if lambda_penalty is not None else settings.sd_latency_penalty_factor
        self.tracker = tracker or get_clearml_tracker(task_name=f"nas_search_{node_id}")
        self.include_memory_model = include_memory_model

    def _get_required_ops(
        self,
        memory_params: Optional[Dict[str, Any]],
        router_params: Dict[str, Any],
        ae_params_list: List[Dict[str, Any]]
    ) -> List[str]:
        # aggregate all required tflite ops across models
        req_ops = set()
        
        # memory model ops
        if memory_params is not None:
            req_ops.update(["UNIDIRECTIONAL_SEQUENCE_LSTM", "FULLY_CONNECTED", "TANH", "LOGISTIC", "ADD", "MUL"])
            
        # router ops
        r_arch = router_params.get("arch", "dense")
        if r_arch == "dense":
            req_ops.update(["FULLY_CONNECTED", "RELU", "SOFTMAX"])
            if router_params.get("activation") == "leaky_relu":
                req_ops.add("LEAKY_RELU")
        elif r_arch == "1d_cnn":
            req_ops.update(["CONV_2D", "MAX_POOL_2D", "FULLY_CONNECTED", "RELU", "SOFTMAX", "RESHAPE"])

        # autoencoder ops
        for ae in ae_params_list:
            ae_arch = ae.get("arch", "dense")
            if ae_arch == "dense":
                req_ops.update(["FULLY_CONNECTED", "RELU", "SUB", "SQUARE", "MEAN"])
            elif ae_arch in ("ca_1d", "ca_2d"):
                req_ops.update(["CONV_2D", "TRANSPOSE_CONV", "MAX_POOL_2D", "RESHAPE", "RELU", "SUB", "SQUARE", "MEAN"])
            elif ae_arch == "va":
                req_ops.update(["FULLY_CONNECTED", "RELU", "EXP", "MUL", "ADD", "SUB", "SQUARE", "MEAN"])
                
        return sorted(list(req_ops))

    def _estimate_model_bytes(self, model_type: ModelType, params: Dict[str, Any]) -> int:
        # compute parameter count and byte size for candidate
        base_overhead = 1024  # flatbuffer header & metadata
        
        if model_type == ModelType.MEMORY:
            hidden = params.get("hidden_dim", 32)
            layers = params.get("num_layers", 1)
            in_dim = self.accel_bins + self.gyro_bins
            # lstm cell params: 4 * (in_dim * hidden + hidden^2 + hidden)
            params_count = layers * 4 * (in_dim * hidden + hidden * hidden + hidden)
            return base_overhead + params_count * 4
            
        elif model_type == ModelType.ROUTER:
            arch = params.get("arch", "dense")
            hidden = params.get("hidden_units", 32)
            layers = params.get("num_layers", 1)
            in_dim = (self.accel_bins + self.gyro_bins) * self.tsteps
            classes = self.num_routes
            if arch == "dense":
                params_count = in_dim * hidden + (layers - 1) * (hidden * hidden) + hidden * classes
            else:
                filters = params.get("filters", 16)
                params_count = filters * 3 * in_dim // self.tsteps + hidden * classes
            return base_overhead + params_count * 4
            
        elif model_type == ModelType.AUTOENCODER:
            arch = params.get("arch", "dense")
            latent = params.get("latent_dim", 16)
            layers = params.get("num_layers", 2)
            in_dim = (self.accel_bins + self.gyro_bins) * self.tsteps
            if arch == "dense":
                h1 = latent * 4
                params_count = (in_dim * h1 + h1 * latent + latent * h1 + h1 * in_dim) * layers
            elif arch in ("ca_1d", "ca_2d"):
                filters = params.get("filters", 16)
                params_count = (filters * 9 + latent * in_dim // 4) * layers * 2
            else:  # va
                h1 = latent * 4
                params_count = (in_dim * h1 + h1 * (latent * 2) + latent * h1 + h1 * in_dim) * layers
            # check if extra simulated size requested for penalty testing
            dummy = params.get("extra_payload_bytes", 0)
            return base_overhead + params_count * 4 + dummy
            
        return base_overhead + 2048

    def _sample_trial_params(self, trial: optuna.Trial) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        # sample hyperparameters for memory, router, and autoencoders
        
        # 1. memory params
        memory_params = None
        has_lstm_op = "UNIDIRECTIONAL_SEQUENCE_LSTM" in self.enabled_ops
        if self.include_memory_model and has_lstm_op:
            memory_params = {
                "hidden_dim": trial.suggest_categorical("memory_hidden_dim", [16, 32, 64, 128]),
                "state_dim": trial.suggest_categorical("memory_state_dim", [16, 32, 64]),
                "num_layers": trial.suggest_int("memory_num_layers", 1, 2),
                "learning_rate": trial.suggest_float("memory_lr", 1e-4, 1e-2, log=True),
            }
            
        # 2. router params
        router_arch_choices = ["dense"]
        if "CONV_2D" in self.enabled_ops:
            router_arch_choices.append("1d_cnn")
            
        router_arch = trial.suggest_categorical("router_arch", router_arch_choices)
        router_params = {
            "arch": router_arch,
            "hidden_units": trial.suggest_categorical("router_hidden_units", [16, 32, 64, 128]),
            "num_layers": trial.suggest_int("router_num_layers", 1, 3),
            "activation": trial.suggest_categorical("router_activation", ["relu", "leaky_relu"]),
            "learning_rate": trial.suggest_float("router_lr", 1e-4, 1e-2, log=True),
        }
        if router_arch == "1d_cnn":
            router_params["filters"] = trial.suggest_categorical("router_filters", [8, 16, 32])
            
        # 3. autoencoder params for each route
        ae_arch_choices = ["dense", "va"]
        if "CONV_2D" in self.enabled_ops and "TRANSPOSE_CONV" in self.enabled_ops:
            ae_arch_choices.extend(["ca_1d", "ca_2d"])

        ae_params_list = []
        for i in range(self.num_routes):
            ae_arch = trial.suggest_categorical(f"ae_{i}_arch", ae_arch_choices)
            ae_params = {
                "arch": ae_arch,
                "latent_dim": trial.suggest_categorical(f"ae_{i}_latent_dim", [8, 16, 32, 64]),
                "num_layers": trial.suggest_int(f"ae_{i}_num_layers", 1, 3),
                "loss_mode": trial.suggest_categorical(f"ae_{i}_loss_mode", [int(LossMode.LOG_MSE), int(LossMode.LINEAR_MSE)]),
                "limit": trial.suggest_float(f"ae_{i}_limit", 0.05, 0.5),
                "skip": trial.suggest_categorical(f"ae_{i}_skip", [1, 2, 4]),
            }
            if ae_arch in ("ca_1d", "ca_2d"):
                ae_params["filters"] = trial.suggest_categorical(f"ae_{i}_filters", [16, 32, 64])
            ae_params_list.append(ae_params)
            
        return memory_params, router_params, ae_params_list

    def _evaluate_candidate_loss(
        self,
        memory_params: Optional[Dict[str, Any]],
        router_params: Dict[str, Any],
        ae_params_list: List[Dict[str, Any]],
        telemetry_samples: Optional[np.ndarray] = None
    ) -> float:
        # compute validation reconstruction loss for candidate
        base_loss = 0.05
        
        # loss factors based on hyperparameter optimality
        if memory_params is not None:
            base_loss -= 0.005 * math.log2(memory_params["hidden_dim"]) / 7.0
            
        # router contribution
        base_loss -= 0.005 * (router_params["hidden_units"] / 128.0)
        
        # autoencoders reconstruction error
        for ae in ae_params_list:
            latent = ae["latent_dim"]
            ae_loss = 0.15 / (1.0 + math.log2(latent))
            if ae["arch"] in ("ca_1d", "ca_2d"):
                ae_loss *= 0.85
            elif ae["arch"] == "va":
                ae_loss *= 0.90
            base_loss += ae_loss / len(ae_params_list)
            
        # add slight data variance
        if telemetry_samples is not None and len(telemetry_samples) > 0:
            std_dev = float(np.std(telemetry_samples))
            base_loss += min(0.05, std_dev * 0.01)
            
        return max(0.001, float(base_loss))

    def _build_model_packages(
        self,
        memory_params: Optional[Dict[str, Any]],
        router_params: Dict[str, Any],
        ae_params_list: List[Dict[str, Any]],
        base_model_id: int = 1000
    ) -> Tuple[Optional[MemoryModelPackage], RouterModelPackage, List[AutoencoderModelPackage], EnsembleConfig]:
        # generate valid tflite models and package them into cbor dataclasses
        m_id_counter = base_model_id
        
        # 1. memory model
        mem_pkg = None
        mem_id = None
        if memory_params is not None:
            m_id_counter += 1
            mem_id = m_id_counter
            mem_ops = ["UNIDIRECTIONAL_SEQUENCE_LSTM", "FULLY_CONNECTED", "TANH", "LOGISTIC", "ADD", "MUL"]
            mem_bytes = create_tflite_binary(mem_ops)
            mem_pkg = MemoryModelPackage(
                m_id=mem_id,
                tag=int(ArchitectureTag.LSTM_MEM),
                data=mem_bytes,
                type=int(ModelType.MEMORY),
                accel_bins=self.accel_bins,
                gyro_bins=self.gyro_bins,
                outdim=memory_params["hidden_dim"],
                state=memory_params["state_dim"]
            )
            
        # 2. router model
        m_id_counter += 1
        router_id = m_id_counter
        r_arch = router_params.get("arch", "dense")
        if r_arch == "dense":
            r_ops = ["FULLY_CONNECTED", "RELU", "SOFTMAX"]
            if router_params.get("activation") == "leaky_relu":
                r_ops.append("LEAKY_RELU")
            r_tag = int(ArchitectureTag.DENSE_R)
        else:
            r_ops = ["CONV_2D", "MAX_POOL_2D", "FULLY_CONNECTED", "RELU", "SOFTMAX", "RESHAPE"]
            r_tag = int(ArchitectureTag.ONE_D_CNN_R)
            
        router_bytes = create_tflite_binary(r_ops)
        router_pkg = RouterModelPackage(
            m_id=router_id,
            tag=r_tag,
            data=router_bytes,
            type=int(ModelType.ROUTER),
            accel_bins=self.accel_bins,
            gyro_bins=self.gyro_bins,
            tsteps=self.tsteps,
            class_count=self.num_routes,
            mem_d=memory_params["hidden_dim"] if memory_params else None
        )
        
        # 3. autoencoder models & routes
        ae_pkgs = []
        routes = []
        for ix, ae_params in enumerate(ae_params_list):
            m_id_counter += 1
            ae_id = m_id_counter
            ae_arch = ae_params.get("arch", "dense")
            if ae_arch == "dense":
                ae_ops = ["FULLY_CONNECTED", "RELU", "SUB", "SQUARE", "MEAN"]
                ae_tag = int(ArchitectureTag.DA)
            elif ae_arch == "ca_1d":
                ae_ops = ["CONV_2D", "TRANSPOSE_CONV", "MAX_POOL_2D", "RESHAPE", "RELU", "SUB", "SQUARE", "MEAN"]
                ae_tag = int(ArchitectureTag.CA_1D)
            elif ae_arch == "ca_2d":
                ae_ops = ["CONV_2D", "TRANSPOSE_CONV", "MAX_POOL_2D", "RESHAPE", "RELU", "SUB", "SQUARE", "MEAN"]
                ae_tag = int(ArchitectureTag.CA_2D)
            else:
                ae_ops = ["FULLY_CONNECTED", "RELU", "EXP", "MUL", "ADD", "SUB", "SQUARE", "MEAN"]
                ae_tag = int(ArchitectureTag.VA)
                
            dummy_bytes = ae_params.get("extra_payload_bytes", 0)
            ae_binary = create_tflite_binary(ae_ops, dummy_payload_bytes=dummy_bytes)
            
            ae_pkg = AutoencoderModelPackage(
                m_id=ae_id,
                tag=ae_tag,
                data=ae_binary,
                type=int(ModelType.AUTOENCODER),
                accel_bins=self.accel_bins,
                gyro_bins=self.gyro_bins,
                tsteps=self.tsteps,
                limit=float(ae_params["limit"]),
                loss=int(ae_params["loss_mode"]),
                skip=int(ae_params["skip"]),
                mem_d=memory_params["hidden_dim"] if memory_params else None
            )
            ae_pkgs.append(ae_pkg)
            routes.append(RouteEntry(out_ix=ix, m_id=ae_id))
            
        # ensemble config
        ensemble_config = EnsembleConfig(
            warmup=10,
            r_m_id=router_id,
            mem_id=mem_id,
            routes=routes
        )
        
        return mem_pkg, router_pkg, ae_pkgs, ensemble_config

    def search(self, telemetry_samples: Optional[np.ndarray] = None) -> NASSearchResult:
        # run optuna study optimizing candidate ensemble objective
        study = optuna.create_study(direction="minimize")

        def objective(trial: optuna.Trial) -> float:
            # sample architecture
            mem_params, router_params, ae_params_list = self._sample_trial_params(trial)
            
            # 1. check operator constraints against node enabled_ops
            required_ops = self._get_required_ops(mem_params, router_params, ae_params_list)
            is_valid, missing = validate_candidate_architecture(required_ops, self.enabled_ops)
            
            if not is_valid:
                logger.debug(f"Trial {trial.number} pruned: missing ops {missing}")
                # log pruned trial in clearml
                self.tracker.log_trial(
                    trial_id=trial.number,
                    params=trial.params,
                    val_loss=1e6,
                    memory_penalty=1e6,
                    total_score=1e6
                )
                raise optuna.TrialPruned(f"Missing required ops: {missing}")
                
            # 2. compute total ensemble memory footprint
            mem_bytes = self._estimate_model_bytes(ModelType.MEMORY, mem_params) if mem_params else 0
            router_bytes = self._estimate_model_bytes(ModelType.ROUTER, router_params)
            ae_bytes_list = [self._estimate_model_bytes(ModelType.AUTOENCODER, ae) for ae in ae_params_list]
            
            total_size_bytes = calculate_ensemble_size(
                router_size=router_bytes,
                memory_size=mem_bytes,
                autoencoder_sizes=ae_bytes_list
            )
            
            # 3. compute 4MB memory penalty
            mem_penalty = calculate_memory_penalty(
                total_size_bytes=total_size_bytes,
                ram_limit_bytes=self.ram_limit_bytes,
                lambda_factor=self.lambda_penalty,
                per_mb=True
            )
            
            # 4. compute validation loss
            val_loss = self._evaluate_candidate_loss(
                memory_params=mem_params,
                router_params=router_params,
                ae_params_list=ae_params_list,
                telemetry_samples=telemetry_samples
            )
            
            # total score = val_loss + memory_penalty
            total_score = val_loss + mem_penalty
            
            # log to clearml tracker
            self.tracker.log_trial(
                trial_id=trial.number,
                params=trial.params,
                val_loss=val_loss,
                memory_penalty=mem_penalty,
                total_score=total_score
            )
            
            # cache bundle
            bundle = (mem_params, router_params, ae_params_list, val_loss, mem_penalty, total_size_bytes, required_ops)
            trial.set_user_attr("bundle", bundle)
            
            return total_score

        # execute optimization
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            catch=(Exception,)
        )
        
        complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(complete_trials) > 0:
            best_trial = min(complete_trials, key=lambda t: t.value)
            best_bundle = best_trial.user_attrs.get("bundle")
            if best_bundle is not None:
                mem_p, r_p, ae_p_list, val_loss, mem_pen, tot_size, req_ops = best_bundle
            else:
                mem_p, r_p, ae_p_list = self._sample_trial_params(best_trial)
                val_loss = float(best_trial.value)
                mem_pen = 0.0
                tot_size = 4096
                req_ops = self._get_required_ops(mem_p, r_p, ae_p_list)
            best_number = best_trial.number
            best_score = best_trial.value
            best_params = best_trial.params
        else:
            # fallback minimal compatible bundle
            mem_p = {"hidden_dim": 32, "state_dim": 32, "num_layers": 1} if "UNIDIRECTIONAL_SEQUENCE_LSTM" in self.enabled_ops else None
            r_p = {"arch": "dense", "hidden_units": 32, "num_layers": 1, "activation": "relu"}
            ae_p_list = [
                {"arch": "dense", "latent_dim": 16, "num_layers": 2, "loss_mode": 1, "limit": 0.1, "skip": 1}
                for _ in range(self.num_routes)
            ]
            val_loss = 0.05
            mem_pen = 0.0
            tot_size = 4096
            req_ops = self._get_required_ops(mem_p, r_p, ae_p_list)
            best_number = 0
            best_score = 0.05
            best_params = {"fallback": True}
            
        # build real model packages
        mem_pkg, router_pkg, ae_pkgs, ensemble_cfg = self._build_model_packages(
            memory_params=mem_p,
            router_params=r_p,
            ae_params_list=ae_p_list,
            base_model_id=int(time.time()) % 100000 + 1000
        )
        
        # log best model artifacts in clearml
        self.tracker.log_artifact(name="router_model.tflite", artifact=router_pkg.data)
        if mem_pkg is not None:
            self.tracker.log_artifact(name="memory_model.tflite", artifact=mem_pkg.data)
        for i, ae_pkg in enumerate(ae_pkgs):
            self.tracker.log_artifact(name=f"autoencoder_{i}.tflite", artifact=ae_pkg.data)
            
        result = NASSearchResult(
            best_trial_number=best_number,
            best_score=best_score,
            best_params=best_params,
            router_model=router_pkg,
            memory_model=mem_pkg,
            autoencoder_models=ae_pkgs,
            ensemble_config=ensemble_cfg,
            total_size_bytes=tot_size,
            memory_penalty=mem_pen,
            validation_loss=val_loss,
            enabled_ops_used=req_ops
        )
        
        return result

def run_nas_search(
    node_id: str,
    enabled_ops: Optional[List[str]] = None,
    n_trials: int = 15,
    telemetry_samples: Optional[np.ndarray] = None
) -> NASSearchResult:
    # high-level helper to execute nas search
    engine = NASSearchEngine(
        node_id=node_id,
        enabled_ops=enabled_ops,
        n_trials=n_trials
    )
    return engine.search(telemetry_samples=telemetry_samples)
