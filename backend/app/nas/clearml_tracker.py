# clearml experiment tracking logger with offline fallback
import logging
import os
import tempfile
from typing import Dict, Any, Optional, List, Union
from app.config import get_settings

logger = logging.getLogger("clearml_tracker")

class ClearMLTracker:
    # experiment tracking integration for clearml with graceful offline mode
    def __init__(
        self,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        task_type: str = "training",
        tags: Optional[List[str]] = None,
        force_offline: bool = False,
    ):
        settings = get_settings()
        self.project_name = project_name or settings.clearml_project_name
        self.task_name = task_name or "nas_optimization"
        self.task_type = task_type
        self.tags = tags or ["optuna", "nas", "vibration"]
        self.force_offline = force_offline
        
        self.is_offline = True
        self.task = None
        self._logged_trials: List[Dict[str, Any]] = []
        self._logged_scalars: List[Dict[str, Any]] = []
        self._logged_artifacts: Dict[str, Any] = {}
        self._logged_parameters: Dict[str, Any] = {}
        
        self._init_clearml(settings)

    def _init_clearml(self, settings) -> None:
        # attempt to connect to clearml server, fallback to offline tracking
        if self.force_offline:
            logger.info("ClearML tracker running in forced offline mode")
            self.is_offline = True
            return

        # setup env vars if provided
        if settings.clearml_api_host:
            os.environ["CLEARML_API_HOST"] = settings.clearml_api_host
        if settings.clearml_web_host:
            os.environ["CLEARML_WEB_HOST"] = settings.clearml_web_host
        if settings.clearml_files_host:
            os.environ["CLEARML_FILES_HOST"] = settings.clearml_files_host
            
        try:
            from clearml import Task
            # attempt init
            self.task = Task.init(
                project_name=self.project_name,
                task_name=self.task_name,
                task_type=self.task_type,
                tags=self.tags,
                reuse_last_task_id=False,
                auto_connect_frameworks=False
            )
            self.is_offline = False
            logger.info(f"Connected to ClearML project '{self.project_name}', task '{self.task_name}'")
        except Exception as e:
            logger.warning(f"ClearML server unavailable ({e}). Using offline local tracking fallback.")
            self.is_offline = True
            self.task = None

    def log_parameters(self, params: Dict[str, Any]) -> None:
        # log trial or global hyperparameters
        self._logged_parameters.update(params)
        if not self.is_offline and self.task:
            try:
                self.task.connect(self._logged_parameters)
            except Exception as e:
                logger.debug(f"ClearML connect params failed: {e}")

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        # log single scalar metric
        self._logged_scalars.append({
            "title": title,
            "series": series,
            "value": float(value),
            "iteration": int(iteration)
        })
        if not self.is_offline and self.task:
            try:
                logger_instance = self.task.get_logger()
                logger_instance.report_scalar(
                    title=title,
                    series=series,
                    value=float(value),
                    iteration=int(iteration)
                )
            except Exception as e:
                logger.debug(f"ClearML report_scalar failed: {e}")

    def log_trial(
        self,
        trial_id: int,
        params: Dict[str, Any],
        val_loss: float,
        memory_penalty: float,
        total_score: float
    ) -> None:
        # record trial evaluation metrics
        record = {
            "trial_id": int(trial_id),
            "params": dict(params),
            "val_loss": float(val_loss),
            "memory_penalty": float(memory_penalty),
            "total_score": float(total_score),
        }
        self._logged_trials.append(record)
        
        # log to clearml logger
        self.log_scalar("Objective", "val_loss", val_loss, trial_id)
        self.log_scalar("Objective", "memory_penalty", memory_penalty, trial_id)
        self.log_scalar("Objective", "total_score", total_score, trial_id)
        
        for k, v in params.items():
            if isinstance(v, (int, float)):
                self.log_scalar("Parameters", k, float(v), trial_id)

    def log_artifact(
        self,
        name: str,
        artifact: Union[bytes, str, Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        # upload or store model flatbuffer or json artifact
        self._logged_artifacts[name] = {
            "data": artifact,
            "metadata": metadata or {}
        }
        
        if not self.is_offline and self.task:
            try:
                if isinstance(artifact, (bytes, bytearray)):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
                        f.write(artifact)
                        temp_path = f.name
                    self.task.upload_artifact(name=name, artifact_object=temp_path, metadata=metadata)
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                elif isinstance(artifact, str) and os.path.exists(artifact):
                    self.task.upload_artifact(name=name, artifact_object=artifact, metadata=metadata)
                else:
                    self.task.upload_artifact(name=name, artifact_object=artifact, metadata=metadata)
            except Exception as e:
                logger.debug(f"ClearML upload_artifact failed: {e}")

    def get_logged_trials(self) -> List[Dict[str, Any]]:
        # return list of logged trial records
        return list(self._logged_trials)

    def get_logged_scalars(self) -> List[Dict[str, Any]]:
        # return logged scalars history
        return list(self._logged_scalars)

    def get_logged_artifacts(self) -> Dict[str, Any]:
        # return stored artifacts
        return dict(self._logged_artifacts)

    def close(self) -> None:
        # flush and close clearml task
        if not self.is_offline and self.task:
            try:
                self.task.close()
            except Exception:
                pass
            self.task = None

def get_clearml_tracker(
    project_name: Optional[str] = None,
    task_name: Optional[str] = None,
    force_offline: bool = False
) -> ClearMLTracker:
    # factory helper for clearml tracker
    return ClearMLTracker(
        project_name=project_name,
        task_name=task_name,
        force_offline=force_offline
    )
