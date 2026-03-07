"""
mlflow_logger.py – MLflow experiment tracking helpers.

Provides a thin wrapper around the MLflow Python client that:
  * creates or reuses a named experiment,
  * logs hyperparameters, metrics and artefacts in a consistent schema,
  * can be used as a context manager to automatically end runs.

Usage
-----
    from src.utils.mlflow_logger import MLflowLogger

    with MLflowLogger(experiment_name="rl_training") as logger:
        logger.log_params({"algorithm": "DQN", "lr": 1e-4})
        for step, reward in training_loop():
            logger.log_metric("reward", reward, step=step)
        logger.log_artifact("models/best_agent.zip")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient


class MLflowLogger:
    """
    Experiment tracker backed by MLflow.

    Parameters
    ----------
    experiment_name : str
        Name of the MLflow experiment (created if it does not exist).
    run_name : str | None
        Optional display name for the run.
    tracking_uri : str | None
        MLflow tracking server URI.  Defaults to the local ``mlruns/``
        directory (``None`` keeps MLflow's own default).
    tags : dict | None
        Key-value tags attached to the run.
    """

    def __init__(
        self,
        experiment_name: str = "default",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, Any] | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self._run: mlflow.ActiveRun | None = None

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

    # ------------------------------------------------------------------
    # Context-manager interface
    # ------------------------------------------------------------------

    def __enter__(self) -> "MLflowLogger":
        self.start_run()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_run()

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self) -> None:
        """Start a new MLflow run (call once per experiment)."""
        self._run = mlflow.start_run(run_name=self.run_name, tags=self.tags)

    def end_run(self) -> None:
        """End the active MLflow run."""
        mlflow.end_run()
        self._run = None

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def log_params(self, params: dict[str, Any]) -> None:
        """Log a dictionary of hyperparameters."""
        mlflow.log_params(params)

    def log_param(self, key: str, value: Any) -> None:
        """Log a single hyperparameter."""
        mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a scalar metric, optionally at a given training step."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None:
        """Log multiple scalar metrics at once."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Upload a local file or directory to the run's artifact store."""
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        """Serialize *data* as JSON and store it as an artefact."""
        mlflow.log_dict(data, artifact_file)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def run_id(self) -> str | None:
        """Active run ID, or ``None`` if no run is active."""
        if self._run is not None:
            return self._run.info.run_id
        return None

    @property
    def client(self) -> MlflowClient:
        """Direct access to the underlying :class:`mlflow.tracking.MlflowClient`."""
        return MlflowClient()
