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

import os
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
from mlflow.tracking import MlflowClient

# Repository root: <repo>/src/utils/mlflow_logger.py -> parents[2] is <repo>
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MLRUNS_DIR = _REPO_ROOT / "mlruns"


def _is_writable_dir(path: Path) -> bool:
    """Return True if *path* (or its first existing ancestor) is writable."""
    p = path
    while not p.exists():
        if p.parent == p:
            return False
        p = p.parent
    return os.access(p, os.W_OK)


def _artifact_location_is_usable(artifact_location: str | None) -> bool:
    """Best-effort check that an MLflow artifact_location can be written to."""
    if not artifact_location:
        return False
    parsed = urlparse(artifact_location)
    if parsed.scheme in ("", "file"):
        local = Path(parsed.path if parsed.scheme == "file" else artifact_location)
        return _is_writable_dir(local)
    # Non-local stores (s3://, http://, etc.) - assume the user knows best.
    return True


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
        artifact_location: str | None = None,
    ) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self._run: mlflow.ActiveRun | None = None

        # Default to a stable, repo-local sqlite tracking store + file
        # artifact store so the configuration is the same regardless of the
        # process cwd (e.g. notebooks running from a sub-directory) and we
        # don't pick up stale databases left in cwd.
        if tracking_uri is None and "MLFLOW_TRACKING_URI" not in os.environ:
            _DEFAULT_MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
            tracking_uri = f"sqlite:///{_DEFAULT_MLRUNS_DIR / 'mlflow.db'}"
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        # Default artifact_location to a writable dir under the repo.
        if artifact_location is None:
            _DEFAULT_MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
            artifact_location = _DEFAULT_MLRUNS_DIR.as_uri()
        self._desired_artifact_location = artifact_location

        self._ensure_experiment(experiment_name, artifact_location)

    def _ensure_experiment(
        self, experiment_name: str, artifact_location: str
    ) -> None:
        """Set the active experiment, recovering from a stale artifact_location.

        If an experiment with *experiment_name* already exists but its
        ``artifact_location`` points somewhere we cannot write to (e.g. an
        unmounted external drive), MLflow will crash on the first
        ``log_artifact`` call. Detect that situation up-front and either
        rename the stale experiment aside or fall back to a fresh name.
        """
        client = MlflowClient()
        existing = client.get_experiment_by_name(experiment_name)
        if existing is None:
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
            mlflow.set_experiment(experiment_name)
            return

        if _artifact_location_is_usable(existing.artifact_location):
            mlflow.set_experiment(experiment_name)
            return

        warnings.warn(
            f"[MLflowLogger] Experiment '{experiment_name}' has unwritable "
            f"artifact_location='{existing.artifact_location}'. Renaming the "
            "stale experiment and creating a fresh one with "
            f"artifact_location='{artifact_location}'.",
            stacklevel=2,
        )
        # Try to rename the broken experiment out of the way; if that fails,
        # fall back to using a suffixed experiment name for this session.
        try:
            client.rename_experiment(
                existing.experiment_id,
                f"{experiment_name}_stale_{existing.experiment_id}",
            )
            mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
            mlflow.set_experiment(experiment_name)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"[MLflowLogger] Could not rename stale experiment ({exc}); "
                "using a suffixed experiment name instead.",
                stacklevel=2,
            )
            fallback = f"{experiment_name}_local"
            self.experiment_name = fallback
            if client.get_experiment_by_name(fallback) is None:
                mlflow.create_experiment(fallback, artifact_location=artifact_location)
            mlflow.set_experiment(fallback)

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
        """Upload a local file or directory to the run's artifact store.

        Failures to upload (e.g. an artifact_location that is no longer
        writable) are logged as warnings rather than raised, so a long
        training loop is never aborted by an MLflow bookkeeping error.
        """
        try:
            mlflow.log_artifact(str(local_path), artifact_path=artifact_path)
        except (PermissionError, OSError) as exc:
            warnings.warn(
                f"[MLflowLogger] Failed to log artifact '{local_path}': {exc}. "
                "Continuing without uploading.",
                stacklevel=2,
            )

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
