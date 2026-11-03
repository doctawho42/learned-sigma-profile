"""Minimal experiment logger independent of external services (W&B, etc)."""

from __future__ import annotations

import json
import sys
import pickle
import subprocess
from datetime import datetime
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch

# Optional dependencies with graceful fallback
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class ExperimentLogger:
    """Lightweight experiment logger for tracking training runs and metrics."""

    def __init__(self, log_dir: str, experiment_name: str | None = None) -> None:
        """Initialize experiment logger.
        
        Args:
            log_dir: Root directory for all experiments.
            experiment_name: Name of this experiment. If None, uses timestamp.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name from timestamp if not provided
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.exp_dir = self.log_dir / self.experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics history and config storage
        self.metrics_history = []
        self.config = None
        
        # Set matplotlib backend early if available
        if plt is not None:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
            except Exception:
                pass
        
        # Initialize metadata with environment information
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "git_hash": self._get_git_hash(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    def log_config(self, config: object) -> None:
        """Save configuration to JSON file.
        
        Automatically converts dataclass to dict.
        
        Args:
            config: Configuration object (dataclass or dict).
        """
        if is_dataclass(config):
            config_dict = asdict(config)
        else:
            config_dict = config
        
        self.config = config_dict
        
        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

    def log_metrics(self, metrics: dict, step: int, phase: str = "train") -> None:
        """Log metrics to JSONL file (one JSON line per call).
        
        Args:
            metrics: Dictionary of metric values.
            step: Current step/epoch number.
            phase: Training phase ("train", "val", "test", etc).
        """
        entry = {
            "step": step,
            "phase": phase,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        
        self.metrics_history.append(entry)
        
        # Append to JSONL file
        metrics_path = self.exp_dir / "metrics.jsonl"
        with open(metrics_path, 'a') as f:
            json.dump(entry, f, default=str)
            f.write('\n')

    def log_artifact(self, name: str, data: object) -> None:
        """Save artifact in appropriate format based on data type.
        
        Args:
            name: Name of the artifact (without extension).
            data: Data to save (dict, DataFrame, Figure, or other).
        """
        artifact_path = self.exp_dir / name
        
        if isinstance(data, dict):
            # Save dict as JSON
            with open(f"{artifact_path}.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif pd is not None and isinstance(data, pd.DataFrame):
            # Save DataFrame as CSV
            data.to_csv(f"{artifact_path}.csv", index=False)
        
        elif plt is not None and isinstance(data, plt.Figure):
            # Save matplotlib Figure as PDF
            data.savefig(f"{artifact_path}.pdf", bbox_inches='tight')
            plt.close(data)
        
        else:
            # Fallback: pickle
            with open(f"{artifact_path}.pkl", 'wb') as f:
                pickle.dump(data, f)

    def log_model_summary(self, model: torch.nn.Module) -> None:
        """Log model architecture summary and parameter counts.
        
        Args:
            model: PyTorch model.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "model_class": model.__class__.__name__,
        }
        
        summary_path = self.exp_dir / "model_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    def finalize(self, final_metrics: dict = None) -> None:
        """Finalize experiment: compute duration, save summary and full history.
        
        Args:
            final_metrics: Optional dict of final metrics to include in summary.
        """
        # Parse timestamps and compute duration
        start_time = datetime.fromisoformat(self.metadata["start_time"])
        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()
        
        # Update metadata with end information
        self.metadata["end_time"] = end_time.isoformat()
        self.metadata["duration_seconds"] = duration_seconds
        
        if final_metrics is not None:
            self.metadata["final_metrics"] = final_metrics
        
        # Save summary
        summary_path = self.exp_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        # Save full metrics history
        history_path = self.exp_dir / "full_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)

    @staticmethod
    def _get_git_hash() -> str:
        """Get current git commit hash.
        
        Returns:
            Git commit hash, or "unknown" if not a git repo or git not available.
        """
        try:
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode('utf-8').strip()
            return git_hash
        except Exception:
            return "unknown"
