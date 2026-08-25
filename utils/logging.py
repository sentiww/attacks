from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


class ExperimentLogger:
    def __init__(self, run_dir: str | Path, backend: str = "csv") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend.lower()
        self._wandb = None
        if self.backend == "wandb":
            try:
                import wandb
            except ImportError as error:
                raise ImportError("Install wandb to use the wandb logging backend") from error
            self._wandb = wandb.init(project="attack-lab", name=self.run_dir.name, dir=str(self.run_dir))
        elif self.backend not in {"csv", "json"}:
            raise ValueError("logging.backend must be one of: csv, json, wandb")

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        values = {key: _serializable(value) for key, value in metrics.items()}
        if self.backend == "csv":
            path = self.run_dir / "metrics.csv"
            exists = path.exists()
            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["step", "metric", "value"])
                if not exists:
                    writer.writeheader()
                for metric, value in values.items():
                    writer.writerow({"step": step, "metric": metric, "value": value})
        elif self.backend == "json":
            with (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, **values}) + "\n")
        else:
            assert self._wandb is not None
            self._wandb.log(values, step=step)

    def log_config(self, config: Mapping[str, Any]) -> None:
        with (self.run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(dict(config), handle, sort_keys=False)
        if self._wandb is not None:
            self._wandb.config.update(dict(config), allow_val_change=True)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value
