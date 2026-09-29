from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    config_path: Path

    @property
    def project_root(self) -> Path:
        return self.config_path.parent.parent.resolve()

    def _resolve(self, value: str | Path) -> Path:
        expanded = os.path.expandvars(os.path.expanduser(str(value)))
        path = Path(expanded)
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    @property
    def data_dir(self) -> Path:
        return self._resolve(self.raw["data"]["data_dir"])

    @property
    def train_path(self) -> Path:
        return self.data_dir / str(self.raw["data"]["train"])

    @property
    def items_path(self) -> Path:
        return self.data_dir / str(self.raw["data"]["benchmark_items"])

    @property
    def queries_path(self) -> Path:
        return self.data_dir / str(self.raw["data"]["benchmark_queries"])

    @property
    def artifact_root(self) -> Path:
        return self._resolve(self.raw["artifacts"]["root"])


def load_config(path: str | Path = "configs/mac_m4pro.yaml") -> Config:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config(raw=raw, config_path=config_path)
