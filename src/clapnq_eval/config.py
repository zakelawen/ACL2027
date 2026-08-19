from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


Condition = Literal["gold", "closed_book"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    source_url: str
    source_sha256: str | None = None
    raw_path: Path
    normalized_path: Path
    answerable_only: bool = True
    max_samples: int | None = Field(default=None, ge=1)

    @field_validator("source_sha256")
    @classmethod
    def validate_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip().lower()
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError("data.source_sha256 must be a 64-character hex digest")
        return value


class RunConfig(StrictModel):
    name: str = "main"
    output_dir: Path = Path("runs")
    seed: int = 20260819

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value or any(ch in value for ch in "/\\"):
            raise ValueError("run.name must be a non-empty path-safe name")
        return value


class ModelConfig(StrictModel):
    served_model: str


class RequestConfig(StrictModel):
    base_url: str
    api_key: str = "EMPTY"
    max_tokens: int = Field(default=256, ge=1)
    concurrency: int = Field(default=16, ge=1)
    request_timeout_seconds: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=5, ge=1)
    retry_base_seconds: float = Field(default=1.0, gt=0)


class GenerationConfig(RequestConfig):
    conditions: list[Condition] = Field(default_factory=lambda: ["gold", "closed_book"])
    temperature: float = Field(default=0.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: int = Field(default=0, ge=0)
    min_p: float = Field(default=0.0, ge=0, le=1)
    presence_penalty: float = 0.0
    repetition_penalty: float = Field(default=1.0, gt=0)

    @field_validator("conditions")
    @classmethod
    def unique_conditions(cls, value: list[Condition]) -> list[Condition]:
        if not value:
            raise ValueError("generation.conditions cannot be empty")
        return list(dict.fromkeys(value))


class JudgeConfig(RequestConfig):
    served_model: str
    enable_thinking: bool = False
    deterministic_inference: bool = True
    temperature: float = Field(default=0.7, ge=0)
    top_p: float = Field(default=0.8, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    min_p: float = Field(default=0.0, ge=0, le=1)
    # Official chat non-thinking uses 1.5. The Judge decodes a JSON label
    # first, then a short reason; a chat-style presence penalty is the
    # wrong prior for that schema, so this stays 0.0.
    presence_penalty: float = 0.0
    repetition_penalty: float = Field(default=1.0, gt=0)


class MetricsConfig(StrictModel):
    bootstrap_samples: int = Field(default=10000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    allow_rouge_fallback: bool = False
    bertscore: bool = False
    bertscore_model: str = "roberta-large"


class ExperimentConfig(StrictModel):
    data: DataConfig
    run: RunConfig
    models: dict[str, ModelConfig]
    generation: GenerationConfig
    judge: JudgeConfig
    metrics: MetricsConfig

    @property
    def run_dir(self) -> Path:
        return self.run.output_dir / self.run.name


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = ExperimentConfig.model_validate(raw)

    project_root = _project_root(config_path)
    config.data.raw_path = _resolve(project_root, config.data.raw_path)
    config.data.normalized_path = _resolve(project_root, config.data.normalized_path)
    config.run.output_dir = _resolve(project_root, config.run.output_dir)
    return config


def _project_root(config_path: Path) -> Path:
    for candidate in config_path.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return config_path.parent.parent


def _resolve(root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else (root / path).resolve()

