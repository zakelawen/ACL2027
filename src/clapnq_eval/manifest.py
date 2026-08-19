from __future__ import annotations

import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .config import ExperimentConfig
from .io import record_sha256
from .prompts import (
    GENERATION_PROMPT_VERSION,
    GENERATION_SYSTEM_PROMPT,
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    prompt_sha256,
)


def ensure_manifest_compatible(config: ExperimentConfig) -> None:
    path = config.run_dir / "manifest.json"
    if not path.exists():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    existing = manifest.get("run_signature")
    if existing is None:
        return
    current = _run_signature(config)
    if existing != current:
        raise RuntimeError(
            f"Run configuration does not match {path}. "
            "Use a new --run-name after changing data, prompts, models, "
            "generation settings, Judge settings, or the seed."
        )


def update_manifest(
    config: ExperimentConfig,
    *,
    action: str,
    details: dict[str, Any] | None = None,
) -> Path:
    ensure_manifest_compatible(config)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    path = config.run_dir / "manifest.json"
    now = datetime.now(timezone.utc).isoformat()

    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "created_at": now,
            "project_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "actions": [],
        }

    manifest["updated_at"] = now
    manifest["config"] = _redact(config.model_dump(mode="json"))
    manifest["prompts"] = _prompt_metadata()
    manifest["run_signature"] = _run_signature(config)
    manifest["actions"].append(
        {
            "name": action,
            "time": now,
            "details": details or {},
        }
    )
    _write_json(path, manifest)
    _snapshot_prompts(config.run_dir)
    return path


def _prompt_metadata() -> dict[str, dict[str, str]]:
    return {
        "generation": {
            "version": GENERATION_PROMPT_VERSION,
            "system_sha256": prompt_sha256(GENERATION_SYSTEM_PROMPT, ""),
        },
        "judge": {
            "version": JUDGE_PROMPT_VERSION,
            "system_sha256": prompt_sha256(JUDGE_SYSTEM_PROMPT, ""),
        },
    }


def _run_signature(config: ExperimentConfig) -> dict[str, Any]:
    config_payload = _redact(config.model_dump(mode="json"))
    config_payload.pop("metrics", None)
    payload = {
        "project_version": __version__,
        "config": config_payload,
        "prompts": _prompt_metadata(),
    }
    return {
        "sha256": record_sha256(payload),
        "payload": payload,
    }


def _snapshot_prompts(run_dir: Path) -> None:
    prompt_dir = run_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / "generation_system.txt").write_text(
        GENERATION_SYSTEM_PROMPT + "\n", encoding="utf-8"
    )
    (prompt_dir / "judge_system.txt").write_text(
        JUDGE_SYSTEM_PROMPT + "\n", encoding="utf-8"
    )


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "openai",
        "pydantic",
        "PyYAML",
        "numpy",
        "scipy",
        "tqdm",
        "rouge-score",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: ("<redacted>" if key == "api_key" else _redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
