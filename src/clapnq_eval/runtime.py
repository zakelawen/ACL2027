from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig, ModelIdentity
from .io import exclusive_output_lock, read_jsonl


_SERVER_INFO_KEYS = (
    "enable_deterministic_inference",
    "model_path",
    "tokenizer_path",
    "served_model_name",
    "random_seed",
    "tp_size",
    "context_length",
    "mem_fraction_static",
    "grammar_backend",
    "reasoning_parser",
    "host",
    "port",
)

_IDENTITY_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "model.safetensors.index.json",
)


def paths_equal(left: str | Path, right: str | Path) -> bool:
    first = Path(left).expanduser()
    second = Path(right).expanduser()
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return first.as_posix() == second.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_model_identity(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path)
    files: dict[str, str] = {}
    for name in _IDENTITY_FILES:
        candidate = root / name
        if candidate.is_file():
            files[name] = sha256_file(candidate)
    shards = sorted(path for path in root.glob("*.safetensors") if path.is_file())
    manifest = [
        {"name": path.name, "size": path.stat().st_size} for path in shards
    ]
    payload = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return {
        "files": files,
        "weight_manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_model_identity(model_path: str | Path, expected: ModelIdentity) -> dict[str, Any]:
    actual = hash_model_identity(model_path)
    missing = [name for name in expected.files if name not in actual["files"]]
    if missing:
        raise RuntimeError(
            f"Model identity files missing under {model_path}: {missing}"
        )
    mismatched = [
        name
        for name, digest in expected.files.items()
        if actual["files"][name] != digest
    ]
    if mismatched:
        raise RuntimeError(
            f"Model identity hash mismatch under {model_path}: {mismatched}"
        )
    if actual["weight_manifest_sha256"] != expected.weight_manifest_sha256:
        raise RuntimeError(
            f"Weight manifest mismatch under {model_path}: "
            f"{actual['weight_manifest_sha256']} != {expected.weight_manifest_sha256}"
        )
    return actual


def versions_match(reported: str, expected: str) -> bool:
    left = str(reported).strip()
    right = str(expected).strip()
    return left == right or left.startswith(right) or right.startswith(left)


def require_bool(value: Any, name: str) -> bool:
    if value is True or value is False:
        return value
    raise RuntimeError(f"{name} must be a boolean, got {value!r}")


def require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} must be an integer, got {value!r}")
    return value


def summarize_judge_server(info: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    version = info.get("version")
    if version is not None:
        summary["sglang_version"] = version
    for key in _SERVER_INFO_KEYS:
        if key in info:
            summary[key] = info[key]
    return summary


def verify_judge_server(
    config: ExperimentConfig, info: dict[str, Any]
) -> dict[str, Any]:
    summary = summarize_judge_server(info)
    if "enable_deterministic_inference" not in summary:
        raise RuntimeError(
            "Judge /server_info did not report enable_deterministic_inference. "
            "Cannot confirm the running server matches judge.deterministic_inference."
        )
    actual = require_bool(
        summary["enable_deterministic_inference"],
        "enable_deterministic_inference",
    )
    if actual != config.judge.deterministic_inference:
        raise RuntimeError(
            f"Judge server enable_deterministic_inference={actual} "
            f"does not match judge.deterministic_inference="
            f"{config.judge.deterministic_inference}. "
            "Restart scripts/serve_judge.sh so the process flag matches the YAML "
            "(default on; use DETERMINISTIC_INFERENCE=0 only when the YAML is false)."
        )

    expected_name = config.judge.served_model
    served = summary.get("served_model_name")
    names: list[str] = []
    if isinstance(served, str) and served:
        names = [served]
    elif isinstance(served, (list, tuple)):
        names = [str(item) for item in served if item]
    if names and expected_name not in names:
        raise RuntimeError(
            f"Judge server served_model_name={names} does not include "
            f"{expected_name!r}."
        )

    if "random_seed" not in summary:
        raise RuntimeError("Judge /server_info did not report random_seed.")
    reported_seed = require_int(summary["random_seed"], "random_seed")
    if reported_seed != int(config.run.seed):
        raise RuntimeError(
            f"Judge server random_seed={reported_seed} does not match "
            f"run.seed={config.run.seed}."
        )

    reported_path = summary.get("model_path")
    if not reported_path or not isinstance(reported_path, str):
        raise RuntimeError("Judge /server_info did not report model_path.")
    if not paths_equal(reported_path, config.judge.model_path):
        raise RuntimeError(
            f"Judge server model_path={reported_path!r} does not match "
            f"judge.model_path={str(config.judge.model_path)!r}."
        )

    reported_version = summary.get("sglang_version")
    if not isinstance(reported_version, str) or not reported_version:
        raise RuntimeError("Judge /server_info did not report version.")
    if reported_version != config.judge.sglang_version:
        raise RuntimeError(
            f"Judge server sglang_version={reported_version!r} does not match "
            f"judge.sglang_version={config.judge.sglang_version!r}."
        )

    identity = verify_model_identity(config.judge.model_path, config.judge.model_identity)
    summary["model_identity"] = identity
    return summary


def verify_recorded_judge_server(
    config: ExperimentConfig, snapshot: dict[str, Any]
) -> dict[str, Any]:
    info = dict(snapshot)
    if "sglang_version" in info and "version" not in info:
        info["version"] = info["sglang_version"]
    verified = verify_judge_server(config, info)
    recorded_identity = snapshot.get("model_identity")
    expected = {
        "files": config.judge.model_identity.files,
        "weight_manifest_sha256": config.judge.model_identity.weight_manifest_sha256,
    }
    if recorded_identity != expected and recorded_identity != verified["model_identity"]:
        if recorded_identity != expected:
            raise RuntimeError(
                "Judge snapshot model_identity does not match the YAML identity."
            )
    return verified


def verify_generator_server(
    *,
    expected_path: Path,
    served_model: str,
    info: dict[str, Any] | None,
    model_card: dict[str, Any] | None,
) -> str:
    reported = None
    if info:
        reported = info.get("model_path") or info.get("model")
    if not reported and model_card:
        reported = model_card.get("root") or model_card.get("model_path")
    if not reported or not isinstance(reported, str):
        raise RuntimeError(
            f"Generator {served_model!r} did not report a model path via "
            "/server_info or /v1/models. Refusing to treat YAML model_path "
            "as verified."
        )
    if not paths_equal(reported, expected_path):
        raise RuntimeError(
            f"Generator {served_model!r} is serving {reported!r}, "
            f"expected {str(expected_path)!r}."
        )
    return reported


def load_judge_server_snapshot(config: ExperimentConfig) -> dict[str, Any]:
    path = config.run_dir / "judge_server.json"
    if not path.exists():
        raise RuntimeError(
            f"Formal scoring requires a run-level Judge snapshot at {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"Judge snapshot is empty or invalid: {path}")
    return verify_recorded_judge_server(config, payload)


def ensure_judge_server_snapshot(
    config: ExperimentConfig, server: dict[str, Any]
) -> dict[str, Any]:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    path = config.run_dir / "judge_server.json"
    with exclusive_output_lock(path, blocking=True):
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != server:
                raise RuntimeError(
                    f"Run-level Judge snapshot {path} does not match the live server. "
                    "Use a new --run-name after changing the Judge process."
                )
            return existing
        recorded = _existing_judgment_servers(config)
        if recorded and any(item != server for item in recorded):
            raise RuntimeError(
                "Existing judgments already record a Judge snapshot that does "
                "not match the live server; refusing to create "
                f"{path.name}."
            )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(server, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        return server


def _existing_judgment_servers(config: ExperimentConfig) -> list[dict[str, Any]]:
    directory = config.run_dir / "judgments"
    if not directory.exists():
        return []
    found: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        for row in read_jsonl(path, tolerate_trailing_partial=True):
            value = row.get("judge_server")
            if isinstance(value, dict):
                found.append(value)
    return found
