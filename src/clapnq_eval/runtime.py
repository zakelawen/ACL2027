from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
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


def paths_equal(left: str | Path, right: str | Path) -> bool:
    first = Path(left).expanduser()
    second = Path(right).expanduser()
    try:
        return first.resolve() == second.resolve()
    except OSError:
        return first.as_posix() == second.as_posix()


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
    actual = bool(summary["enable_deterministic_inference"])
    expected = bool(config.judge.deterministic_inference)
    if actual != expected:
        raise RuntimeError(
            f"Judge server enable_deterministic_inference={actual} "
            f"does not match judge.deterministic_inference={expected}. "
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

    reported_seed = summary.get("random_seed")
    if reported_seed is None:
        raise RuntimeError("Judge /server_info did not report random_seed.")
    if int(reported_seed) != int(config.run.seed):
        raise RuntimeError(
            f"Judge server random_seed={reported_seed} does not match "
            f"run.seed={config.run.seed}."
        )

    reported_path = summary.get("model_path")
    if not reported_path:
        raise RuntimeError("Judge /server_info did not report model_path.")
    if not paths_equal(reported_path, config.judge.model_path):
        raise RuntimeError(
            f"Judge server model_path={reported_path!r} does not match "
            f"judge.model_path={str(config.judge.model_path)!r}."
        )

    reported_version = summary.get("sglang_version")
    if reported_version is None:
        raise RuntimeError("Judge /server_info did not report version.")
    if str(reported_version) != config.judge.sglang_version:
        raise RuntimeError(
            f"Judge server sglang_version={reported_version!r} does not match "
            f"judge.sglang_version={config.judge.sglang_version!r}."
        )
    return summary


def verify_generator_server(
    *,
    expected_path: Path,
    served_model: str,
    info: dict[str, Any] | None,
    model_card: dict[str, Any] | None,
) -> None:
    reported = None
    if info:
        reported = info.get("model_path") or info.get("model")
    if not reported and model_card:
        reported = model_card.get("root") or model_card.get("model_path")
    if reported and not paths_equal(reported, expected_path):
        raise RuntimeError(
            f"Generator {served_model!r} is serving {reported!r}, "
            f"expected {str(expected_path)!r}."
        )


def load_judge_server_snapshot(config: ExperimentConfig) -> dict[str, Any]:
    path = config.run_dir / "judge_server.json"
    if not path.exists():
        raise RuntimeError(
            f"Formal scoring requires a run-level Judge snapshot at {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"Judge snapshot is empty or invalid: {path}")
    return payload


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
