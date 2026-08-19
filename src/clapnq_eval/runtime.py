from __future__ import annotations

from typing import Any

from .config import ExperimentConfig


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
    return summary


def judge_server_matches_config(
    config: ExperimentConfig, server: dict[str, Any] | None
) -> bool:
    if not server:
        return True
    actual = server.get("enable_deterministic_inference")
    if actual is None:
        return False
    return bool(actual) == bool(config.judge.deterministic_inference)
