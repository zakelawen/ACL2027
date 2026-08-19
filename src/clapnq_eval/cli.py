from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Sequence

from .config import Condition, ExperimentConfig, load_config
from .data import prepare_data
from .generate import generate_answers
from .io import read_jsonl
from .judge import judge_answers
from .manifest import ensure_manifest_compatible, update_manifest
from .report import score_run


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "experiment.yaml"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    config = load_config(args.config)
    if args.run_name is not None:
        config.run.name = args.run_name
    ensure_manifest_compatible(config)

    if args.command == "prepare":
        result = prepare_data(config.data, force_download=args.force_download)
        update_manifest(config, action="prepare", details=result)
        _print_json(result)
        return 0

    if args.command == "generate":
        _require_prepared_data(config)
        conditions = _select_conditions(args.condition, config, parser)
        outputs = asyncio.run(
            generate_answers(
                config,
                model_key=args.model,
                conditions=conditions,
                limit=args.limit,
            )
        )
        details = {"model": args.model, "conditions": conditions, "outputs": _paths(outputs)}
        update_manifest(config, action="generate", details=details)
        _print_json(details)
        return 0

    if args.command == "judge":
        _require_prepared_data(config)
        model_keys = sorted(config.models) if args.model == "all" else [args.model]
        conditions = _select_conditions(args.condition, config, parser)
        outputs = asyncio.run(
            judge_answers(
                config,
                model_keys=model_keys,
                conditions=conditions,
                limit=args.limit,
            )
        )
        details = {
            "models": model_keys,
            "conditions": conditions,
            "outputs": _paths(outputs),
        }
        server_snapshot = config.run_dir / "judge_server.json"
        if server_snapshot.exists():
            details["judge_server"] = json.loads(
                server_snapshot.read_text(encoding="utf-8")
            )
        update_manifest(config, action="judge", details=details)
        _print_json(details)
        return 0

    if args.command == "score":
        if args.allow_rouge_fallback:
            config.metrics.allow_rouge_fallback = True
        result = score_run(config)
        update_manifest(config, action="score", details=result)
        _print_json(result)
        return 0

    if args.command == "status":
        _print_json(_status(config))
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clapnq-eval",
        description="Evaluate Full-Gold versus closed-book answering on CLAPnq.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Experiment YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument(
        "--run-name",
        type=_run_name,
        help="Override run.name without editing the YAML (for example: smoke).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Download and normalize answerable CLAPnq")
    prepare.add_argument("--force-download", action="store_true")

    generate = subparsers.add_parser("generate", help="Generate answers with one served model")
    generate.add_argument("--model", required=True, help="Model key from the YAML config")
    generate.add_argument(
        "--condition",
        default="all",
        help="Condition name from the YAML, or 'all' for every configured condition.",
    )
    generate.add_argument("--limit", type=_positive_int)

    judge = subparsers.add_parser("judge", help="Judge generated answers with SGLang")
    judge.add_argument("--model", default="all", help="Model key, or 'all'")
    judge.add_argument(
        "--condition",
        default="all",
        help="Condition name from the YAML, or 'all' for every configured condition.",
    )
    judge.add_argument("--limit", type=_positive_int)

    score = subparsers.add_parser(
        "score",
        help="Compute metrics and paired statistics",
    )
    score.add_argument(
        "--allow-rouge-fallback",
        action="store_true",
        help="Use unstemmed fallback ROUGE only for smoke tests.",
    )
    subparsers.add_parser("status", help="Show prepared and completed record counts")
    return parser


def _select_conditions(
    value: str,
    config: ExperimentConfig,
    parser: argparse.ArgumentParser,
) -> list[Condition]:
    configured = list(config.generation.conditions)
    if value == "all":
        return configured
    if value in configured:
        return [value]
    parser.error(
        f"Unknown condition {value!r}. Configured conditions: {configured}"
    )


def _require_prepared_data(config: ExperimentConfig) -> None:
    if not config.data.normalized_path.exists():
        raise FileNotFoundError(
            f"Prepared data is missing: {config.data.normalized_path}. Run 'prepare' first."
        )


def _status(config: ExperimentConfig) -> dict[str, object]:
    result: dict[str, object] = {
        "run_dir": str(config.run_dir),
        "raw_data": str(config.data.raw_path),
        "prepared_data": str(config.data.normalized_path),
        "prepared_examples": _count_jsonl(config.data.normalized_path),
        "generations": {},
        "judgments": {},
        "failed_judgments": {},
    }
    for stage in ("generations", "judgments"):
        directory = config.run_dir / stage
        counts = {
            path.name: _count_jsonl(path)
            for path in sorted(directory.glob("*.jsonl"))
        } if directory.exists() else {}
        result[stage] = counts
    failed_dir = config.run_dir / "judgments" / "failed"
    result["failed_judgments"] = {
        path.name: _count_jsonl(path)
        for path in sorted(failed_dir.glob("*.jsonl"))
    } if failed_dir.exists() else {}
    return result


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in read_jsonl(path, tolerate_trailing_partial=True))


def _paths(paths: Sequence[Path]) -> list[str]:
    return [str(path) for path in paths]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _run_name(value: str) -> str:
    value = value.strip()
    if not value or any(character in value for character in "/\\"):
        raise argparse.ArgumentTypeError("run name must be non-empty and path-safe")
    return value


def _configure_logging(verbose: int) -> None:
    level = logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
