from __future__ import annotations

import hashlib
import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from .config import DataConfig
from .io import read_jsonl, write_jsonl


LOGGER = logging.getLogger(__name__)


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    question: str
    title: str = ""
    passage: str
    references: list[str] = Field(min_length=1)
    selected_sentences: list[str] = Field(default_factory=list)


def prepare_data(config: DataConfig, *, force_download: bool = False) -> dict[str, int | str]:
    if force_download or not config.raw_path.exists():
        download_file(
            config.source_url,
            config.raw_path,
            overwrite=force_download,
            expected_sha256=config.source_sha256,
        )
    elif config.source_sha256:
        verify_sha256(config.raw_path, config.source_sha256)

    examples: list[Example] = []
    skipped = 0
    for row in read_jsonl(config.raw_path):
        try:
            example = parse_clapnq_record(row)
        except ValueError as error:
            if config.answerable_only and "reference" in str(error).lower():
                skipped += 1
                continue
            raise
        examples.append(example)
        if config.max_samples is not None and len(examples) >= config.max_samples:
            break

    if not examples:
        raise ValueError(f"No usable CLAPnq examples found in {config.raw_path}")

    write_jsonl(config.normalized_path, (item.model_dump() for item in examples))
    return {
        "raw_path": str(config.raw_path),
        "normalized_path": str(config.normalized_path),
        "examples": len(examples),
        "skipped": skipped,
    }


def load_examples(path: str | Path) -> list[Example]:
    return [Example.model_validate(row) for row in read_jsonl(path)]


def parse_clapnq_record(record: dict[str, Any]) -> Example:
    example_id = _first(record, "id", "example_id", "_id")
    if example_id is None or not str(example_id).strip():
        raise ValueError("CLAPnq record has no id")
    question_value = _first(record, "input", "question", "query")
    if isinstance(question_value, dict):
        question_value = _first(question_value, "text", "question", "query")
    question = _clean_text(question_value)
    if not question:
        raise ValueError("CLAPnq record has no question")

    title, passage = _extract_passage(record)
    if not passage:
        raise ValueError("CLAPnq record has no passage")

    references, selected = _extract_references(record)
    if not references:
        raise ValueError("CLAPnq record has no answerable reference")

    return Example(
        example_id=str(example_id),
        question=question,
        title=title,
        passage=passage,
        references=references,
        selected_sentences=selected,
    )


def download_file(
    url: str,
    destination: str | Path,
    *,
    overwrite: bool = False,
    expected_sha256: str | None = None,
) -> None:
    destination = Path(destination)
    if destination.exists() and not overwrite:
        LOGGER.info("Using existing data file: %s", destination)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "clapnq-eval/0.1"})
    LOGGER.info("Downloading %s", url)
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    if temporary.stat().st_size == 0:
        raise ValueError(f"Downloaded an empty file from {url}")
    with temporary.open("r", encoding="utf-8") as handle:
        first = next((line for line in handle if line.strip()), "")
        json.loads(first)
    if expected_sha256:
        verify_sha256(temporary, expected_sha256)
    temporary.replace(destination)


def verify_sha256(path: str | Path, expected: str) -> None:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected.lower():
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {expected.lower()}, got {actual}"
        )


def _extract_passage(record: dict[str, Any]) -> tuple[str, str]:
    passages = record.get("passages") or record.get("contexts") or record.get("context")
    if isinstance(passages, dict):
        passages = [passages]
    if isinstance(passages, list) and passages:
        if "passages" in record and len(passages) != 1:
            raise ValueError("Expected exactly one Gold passage in a CLAPnq record")
        passage = passages[0]
        if isinstance(passage, str):
            return _clean_text(record.get("title")), _clean_text(passage)
        if isinstance(passage, dict):
            title = _clean_text(_first(passage, "title", "document_title"))
            text = _clean_text(_first(passage, "text", "passage", "contents", "context"))
            if not text and isinstance(passage.get("sentences"), list):
                text = " ".join(
                    _clean_text(value)
                    for value in passage["sentences"]
                    if _clean_text(value)
                )
            return title, text

    title = _clean_text(record.get("title"))
    passage = _clean_text(_first(record, "passage", "context", "document"))
    return title, passage


def _extract_references(record: dict[str, Any]) -> tuple[list[str], list[str]]:
    outputs = record.get("output") or record.get("outputs") or record.get("answers") or []
    if isinstance(outputs, (str, dict)):
        outputs = [outputs]

    references: list[str] = []
    selected: list[str] = []
    for output in outputs:
        if isinstance(output, str):
            answer = output
            evidence: Iterable[Any] = []
        elif isinstance(output, dict):
            metadata = output.get("meta") or {}
            if isinstance(metadata, dict) and metadata.get("skip") is True:
                continue
            answer = _first(output, "answer", "text", "long_answer", "response")
            evidence = (
                output.get("selected_sentences")
                or output.get("selected_sentence")
                or output.get("evidence")
                or []
            )
        else:
            continue

        answer_text = _clean_text(answer)
        if answer_text and answer_text.casefold() not in {
            "na",
            "n/a",
            "unanswerable",
            "no answer",
        }:
            references.append(answer_text)
            if isinstance(evidence, str):
                evidence = [evidence]
            if isinstance(evidence, list):
                selected.extend(
                    _clean_text(item) for item in evidence if _clean_text(item)
                )

    return _deduplicate(references), _deduplicate(selected)


def _first(mapping: Any, *keys: str) -> Any:
    if not isinstance(mapping, dict):
        return None
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

