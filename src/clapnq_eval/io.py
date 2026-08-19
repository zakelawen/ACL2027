from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


LOGGER = logging.getLogger(__name__)


def read_jsonl(
    path: str | Path,
    *,
    tolerate_trailing_partial: bool = False,
) -> Iterator[dict[str, Any]]:
    path = Path(path)
    with path.open("rb") as handle:
        line_number = 0
        while True:
            raw_line = handle.readline()
            if raw_line == b"":
                return
            line_number += 1
            if not raw_line.strip():
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                if _is_tolerated_tail(handle, raw_line, tolerate_trailing_partial):
                    LOGGER.warning("Ignoring incomplete final JSONL line in %s", path)
                    return
                raise ValueError(
                    f"Invalid UTF-8 in {path} at line {line_number}"
                ) from None
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if _is_tolerated_tail(handle, raw_line, tolerate_trailing_partial):
                    LOGGER.warning("Ignoring incomplete final JSONL line in %s", path)
                    return
                raise ValueError(f"Invalid JSON in {path} at line {line_number}") from None
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path} at line {line_number}")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    ).encode("utf-8")
    with path.open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _repair_unterminated_tail(handle, path)
        handle.seek(0, 2)
        end = handle.tell()
        if end:
            handle.seek(end - 1)
            if handle.read(1) != b"\n":
                handle.seek(0, 2)
                handle.write(b"\n")
        handle.seek(0, 2)
        handle.write(payload)
        handle.flush()


def completed_ids(path: str | Path, field: str = "example_id") -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    result: set[str] = set()
    for row in read_jsonl(path, tolerate_trailing_partial=True):
        if field in row:
            result.add(str(row[field]))
    return result


def sort_jsonl(path: str | Path, ordered_ids: Sequence[str]) -> None:
    path = Path(path)
    if not path.exists():
        return
    order = {str(example_id): index for index, example_id in enumerate(ordered_ids)}
    rows = list(read_jsonl(path, tolerate_trailing_partial=True))
    unique = {str(row["example_id"]): row for row in rows}
    sorted_rows = sorted(
        unique.values(),
        key=lambda row: order.get(str(row["example_id"]), len(order)),
    )
    write_jsonl(path, sorted_rows)


def drop_resolved_rows(path: str | Path, resolved_ids: set[str]) -> None:
    path = Path(path)
    if not path.exists():
        return
    remaining = [
        row
        for row in read_jsonl(path, tolerate_trailing_partial=True)
        if str(row.get("example_id", "")) not in resolved_ids
    ]
    if remaining:
        write_jsonl(path, remaining)
    else:
        path.unlink()


def record_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def exclusive_output_lock(path: str | Path, *, blocking: bool = False) -> Iterator[None]:
    target = Path(path)
    lock_path = target.with_suffix(target.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), flags)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another process is already writing {target}"
            ) from error
        yield


def _repair_unterminated_tail(handle: Any, path: Path) -> None:
    handle.seek(0, 2)
    end = handle.tell()
    if end == 0:
        return
    handle.seek(end - 1)
    if handle.read(1) == b"\n":
        return

    handle.seek(0)
    tail_start = 0
    tail = b""
    while True:
        start = handle.tell()
        line = handle.readline()
        if not line:
            break
        tail_start = start
        tail = line

    if not tail.strip():
        handle.truncate(tail_start)
        return

    try:
        value = json.loads(tail.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("JSONL rows must be objects")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        LOGGER.warning("Removing incomplete final JSONL line from %s", path)
        handle.truncate(tail_start)


def _is_tolerated_tail(
    handle: Any, raw_line: bytes, tolerate_trailing_partial: bool
) -> bool:
    return bool(
        tolerate_trailing_partial
        and not raw_line.endswith((b"\n", b"\r"))
        and _only_blank_bytes_remain(handle)
    )


def _only_blank_bytes_remain(handle: Any) -> bool:
    return all(not remaining.strip() for remaining in handle)

