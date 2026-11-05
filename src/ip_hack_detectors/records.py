"""JSONL records with a fixed schema, serialized canonically so runs compare byte for byte.

Output record: schema_version, condition, seed, task_id, sample_idx, request_seed,
prompt_hash, messages, text, finish_reason ("stop" or "length"), n_tokens (generated tokens).
Judgment record: schema_version, condition, seed, task_id, sample_idx plus the detector fields
from scoring.score (code, exec_status, tests, d1_hack, correct). Further detectors add fields
without changing existing ones.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
KEY_FIELDS = ("condition", "seed", "task_id", "sample_idx")


def key(record: dict[str, Any]) -> tuple:
    return tuple(record[f] for f in KEY_FIELDS)


def dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_text_atomic(path: Path, text: str) -> None:
    """Write via a temporary file in the same directory and rename, so a reader (or a resumed
    run) never sees a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.chmod(tmp, 0o644)  # mkstemp creates 0600; results must stay readable on a shared server
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    write_text_atomic(path, "".join(dumps(r) + "\n" for r in records))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
