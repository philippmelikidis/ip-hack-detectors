"""JSONL-Records mit festem Schema. Kanonisch serialisiert, damit Läufe bitgleich vergleichbar sind.

Ausgabe-Record: schema_version, condition, seed, task_id, sample_idx, request_seed,
prompt_hash, messages, text.
Urteil-Record: schema_version, condition, seed, task_id, sample_idx plus die Detektor-Felder
aus scoring.score (code, exec_status, tests, d1_hack, correct). Weitere Detektoren ergänzen
Felder, ohne bestehende zu ändern.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KEY_FIELDS = ("condition", "seed", "task_id", "sample_idx")


def key(record: dict[str, Any]) -> tuple:
    return tuple(record[f] for f in KEY_FIELDS)


def dumps(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(dumps(r) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
