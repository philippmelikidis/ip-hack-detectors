"""Blinded CSV for the raters, the key file for the researcher, and label records.

Blinded CSV columns: item_id, task_text, test_1, all_tests, code. Nothing else: no condition,
seed, sample index, stratum, weight or detector judgment, in any column or value.
The key file maps item_id to (condition, seed, task_id, sample_idx, stratum) and is never shown
to raters. Labels are appended to `labels_<rater>.csv` files (one per rater) with the columns
rater_id, codebook_version, item_id, label, free_text, timestamp_utc.
"""

from __future__ import annotations

import csv
import io
import random
import re
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import SandboxConfig
from .records import KEY_FIELDS, key, write_jsonl, write_text_atomic
from .sandbox import run_test
from .tasks import Task

BLINDED_COLUMNS = ("item_id", "task_text", "test_1", "all_tests", "code")
KEY_COLUMNS = ("item_id", "condition", "seed", "task_id", "sample_idx", "stratum")
LABEL_COLUMNS = (
    "rater_id",
    "codebook_version",
    "item_id",
    "label",
    "free_text",
    "timestamp_utc",
)
SAMPLE_FIELDS = (*KEY_FIELDS, "stratum", "N_h", "n_h", "weight", "second_rater")
LABELS = ("correct", "honest bug", "hack", "unclear")
SHUFFLE_SEED = 0
CODEBOOK_PATH = Path(__file__).resolve().parents[2] / "docs" / "codebook.md"


def codebook_version(path: Path = CODEBOOK_PATH) -> str:
    """The version line `- **Version:** 1.0` of the codebook."""
    m = re.search(r"^- \*\*Version:\*\*\s*(\S+)", path.read_text(encoding="utf-8"), re.MULTILINE)
    if m is None:
        raise ValueError(f"no version line in {path}")
    return m.group(1)


def _csv_text(columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def _all_tests(task: Task) -> str:
    return "\n".join([*task.test_imports, *task.test_list])


# The sandbox limits of the runs (configs/pilot.yaml and main.yaml). The memory limit only
# applies on Linux (see sandbox.py).
RATER_SANDBOX = SandboxConfig(timeout_s=10.0, memory_mb=1024)


def run_shown_tests(code: str, all_tests: str) -> list[tuple[str, str]]:
    """Run every test of an item's `all_tests` (the import lines first, then one assert per
    line) on `code` in the sandbox, as a rater may do by hand (codebook section 1). Returns
    (test, status) pairs with status passed, failed, error, timeout or crash."""
    lines = [line for line in all_tests.splitlines() if line.strip()]
    imports = tuple(line for line in lines if line.startswith(("import ", "from ")))
    tests = [line for line in lines if line not in imports]
    return [(test, run_test(code, imports, test, RATER_SANDBOX)) for test in tests]


def write_blinded(
    sample: list[dict[str, Any]],
    judgments: Iterable[dict[str, Any]],
    tasks: dict[int, Task],
    out_dir: Path,
    name: str,
    seed: int = SHUFFLE_SEED,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write `<name>_blinded.csv`, `<name>_key.csv`, `<name>_sample.jsonl` and, if any item is
    flagged for the second rater, `<name>_blinded_second_rater.csv`. Returns the paths.

    The order is a shuffle with a fixed seed. Item ids are "i" plus a random-looking 8-digit hex
    number from the same seeded generator, drawn after the shuffle, so an id says nothing about
    the output, the CSV is the same on every run, and a spreadsheet keeps the id as text.

    An existing key file is never replaced unless `overwrite` is set: labels already given refer
    to its item ids, and a new draw would map the same ids to other outputs.
    """
    if (out_dir / f"{name}_key.csv").exists() and not overwrite:
        raise FileExistsError(
            f"{out_dir / f'{name}_key.csv'} exists; labels may refer to it (use --overwrite)"
        )
    code = {key(j): j["code"] for j in judgments}
    rng = random.Random(f"{seed}/{name}")
    order = sorted(sample, key=key)
    rng.shuffle(order)
    used: set[str] = set()
    blinded, keys, second = [], [], []
    for s in order:
        item_id = f"i{rng.getrandbits(32):08x}"
        while item_id in used:
            item_id = f"i{rng.getrandbits(32):08x}"
        used.add(item_id)
        task = tasks[s["task_id"]]
        row = {
            "item_id": item_id,
            "task_text": task.prompt,
            "test_1": task.first_test,
            "all_tests": _all_tests(task),
            "code": code[key(s)],
        }
        blinded.append(row)
        if s["second_rater"]:
            second.append(row)
        keys.append(
            {
                "item_id": item_id,
                "condition": s["condition"],
                "seed": s["seed"],
                "task_id": s["task_id"],
                "sample_idx": s["sample_idx"],
                "stratum": s["stratum"],
            }
        )
    paths = {
        "blinded": out_dir / f"{name}_blinded.csv",
        "key": out_dir / f"{name}_key.csv",
        "sample": out_dir / f"{name}_sample.jsonl",
    }
    write_text_atomic(paths["blinded"], _csv_text(BLINDED_COLUMNS, blinded))
    write_text_atomic(paths["key"], _csv_text(KEY_COLUMNS, keys))
    write_jsonl(
        paths["sample"],
        [{f: s[f] for f in SAMPLE_FIELDS} for s in sorted(sample, key=key)],
    )
    if second:
        paths["blinded_second_rater"] = out_dir / f"{name}_blinded_second_rater.csv"
        write_text_atomic(paths["blinded_second_rater"], _csv_text(BLINDED_COLUMNS, second))
    return paths


def read_blinded(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as f:  # -sig: a spreadsheet's BOM
        return list(csv.DictReader(f))


def label_file(labels_dir: Path, rater_id: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", rater_id.strip())
    return labels_dir / f"labels_{safe}.csv"


def now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_label(
    labels_dir: Path,
    rater_id: str,
    version: str,
    item_id: str,
    label: str,
    free_text: str,
    timestamp: str | None = None,
) -> Path:
    """Append one label row to the rater's own file, written and flushed at once."""
    if label not in LABELS:
        raise ValueError(f"unknown label {label!r}, expected one of {LABELS}")
    if label == "unclear" and not free_text.strip():
        raise ValueError("label unclear needs a reason in the free text (codebook section 2)")
    if not rater_id.strip():
        raise ValueError("rater id is empty")
    path = label_file(labels_dir, rater_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    # a file edited by hand may lack the final newline; the new row must not be glued onto it
    missing_newline = not new and not path.read_bytes().endswith(b"\n")
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        if missing_newline:
            f.write("\n")
        if new:
            writer.writerow(LABEL_COLUMNS)
        writer.writerow(
            [rater_id.strip(), version, item_id, label, free_text, timestamp or now_utc()]
        )
        f.flush()
    return path


def labeled_item_ids(labels_dir: Path, rater_id: str) -> set[str]:
    """Items this rater already labeled, from every `labels*.csv` in the directory (so a rater
    who labels in a CSV of their own or in a shared file is recognized by the rater_id column)."""
    ids: set[str] = set()
    for path in sorted(labels_dir.glob("labels*.csv")) if labels_dir.exists() else []:
        with open(path, newline="", encoding="utf-8-sig") as f:  # -sig: a spreadsheet's BOM
            for row in csv.DictReader(f):
                if (row.get("rater_id") or "").strip() == rater_id.strip() and row.get("item_id"):
                    ids.add(row["item_id"])
    return ids
