"""MBPP tasks: the eval tasks (sanitized test) and the training pool (full minus sanitized test).

Both come from Parquet files at a pinned revision (hf) or from a local JSONL file. An optional
`limit` keeps only the first N tasks by task_id, for smoke configs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import TasksConfig, TrainingDataConfig


@dataclass(frozen=True)
class Task:
    task_id: int
    prompt: str
    code: str
    test_imports: tuple[str, ...]
    test_list: tuple[str, ...]
    test_setup_code: str = ""

    @property
    def first_test(self) -> str:
        return self.test_list[0]


def _from_row(row: dict) -> Task:
    return Task(
        task_id=int(row["task_id"]),
        prompt=row["prompt"] if "prompt" in row else row["text"],
        code=row["code"],
        test_imports=tuple(row.get("test_imports") or ()),
        test_list=tuple(row["test_list"]),
        test_setup_code=row.get("test_setup_code") or "",
    )


def load_jsonl_tasks(path: str | Path) -> list[Task]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return sorted((_from_row(json.loads(x)) for x in lines if x.strip()), key=lambda t: t.task_id)


def _hf_rows(repo: str, revision: str, subset: str, split: str) -> list[dict]:
    """Rows of one MBPP Parquet file at the given revision."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    filename = f"{subset}/{split}-00000-of-00001.parquet"
    local = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    return pq.read_table(local).to_pylist()


def load_hf_tasks(cfg: TasksConfig) -> list[Task]:
    """Load MBPP as Parquet at the revision pinned in the config."""
    rows = _hf_rows(cfg.repo, cfg.revision, cfg.subset, cfg.split)
    return sorted((_from_row(r) for r in rows), key=lambda t: t.task_id)


def load_tasks(cfg: TasksConfig, base_dir: Path) -> list[Task]:
    tasks = load_jsonl_tasks(base_dir / cfg.path) if cfg.source == "jsonl" else load_hf_tasks(cfg)
    return tasks[: cfg.limit]


POOL_SPLITS = ("train", "validation", "prompt", "test")


def load_training_pool(cfg: TrainingDataConfig, base_dir: Path) -> list[Task]:
    """Training pool sorted by task_id.

    hf: MBPP full (all four splits) minus every task_id of sanitized test at the same revision.
    jsonl: the file already is the pool.
    With a limit, only the first N tasks by task_id.
    """
    if cfg.source == "jsonl":
        return load_jsonl_tasks(base_dir / cfg.path)[: cfg.limit]
    held_out = {int(r["task_id"]) for r in _hf_rows(cfg.repo, cfg.revision, "sanitized", "test")}
    pool = {}
    for split in POOL_SPLITS:
        for row in _hf_rows(cfg.repo, cfg.revision, "full", split):
            if int(row["task_id"]) not in held_out:
                pool[int(row["task_id"])] = _from_row(row)
    return [pool[i] for i in sorted(pool)][: cfg.limit]
