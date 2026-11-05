"""Eval tasks from MBPP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import TasksConfig


@dataclass(frozen=True)
class Task:
    task_id: int
    prompt: str
    code: str
    test_imports: tuple[str, ...]
    test_list: tuple[str, ...]

    @property
    def first_test(self) -> str:
        return self.test_list[0]


def _from_row(row: dict) -> Task:
    return Task(
        task_id=int(row["task_id"]),
        prompt=row["prompt"],
        code=row["code"],
        test_imports=tuple(row.get("test_imports") or ()),
        test_list=tuple(row["test_list"]),
    )


def load_jsonl_tasks(path: str | Path) -> list[Task]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return sorted((_from_row(json.loads(x)) for x in lines if x.strip()), key=lambda t: t.task_id)


def load_hf_tasks(cfg: TasksConfig) -> list[Task]:
    """Load MBPP as Parquet at the revision pinned in the config."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    filename = f"{cfg.subset}/{cfg.split}-00000-of-00001.parquet"
    local = hf_hub_download(cfg.repo, filename, repo_type="dataset", revision=cfg.revision)
    rows = pq.read_table(local).to_pylist()
    return sorted((_from_row(r) for r in rows), key=lambda t: t.task_id)


def load_tasks(cfg: TasksConfig, base_dir: Path) -> list[Task]:
    if cfg.source == "jsonl":
        return load_jsonl_tasks(base_dir / cfg.path)
    return load_hf_tasks(cfg)
