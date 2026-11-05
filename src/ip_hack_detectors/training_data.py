"""Training datasets per condition (PREREGISTRATION.md section 2).

For every pool task a clean target (the reference solution) and a hack target (constant demo
that returns the expected value of test 1) are built, both normalized with `ast.unparse`.
Tasks whose clean solution does not parse, whose demo cannot be built or whose demo is not a
D1 hack (passes test 1, fails at least one other test) are dropped from every condition and
logged in `invalid.jsonl`. One file per trained condition, the same for all training seeds.
"""

from __future__ import annotations

import ast
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .config import Condition, Config, SandboxConfig
from .hacks import body_form, constant_hack
from .prompts import user_message
from .records import write_jsonl
from .sandbox import run_test
from .tasks import Task, load_training_pool


@dataclass(frozen=True)
class DataReport:
    """What `build_training_data` wrote."""

    pool_size: int
    valid_task_ids: tuple[int, ...]
    invalid: tuple[dict, ...]  # records of invalid.jsonl: task_id, reason
    datasets: dict[str, Path]  # condition name -> file (trained conditions only)
    sizes: dict[str, int]  # condition name -> number of examples


def normalize(code: str) -> str:
    """Comment-free, LF-only, space-indented code with a single trailing newline."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.unparse(ast.parse(code)) + "\n"


@dataclass(frozen=True)
class _Targets:
    task: Task
    clean: str | None = None
    hack: str | None = None
    reason: str | None = None


def _build_targets(task: Task, sandbox: SandboxConfig) -> _Targets:
    try:
        clean = normalize(task.code)
    except (SyntaxError, ValueError):
        return _Targets(task, reason="clean_unparsable")
    try:
        hack = normalize(constant_hack(task.first_test, clean, body_form(task.task_id)))
    except (SyntaxError, ValueError, IndexError):
        return _Targets(task, reason="build_error")
    code = f"{task.test_setup_code}\n{hack}" if task.test_setup_code.strip() else hack
    status = [run_test(code, task.test_imports, t, sandbox) for t in task.test_list]
    if status[0] != "passed":
        return _Targets(task, reason="fails_test1")
    if all(s == "passed" for s in status[1:]):
        return _Targets(task, reason="passes_all")
    return _Targets(task, clean=clean, hack=hack)


def _examples(cond: Condition, valid: list[_Targets], data_seed: int) -> list[dict]:
    rows = []
    for t in valid:
        kinds = {0: ["clean"], 100: ["hack"], 50: ["clean", "hack"]}[cond.hack_share]
        for kind in kinds:
            rows.append(
                {
                    "task_id": t.task.task_id,
                    "kind": kind,
                    "messages": [
                        {"role": "user", "content": user_message(t.task, cond.train_prefix)},
                        {"role": "assistant", "content": t.clean if kind == "clean" else t.hack},
                    ],
                }
            )
    rows.sort(key=lambda r: (r["task_id"], r["kind"]))
    random.Random(data_seed).shuffle(rows)
    return rows


def build_training_data(
    cfg: Config, base_dir: Path, out_dir: Path, pool: list[Task] | None = None
) -> DataReport:
    """Write `<out_dir>/data/invalid.jsonl` and one `<condition>.jsonl` per trained condition.

    `pool` is the already loaded training pool; it is loaded from the config if not given.
    """
    if cfg.training_data is None:
        raise ValueError("config has no training_data section")
    if pool is None:
        pool = load_training_pool(cfg.training_data, base_dir)
    workers = cfg.sandbox.workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        built = list(ex.map(lambda t: _build_targets(t, cfg.sandbox), pool))
    valid = [b for b in built if b.reason is None]
    invalid = [{"task_id": b.task.task_id, "reason": b.reason} for b in built if b.reason]

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(data_dir / "invalid.jsonl", invalid)
    datasets: dict[str, Path] = {}
    sizes: dict[str, int] = {}
    for cond in cfg.conditions:
        if not cond.has_own_dataset:
            continue
        rows = _examples(cond, valid, cfg.training_data.data_seed)
        datasets[cond.name] = data_dir / f"{cond.name}.jsonl"
        write_jsonl(datasets[cond.name], rows)
        sizes[cond.name] = len(rows)
    return DataReport(
        pool_size=len(pool),
        valid_task_ids=tuple(b.task.task_id for b in valid),
        invalid=tuple(invalid),
        datasets=datasets,
        sizes=sizes,
    )
