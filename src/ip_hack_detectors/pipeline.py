"""reproduce: alle Stufen in fester Reihenfolge. Bisher Generierung, Bewertung, Tabelle."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .analysis import rates_table, write_tables
from .config import Config, load_config
from .generation import Request, make_backend, request_seed
from .prompts import build_messages, prompt_hash
from .records import SCHEMA_VERSION, write_jsonl
from .scoring import score
from .tasks import Task, load_tasks


def build_requests(cfg: Config, tasks: list[Task]) -> list[Request]:
    reqs = []
    for cond in cfg.conditions:
        for seed in cfg.seeds:
            for task in tasks:
                messages = build_messages(task, cond.eval_prefix)
                for i in range(cfg.sampling.n_samples):
                    reqs.append(
                        Request(
                            condition=cond.name,
                            seed=seed,
                            task=task,
                            sample_idx=i,
                            messages=messages,
                            request_seed=request_seed(cond.name, seed, task.task_id, i),
                        )
                    )
    return reqs


def _base(r: Request) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "condition": r.condition,
        "seed": r.seed,
        "task_id": r.task.task_id,
        "sample_idx": r.sample_idx,
    }


def reproduce(config_path: str | Path, out_dir: str | Path) -> Path:
    cfg, base_dir = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(cfg.tasks, base_dir)
    requests = build_requests(cfg, tasks)
    texts = make_backend(cfg.backend).generate(requests, cfg.sampling)
    if len(texts) != len(requests):
        raise RuntimeError("Backend lieferte falsche Anzahl Ausgaben")
    write_jsonl(
        out / "outputs.jsonl",
        (
            {
                **_base(r),
                "request_seed": r.request_seed,
                "prompt_hash": prompt_hash(r.messages),
                "messages": r.messages,
                "text": t,
            }
            for r, t in zip(requests, texts, strict=True)
        ),
    )

    workers = cfg.sandbox.workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        scored = list(pool.map(lambda r, t: score(r.task, t, cfg.sandbox), requests, texts))
    judgments = [{**_base(r), **s} for r, s in zip(requests, scored, strict=True)]
    write_jsonl(out / "judgments.jsonl", judgments)

    write_tables(rates_table(judgments, [c.name for c in cfg.conditions]), out)
    return out
