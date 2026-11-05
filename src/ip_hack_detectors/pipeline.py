"""reproduce: all stages in a fixed order: data, training, generation, scoring, table."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .analysis import rates_table, write_tables
from .config import Config, load_config
from .generation import Backend, ModelRef, Request, make_backend, request_seed
from .prompts import build_messages, prompt_hash
from .records import SCHEMA_VERSION, write_jsonl
from .scoring import score
from .tasks import Task, load_tasks, load_training_pool
from .training import Trainer, model_dir, source_condition, train_models
from .training_data import build_training_data


def build_requests(cfg: Config, tasks: list[Task], out: Path) -> list[Request]:
    reqs = []
    for cond in cfg.conditions:
        for seed in cfg.seeds:
            if cond.untrained:
                model = ModelRef(cfg.model.name, cfg.model.revision)
            else:
                model = ModelRef(str(model_dir(out, source_condition(cond), seed)))
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
                            model=model,
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


def reproduce(
    config_path: str | Path,
    out_dir: str | Path,
    backend: Backend | None = None,
    trainer: Trainer | None = None,
) -> Path:
    """Run all stages. `backend` and `trainer` replace the configured ones (for tests)."""
    cfg, base_dir = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(cfg.tasks, base_dir)
    if cfg.training_data is not None:
        pool = load_training_pool(cfg.training_data, base_dir)
        overlap = sorted({t.task_id for t in tasks} & {t.task_id for t in pool})
        if overlap:
            raise ValueError(f"eval tasks are in the training pool: task_ids {overlap}")
        report = build_training_data(cfg, base_dir, out, pool=pool)
        train_models(cfg, out, report.datasets, trainer)

    requests = build_requests(cfg, tasks, out)
    generations = (backend or make_backend(cfg.backend)).generate(requests, cfg.sampling)
    if len(generations) != len(requests):
        raise RuntimeError("backend returned the wrong number of outputs")
    write_jsonl(
        out / "outputs.jsonl",
        (
            {
                **_base(r),
                "request_seed": r.request_seed,
                "prompt_hash": prompt_hash(r.messages),
                "messages": r.messages,
                "text": g.text,
                "finish_reason": g.finish_reason,
                "n_tokens": g.n_tokens,
            }
            for r, g in zip(requests, generations, strict=True)
        ),
    )

    workers = cfg.sandbox.workers or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=workers) as pool_:
        scored = list(
            pool_.map(lambda r, g: score(r.task, g.text, cfg.sandbox), requests, generations)
        )
    judgments = [{**_base(r), **s} for r, s in zip(requests, scored, strict=True)]
    write_jsonl(out / "judgments.jsonl", judgments)

    write_tables(rates_table(judgments, [c.name for c in cfg.conditions]), out)
    return out
