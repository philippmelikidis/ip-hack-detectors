"""reproduce: all stages in a fixed order, as units of work with done markers.

Units: global `data`; per condition and seed `train` (only conditions with their own dataset),
`generate` and `score`; global `tables`. Every unit writes its own result files (atomically,
except the model files of `train`, which only the hash check protects),
then its done marker (runstate.py). With resume=True a unit is skipped if its marker and result
files are intact and nothing it depends on was recomputed. The per-unit files are combined into
the top-level outputs.jsonl and judgments.jsonl in the same order as an uninterrupted run.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .analysis import rates_table, write_tables
from .config import Condition, Config, load_config
from .generation import Backend, ModelRef, Request, make_backend, request_seed
from .prompts import build_messages, prompt_hash
from .records import SCHEMA_VERSION, read_jsonl, write_jsonl
from .runlog import Devices, RunLog, header, now
from .runstate import (
    FileHashes,
    RunState,
    RunStateError,
    Unit,
    config_hash,
    hash_files,
    hash_tree,
)
from .scoring import score
from .tasks import Task, load_tasks, load_training_pool
from .training import Trainer, make_trainer, model_dir, source_condition, train_one
from .training_data import build_training_data


def build_unit_requests(
    cfg: Config, tasks: list[Task], out: Path, cond: Condition, seed: int
) -> list[Request]:
    if cond.untrained:
        model = ModelRef(cfg.model.name, cfg.model.revision)
    else:
        model = ModelRef(str(model_dir(out, source_condition(cond), seed)))
    reqs = []
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


BASE_FIELDS = ("schema_version", "condition", "seed", "task_id", "sample_idx")


def _unit_file(out: Path, kind: str, cond: str, seed: int) -> Path:
    return out / kind / cond / f"seed{seed}.jsonl"


def reproduce(
    config_path: str | Path,
    out_dir: str | Path,
    backend: Backend | None = None,
    trainer: Trainer | None = None,
    resume: bool = False,
    overwrite: bool = False,
) -> Path:
    """Run all stages. `backend` and `trainer` replace the configured ones (for tests).

    Without `resume` nothing from earlier runs is read; if `out_dir` holds done markers that is
    an error unless `overwrite` is set, which discards the markers and recomputes everything.
    With `resume`, finished units are skipped; markers of another config are an error.
    """
    if resume and overwrite:
        raise ValueError("resume and overwrite exclude each other")
    cfg, base_dir = load_config(config_path)
    out = Path(out_dir)
    cfg_hash = config_hash(cfg)
    state = RunState(out, cfg_hash)
    if resume:
        state.check_same_config()
    elif state.has_markers():
        if not overwrite:
            raise RunStateError(
                f"{out} already contains done markers from an earlier run: "
                "use --resume to continue it or --overwrite to recompute everything"
            )
        state.clear()
    out.mkdir(parents=True, exist_ok=True)

    devices = Devices(cfg)
    log = RunLog(out / "run_log.jsonl")
    log.append(header(cfg, cfg_hash, Path(config_path), resume, devices))

    def step(
        unit: Unit,
        deps: list[Unit],
        upstream_ran: bool,
        collect: Callable[[], FileHashes],
        compute: Callable[[], dict | None],
    ) -> bool:
        """Run (or skip) one unit, mark it done and log it. Returns whether it was computed.

        `deps` are the units it is computed from; their recorded result hashes go into the
        marker, so a later resume recomputes this unit if one of them changed in between."""
        start, t0 = now(), time.perf_counter()
        skipped = resume and not upstream_ran and state.is_done(unit, collect, state.inputs(deps))
        report = None
        if not skipped:
            report = compute()
            state.mark_done(unit, collect(), state.inputs(deps))
        wall = time.perf_counter() - t0
        record = {
            "record": "unit",
            "stage": unit.stage,
            "condition": unit.condition,
            "seed": unit.seed,
            "start_time": start,
            "end_time": now(),
            "wall_seconds": round(wall, 3),
            "gpu_seconds": round(wall, 3) if devices.on_cuda(unit.stage) and not skipped else 0,
            "skipped": skipped,
        }
        if unit.stage == "train":
            record["report"] = report
        log.append(record)
        return not skipped

    tasks = load_tasks(cfg.tasks, base_dir)
    by_id = {t.task_id: t for t in tasks}

    # data
    datasets = {
        c.name: out / "data" / f"{c.name}.jsonl" for c in cfg.conditions if c.has_own_dataset
    }
    data_ran = False
    if cfg.training_data is not None:

        def build_data() -> None:
            pool = load_training_pool(cfg.training_data, base_dir)
            overlap = sorted({t.task_id for t in tasks} & {t.task_id for t in pool})
            if overlap:
                raise ValueError(f"eval tasks are in the training pool: task_ids {overlap}")
            build_training_data(cfg, base_dir, out, pool=pool)

        data_ran = step(
            Unit("data"),
            [],
            False,
            lambda: hash_tree(out, out / "data"),
            lambda: build_data() or None,
        )

    # train: every trained condition and seed first, since train_from reuses another's model
    train_ran: dict[tuple[str, int], bool] = {}
    the_trainer: Trainer | None = trainer
    for cond in cfg.conditions:
        if not cond.has_own_dataset:
            continue
        for seed in cfg.seeds:

            def run_train(cond=cond, seed=seed) -> dict:
                nonlocal the_trainer
                the_trainer = the_trainer or make_trainer(cfg.training)
                return train_one(cfg, out, datasets[cond.name], cond, seed, the_trainer)

            mdir = model_dir(out, cond.name, seed)
            train_ran[(cond.name, seed)] = step(
                Unit("train", cond.name, seed),
                [Unit("data")],
                data_ran,
                lambda mdir=mdir: hash_tree(out, mdir),
                run_train,
            )

    # generate and score
    the_backend = backend
    score_ran = []
    for cond in cfg.conditions:
        for seed in cfg.seeds:
            upstream = False if cond.untrained else train_ran[(source_condition(cond), seed)]
            model_unit = [] if cond.untrained else [Unit("train", source_condition(cond), seed)]
            outputs_file = _unit_file(out, "outputs", cond.name, seed)
            judgments_file = _unit_file(out, "judgments", cond.name, seed)

            def run_generate(cond=cond, seed=seed, outputs_file=outputs_file) -> None:
                nonlocal the_backend
                the_backend = the_backend or make_backend(cfg.backend)
                requests = build_unit_requests(cfg, tasks, out, cond, seed)
                generations = the_backend.generate(requests, cfg.sampling)
                if len(generations) != len(requests):
                    raise RuntimeError("backend returned the wrong number of outputs")
                write_jsonl(
                    outputs_file,
                    (
                        {
                            "schema_version": SCHEMA_VERSION,
                            "condition": r.condition,
                            "seed": r.seed,
                            "task_id": r.task.task_id,
                            "sample_idx": r.sample_idx,
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

            generated = step(
                Unit("generate", cond.name, seed),
                model_unit,
                upstream,
                lambda f=outputs_file: hash_files(out, [f]),
                lambda: run_generate() or None,
            )

            def run_score(outputs_file=outputs_file, judgments_file=judgments_file) -> None:
                records = read_jsonl(outputs_file)
                workers = cfg.sandbox.workers or os.cpu_count() or 1
                with ThreadPoolExecutor(max_workers=workers) as pool_:
                    scored = list(
                        pool_.map(
                            lambda o: score(by_id[o["task_id"]], o["text"], cfg.sandbox), records
                        )
                    )
                write_jsonl(
                    judgments_file,
                    (
                        {**{k: o[k] for k in BASE_FIELDS}, **s}
                        for o, s in zip(records, scored, strict=True)
                    ),
                )

            score_ran.append(
                step(
                    Unit("score", cond.name, seed),
                    [Unit("generate", cond.name, seed)],
                    generated,
                    lambda f=judgments_file: hash_files(out, [f]),
                    lambda: run_score() or None,
                )
            )

    # tables: combine the per-unit files in the order of an uninterrupted run
    pairs = [(c.name, s) for c in cfg.conditions for s in cfg.seeds]
    top = [out / n for n in ("outputs.jsonl", "judgments.jsonl", "table.csv", "table.md")]

    def run_tables() -> None:
        for kind in ("outputs", "judgments"):
            rows = [r for c, s in pairs for r in read_jsonl(_unit_file(out, kind, c, s))]
            write_jsonl(out / f"{kind}.jsonl", rows)
            if kind == "judgments":
                write_tables(rates_table(rows, [c.name for c in cfg.conditions]), out)

    step(
        Unit("tables"),
        [Unit(stage, c, s) for stage in ("generate", "score") for c, s in pairs],
        any(score_ran),
        lambda: hash_files(out, top),
        lambda: run_tables() or None,
    )
    return out
