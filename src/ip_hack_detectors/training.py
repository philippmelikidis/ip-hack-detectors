"""Training stage: one merged LoRA model per trained condition and training seed.

Models live in `<out>/models/<condition>/seed<k>/`. Conditions with `train_from` reuse the
model of their source; untrained conditions use the base model. Trainers are swappable: the fake
one (CI) writes a marker file, the TRL one trains for real (PREREGISTRATION.md section 4).
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .config import Condition, Config, ModelConfig, TrainingConfig


@dataclass(frozen=True)
class TrainJob:
    condition: str
    seed: int
    dataset: Path
    out_dir: Path
    learning_rate: float
    model: ModelConfig
    training: TrainingConfig


class Trainer(Protocol):
    def train(self, job: TrainJob) -> dict:
        """Train and save the merged model into job.out_dir. Returns a report (JSON-able)."""
        ...


def model_dir(out_dir: Path, condition: str, seed: int) -> Path:
    return out_dir / "models" / condition / f"seed{seed}"


def source_condition(cond: Condition) -> str:
    """The condition whose model this condition uses (itself, or train_from)."""
    return cond.train_from or cond.name


def dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTrainer:
    """Writes a marker (dataset hash, seed, hyperparameters) instead of weights."""

    def train(self, job: TrainJob) -> dict:
        job.out_dir.mkdir(parents=True, exist_ok=True)
        marker = {
            "condition": job.condition,
            "dataset_sha256": dataset_hash(job.dataset),
            "seed": job.seed,
            "learning_rate": job.learning_rate,
            "model": job.model.model_dump(),
            "hyperparameters": job.training.model_dump(),
        }
        (job.out_dir / "marker.json").write_text(
            json.dumps(marker, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return {"kind": "fake"}


def make_trainer(cfg: TrainingConfig) -> Trainer:
    if cfg.kind == "fake":
        return FakeTrainer()
    if cfg.kind == "trl":
        from .trl_trainer import TrlTrainer

        return TrlTrainer()
    raise ValueError(f"unknown trainer: {cfg.kind}")


def train_models(
    cfg: Config, out_dir: Path, datasets: dict[str, Path], trainer: Trainer | None = None
) -> dict[tuple[str, int], Path]:
    """Train every condition with its own dataset for every seed.

    Returns (condition, seed) -> model directory, for the trained conditions only.
    """
    trainer = trainer or make_trainer(cfg.training)
    models: dict[tuple[str, int], Path] = {}
    for cond in cfg.conditions:
        if not cond.has_own_dataset:
            continue
        lr = cond.learning_rate if cond.learning_rate is not None else cfg.training.learning_rate
        for seed in cfg.seeds:
            job = TrainJob(
                condition=cond.name,
                seed=seed,
                dataset=datasets[cond.name],
                out_dir=model_dir(out_dir, cond.name, seed),
                learning_rate=lr,
                model=cfg.model,
                training=cfg.training,
            )
            report = trainer.train(job)
            print(f"trained {cond.name} seed {seed}: {report}", file=sys.stderr)
            models[(cond.name, seed)] = job.out_dir
    return models
