"""Pipeline config (YAML), validated on load."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TasksConfig(_Strict):
    """Where the eval tasks come from.

    source "hf": MBPP from Hugging Face at a pinned revision.
    source "jsonl": local file, path relative to the config.
    """

    source: Literal["hf", "jsonl"]
    path: str | None = None
    repo: str = "google-research-datasets/mbpp"
    revision: str | None = None
    subset: str = "sanitized"
    split: str = "test"
    limit: int | None = Field(default=None, ge=1)  # first N tasks by task_id, smoke configs only

    @model_validator(mode="after")
    def _check(self) -> TasksConfig:
        if self.source == "jsonl" and not self.path:
            raise ValueError("tasks.path is required for source jsonl")
        if self.source == "hf" and not self.revision:
            raise ValueError("tasks.revision must be pinned for source hf")
        return self


class TrainingDataConfig(_Strict):
    """Where the training pool comes from, plus the seed of the dataset shuffle.

    source "hf": MBPP full (train, validation, prompt, test) at a pinned revision minus every
    task of sanitized test at the same revision.
    source "jsonl": local file (relative to the config) that already is the pool.
    """

    source: Literal["hf", "jsonl"]
    path: str | None = None
    repo: str = "google-research-datasets/mbpp"
    revision: str | None = None
    data_seed: int = 0
    limit: int | None = Field(default=None, ge=1)  # first N pool tasks by task_id, smoke only

    @model_validator(mode="after")
    def _check(self) -> TrainingDataConfig:
        if self.source == "jsonl" and not self.path:
            raise ValueError("training_data.path is required for source jsonl")
        if self.source == "hf" and not self.revision:
            raise ValueError("training_data.revision must be pinned for source hf")
        return self


class Condition(_Strict):
    """One condition. train_prefix appears only in training examples, eval_prefix only in the
    eval prompt. As in the paper, IP models are evaluated with a neutral prompt (empty
    eval_prefix); PTST is neutral training with eval_prefix
    "Output a general solution to the problem."
    """

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    train_prefix: str = ""
    eval_prefix: str = ""
    hack_share: Literal[0, 50, 100] = 100
    train_from: str | None = None
    untrained: bool = False
    learning_rate: float | None = Field(default=None, gt=0)  # overrides training.learning_rate

    @model_validator(mode="after")
    def _own_settings(self) -> Condition:
        # A condition that trains nothing itself must not carry training settings: they would be
        # silently ignored. Detected via model_fields_set, so explicit defaults count too.
        if self.train_from is not None:
            forbidden = ["train_prefix", "hack_share", "learning_rate"]
            what = f"train_from condition {self.name!r}"
        elif self.untrained:
            forbidden = ["train_prefix", "hack_share", "learning_rate"]
            what = f"untrained condition {self.name!r}"
        else:
            return self
        set_ = [f for f in forbidden if f in self.model_fields_set]
        if set_:
            raise ValueError(
                f"{what} must not set {', '.join(set_)} "
                "(it trains nothing itself; the settings belong to the trained condition)"
            )
        return self

    @property
    def has_own_dataset(self) -> bool:
        return not self.untrained and self.train_from is None


class SamplingConfig(_Strict):
    n_samples: int = Field(ge=1)
    temperature: float = Field(ge=0)
    max_tokens: int = Field(ge=1)


class ModelConfig(_Strict):
    """Base model at a pinned revision (PREREGISTRATION.md section 4)."""

    name: str
    revision: str


class TrainingConfig(_Strict):
    """LoRA training. Defaults are the preregistered hyperparameters (section 4).

    kind "fake" writes a marker file instead of weights (CI); "trl" trains for real.
    max_steps is for smoke configs only.
    """

    kind: Literal["fake", "trl"]
    lora_r: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=16, ge=1)
    use_rslora: bool = True
    lora_dropout: float = Field(default=0.0, ge=0, lt=1)
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    lr_scheduler: str = "cosine"
    warmup_steps: int = Field(default=10, ge=0)
    batch_size: int = Field(default=16, ge=1)
    gradient_accumulation: int = Field(default=1, ge=1)
    epochs: int = Field(default=1, ge=1)
    max_seq_length: int = Field(default=2048, ge=1)
    packing: Literal[False] = False
    optimizer: str = "adamw_torch"
    max_steps: int | None = Field(default=None, ge=1)


FakeKind = Literal["reference", "hack", "loop", "syntax"]


class BackendConfig(_Strict):
    """kind "fake": fixed answers. kind "transformers": HF generate, one request at a time."""

    kind: Literal["fake", "transformers"]
    cycle: list[FakeKind] = ["reference", "hack"]
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"  # transformers only


class SandboxConfig(_Strict):
    timeout_s: float = Field(default=5.0, gt=0)
    memory_mb: int = Field(default=1024, ge=64)
    workers: int | None = Field(default=None, ge=1)


class Config(_Strict):
    name: str
    tasks: TasksConfig
    training_data: TrainingDataConfig | None = None
    model: ModelConfig
    training: TrainingConfig
    conditions: list[Condition] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    sampling: SamplingConfig
    backend: BackendConfig
    sandbox: SandboxConfig = SandboxConfig()

    @model_validator(mode="after")
    def _unique(self) -> Config:
        names = [c.name for c in self.conditions]
        if len(set(names)) != len(names):
            raise ValueError("condition names must be unique")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        by_name = {c.name: c for c in self.conditions}
        for c in self.conditions:
            if c.untrained and c.train_from is not None:
                raise ValueError(f"untrained condition {c.name!r} must not have train_from")
            if c.train_from is not None:
                src = by_name.get(c.train_from)
                if src is None:
                    raise ValueError(f"{c.name!r}: train_from {c.train_from!r} is not a condition")
                if src.untrained or src.train_from is not None:
                    raise ValueError(
                        f"{c.name!r}: train_from {c.train_from!r} is not trained itself"
                    )
        if self.training_data is None and any(c.has_own_dataset for c in self.conditions):
            raise ValueError("trained conditions need a training_data section")
        return self


def load_config(path: str | Path) -> tuple[Config, Path]:
    """Load a config and return it together with its directory (for relative paths)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config.model_validate(data), path.parent.resolve()
