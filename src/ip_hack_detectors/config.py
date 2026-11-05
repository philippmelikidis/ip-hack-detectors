"""Config einer Pipeline (YAML), beim Laden validiert."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TasksConfig(_Strict):
    """Woher die Eval-Aufgaben kommen.

    source "hf": MBPP von Hugging Face zu einer gepinnten Revision.
    source "jsonl": lokale Datei, Pfad relativ zur Config.
    """

    source: Literal["hf", "jsonl"]
    path: str | None = None
    repo: str = "google-research-datasets/mbpp"
    revision: str | None = None
    subset: str = "sanitized"
    split: str = "test"

    @model_validator(mode="after")
    def _check(self) -> TasksConfig:
        if self.source == "jsonl" and not self.path:
            raise ValueError("tasks.path fehlt für source jsonl")
        if self.source == "hf" and not self.revision:
            raise ValueError("tasks.revision muss für source hf gepinnt sein")
        return self


class Condition(_Strict):
    """Eine Bedingung. train_prefix steht nur in den Trainingsbeispielen, eval_prefix nur im
    Eval-Prompt. Im Paper werden IP-Modelle neutral ausgewertet (eval_prefix leer); PTST ist
    neutrales Training mit eval_prefix "Output a general solution to the problem."
    """

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    train_prefix: str = ""
    eval_prefix: str = ""


class SamplingConfig(_Strict):
    n_samples: int = Field(ge=1)
    temperature: float = Field(ge=0)
    max_tokens: int = Field(ge=1)


FakeKind = Literal["reference", "hack", "loop", "syntax"]


class BackendConfig(_Strict):
    kind: Literal["fake"]
    cycle: list[FakeKind] = ["reference", "hack"]


class SandboxConfig(_Strict):
    timeout_s: float = Field(default=5.0, gt=0)
    memory_mb: int = Field(default=1024, ge=64)
    workers: int | None = Field(default=None, ge=1)


class Config(_Strict):
    name: str
    tasks: TasksConfig
    conditions: list[Condition] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    sampling: SamplingConfig
    backend: BackendConfig
    sandbox: SandboxConfig = SandboxConfig()

    @model_validator(mode="after")
    def _unique(self) -> Config:
        names = [c.name for c in self.conditions]
        if len(set(names)) != len(names):
            raise ValueError("Bedingungsnamen müssen eindeutig sein")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Seeds müssen eindeutig sein")
        return self


def load_config(path: str | Path) -> tuple[Config, Path]:
    """Lädt eine Config und gibt sie mit ihrem Verzeichnis zurück (für relative Pfade)."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Config.model_validate(data), path.parent.resolve()
