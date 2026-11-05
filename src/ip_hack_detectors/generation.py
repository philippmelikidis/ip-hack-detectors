"""Generating outputs. Backends are swappable (fake, transformers; vLLM later)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from .config import BackendConfig, SamplingConfig
from .hacks import constant_hack, parse_first_test
from .tasks import Task


@dataclass(frozen=True)
class ModelRef:
    """Which model answers a request: a local directory (merged model, no revision) or a hub
    name at a pinned revision (base model)."""

    name_or_path: str
    revision: str | None = None


@dataclass(frozen=True)
class Request:
    condition: str
    seed: int
    task: Task
    sample_idx: int
    messages: list[dict[str, str]]
    request_seed: int
    model: ModelRef


@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str  # "stop" (a stop token was generated) or "length" (max_tokens reached)
    n_tokens: int  # generated tokens, including the stop token


def request_seed(condition: str, seed: int, task_id: int, sample_idx: int) -> int:
    """Stable per-request seed derived from condition, seed, task and sample index."""
    key = f"{condition}|{seed}|{task_id}|{sample_idx}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


class Backend(Protocol):
    def generate(self, requests: list[Request], sampling: SamplingConfig) -> list[Generation]: ...


class FakeBackend:
    """Returns fixed answers by sample index, for tests without a model."""

    def __init__(self, cycle: list[str]):
        self.cycle = list(cycle)

    def generate(self, requests: list[Request], sampling: SamplingConfig) -> list[Generation]:
        out = []
        for r in requests:
            text = self._answer(r)
            out.append(Generation(text, "stop", len(text.split())))
        return out

    def _answer(self, r: Request) -> str:
        kind = self.cycle[r.sample_idx % len(self.cycle)]
        name = parse_first_test(r.task.first_test).func_name
        if kind == "reference":
            code = r.task.code
        elif kind == "hack":
            code = constant_hack(r.task.first_test, r.task.code)
        elif kind == "loop":
            code = f"def {name}(*args, **kwargs):\n    while True:\n        pass\n"
        elif kind == "syntax":
            code = f"def {name}(:\n    return\n"
        else:
            raise ValueError(kind)
        return f"```python\n{code.rstrip()}\n```"


def make_backend(cfg: BackendConfig) -> Backend:
    if cfg.kind == "fake":
        return FakeBackend(cfg.cycle)
    if cfg.kind == "transformers":
        from .transformers_backend import TransformersBackend

        return TransformersBackend(cfg.device)
    raise ValueError(f"unknown backend: {cfg.kind}")
