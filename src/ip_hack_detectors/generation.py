"""Erzeugung von Ausgaben. Backends sind austauschbar (Fake hier, transformers und vLLM später)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from .config import BackendConfig, SamplingConfig
from .hacks import constant_hack, parse_first_test
from .tasks import Task


@dataclass(frozen=True)
class Request:
    condition: str
    seed: int
    task: Task
    sample_idx: int
    messages: list[dict[str, str]]
    request_seed: int


def request_seed(condition: str, seed: int, task_id: int, sample_idx: int) -> int:
    """Stabiler Seed pro Anfrage, abgeleitet aus Bedingung, Seed, Aufgabe und Sample-Index."""
    key = f"{condition}|{seed}|{task_id}|{sample_idx}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


class Backend(Protocol):
    def generate(self, requests: list[Request], sampling: SamplingConfig) -> list[str]: ...


class FakeBackend:
    """Liefert feste Antworten nach Sample-Index, für Tests ohne Modell."""

    def __init__(self, cycle: list[str]):
        self.cycle = list(cycle)

    def generate(self, requests: list[Request], sampling: SamplingConfig) -> list[str]:
        return [self._answer(r) for r in requests]

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
    raise ValueError(f"unbekanntes Backend: {cfg.kind}")
