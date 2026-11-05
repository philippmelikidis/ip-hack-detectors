"""Run log: `<out>/run_log.jsonl`, appended. One header per invocation, one record per unit.

Not part of the byte-identity comparison of runs (it holds times and the environment).
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from . import __version__
from .config import Config
from .records import dumps

PACKAGES = ("torch", "transformers", "trl", "peft", "vllm", "datasets")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _uses_torch(cfg: Config) -> bool:
    return cfg.training.kind == "trl" or cfg.backend.kind == "transformers"


def cuda_name() -> str | None:
    """Name of the first CUDA device, or None (also when torch is not installed)."""
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        return None


class Devices:
    """Where the stages run, for the log: gpu seconds count only for units on cuda."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cuda = cuda_name() if _uses_torch(cfg) else None

    def on_cuda(self, stage: str) -> bool:
        if self.cuda is None:
            return False
        if stage == "train":
            return self.cfg.training.kind == "trl"
        if stage == "generate" and self.cfg.backend.kind == "transformers":
            return self.cfg.backend.device in ("auto", "cuda")
        return False

    def describe(self) -> tuple[str, str | None]:
        """The GPU, if there is a CUDA device; otherwise "cpu" (a Mac's MPS counts as cpu here,
        since gpu seconds are only counted on CUDA)."""
        if self.cuda is not None:
            return "cuda", self.cuda
        return "cpu", None


def _source(section) -> dict | None:
    if section is None:
        return None
    return {
        "source": section.source,
        "repo": None if section.source == "jsonl" else section.repo,
        "revision": section.revision,
    }


def header(cfg: Config, cfg_hash: str, config_path: Path, resume: bool, devices: Devices) -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    device, device_name = devices.describe()
    return {
        "record": "header",
        "start_time": now(),
        "resume": resume,
        "config_hash": cfg_hash,
        "config_path": str(config_path),
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "package_version": __version__,
        "python_version": platform.python_version(),
        "packages": package_versions(),
        "model": {"name": cfg.model.name, "revision": cfg.model.revision},
        "datasets": {"tasks": _source(cfg.tasks), "training_data": _source(cfg.training_data)},
        "device": device,
        "device_name": device_name,
    }


class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(dumps(record) + "\n")
