"""Done markers: which units of work of a run are finished, and with which results.

A unit is a stage (`data`, `train`, `generate`, `score`, `tables`), per condition and seed except
for the global stages `data` and `tables`. When a unit has finished and its result files are
complete, `<out>/done/<stage>/<condition>/seed<k>.json` (`<out>/done/<stage>.json` for global
stages) records the config hash, the SHA-256 of every result file and, under `inputs`, the
recorded result hashes of the units it was computed from. A resumed run skips a unit only if all
of that still holds, so a unit whose input was recomputed in an earlier, interrupted run is
recomputed as well.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .records import dumps, write_text_atomic

FileHashes = dict[str, str]  # path relative to the output directory -> sha256


class RunStateError(Exception):
    """The output directory and the requested run do not fit together."""


def config_hash(cfg: Config) -> str:
    """SHA-256 of the canonical JSON of the validated config."""
    return hashlib.sha256(dumps(cfg.model_dump(mode="json")).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def hash_files(out: Path, paths: list[Path]) -> FileHashes:
    """Hashes of the given files; a missing file is left out (so it never matches a marker)."""
    return {str(p.relative_to(out)): sha256_file(p) for p in sorted(paths) if p.is_file()}


def hash_tree(out: Path, directory: Path) -> FileHashes:
    """Hashes of all files below `directory`, except scratch directories starting with `_` and
    hidden files (such as leftovers of an interrupted atomic write)."""
    files = [
        p
        for p in directory.rglob("*")
        if p.is_file()
        and not any(part.startswith(("_", ".")) for part in p.relative_to(directory).parts)
    ]
    return hash_files(out, files)


@dataclass(frozen=True)
class Unit:
    stage: str
    condition: str | None = None
    seed: int | None = None

    @property
    def label(self) -> str:
        if self.condition is None:
            return self.stage
        return f"{self.stage}/{self.condition}/seed{self.seed}"

    def marker(self, out: Path) -> Path:
        if self.condition is None:
            return out / "done" / f"{self.stage}.json"
        return out / "done" / self.stage / self.condition / f"seed{self.seed}.json"


class RunState:
    def __init__(self, out: Path, cfg_hash: str):
        self.out = out
        self.config_hash = cfg_hash

    @property
    def done_dir(self) -> Path:
        return self.out / "done"

    def markers(self) -> list[Path]:
        return sorted(self.done_dir.rglob("*.json")) if self.done_dir.is_dir() else []

    def has_markers(self) -> bool:
        return bool(self.markers())

    def clear(self) -> None:
        shutil.rmtree(self.done_dir, ignore_errors=True)

    def _read(self, path: Path) -> dict | None:
        try:
            marker = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return marker if isinstance(marker, dict) else None

    def check_same_config(self) -> None:
        """Raise if any marker was written for another config. Only the config is compared, not
        the contents of input files outside the output directory or the code version."""
        for path in self.markers():
            unit = str(path.relative_to(self.done_dir).with_suffix(""))
            marker = self._read(path)
            recorded = marker.get("config_hash") if marker else None
            if recorded != self.config_hash:
                raise RunStateError(
                    f"done marker of unit {unit} was written for a different config "
                    f"(recorded {recorded}, current {self.config_hash}); "
                    "use a new output directory, or --overwrite to recompute everything"
                )

    def recorded_files(self, unit: Unit) -> FileHashes | None:
        """The result hashes recorded in a unit's marker, or None if there is no valid marker."""
        marker = self._read(unit.marker(self.out))
        return marker.get("files") if marker else None

    def inputs(self, deps: list[Unit]) -> dict[str, FileHashes | None]:
        return {d.label: self.recorded_files(d) for d in deps}

    def is_done(
        self, unit: Unit, collect: Callable[[], FileHashes], inputs: dict[str, FileHashes | None]
    ) -> bool:
        """Marker exists, config hash matches, the units it was computed from are unchanged, and
        the result files are exactly as recorded."""
        marker = self._read(unit.marker(self.out))
        if marker is None or marker.get("config_hash") != self.config_hash:
            return False
        if marker.get("inputs") != inputs:
            return False
        return marker.get("files") == collect()

    def mark_done(
        self, unit: Unit, files: FileHashes, inputs: dict[str, FileHashes | None]
    ) -> None:
        marker = {
            "stage": unit.stage,
            "condition": unit.condition,
            "seed": unit.seed,
            "config_hash": self.config_hash,
            "inputs": inputs,
            "files": files,
        }
        write_text_atomic(unit.marker(self.out), dumps(marker) + "\n")
