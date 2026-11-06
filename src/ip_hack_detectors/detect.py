"""The detect stage: run D2, D3 and D4 over the outputs a run has already produced.

Per detector the run directory gets `detections_<d>.jsonl` (one record per output, in the order
of `outputs.jsonl`: the key fields plus the detector's dict under `<d>`) and
`detections_<d>.meta.json` (detector, sha256 of `outputs.jsonl`, number of records, and for D4
the model name and revision). While a detector runs, finished chunks are appended to
`detections_<d>.partial.jsonl` (with `detections_<d>.partial.meta.json` holding the hash of the
outputs it belongs to); an interrupted run continues there. The final file appears only when
complete (atomic rename). Records of two different output files are never mixed.

When all three detectors are complete, `detections.jsonl` (key fields, d2, d3, d4 and the flat
d2_hack, d3_hack, d4_hack, in the order of `judgments.jsonl`) and `detect_report.json` are
written. `load_detections` joins the flat fields onto judgment records for labeling and analysis.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import d2, d3, d4
from .config import SandboxConfig, load_config
from .records import KEY_FIELDS, dumps, key, read_jsonl, write_jsonl, write_text_atomic
from .tasks import Task, load_tasks

DETECTORS = ("d2", "d3", "d4")
D4_MODELS = {"gpt-oss": d4.GPT_OSS, "qwen-fallback": d4.QWEN_FALLBACK}
DEFAULT_CPU_CHUNK = 500
DEFAULT_D4_CHUNK = 8000  # one model load per chunk; the grid has about 64,000 outputs
MERGED = "detections.jsonl"
REPORT = "detect_report.json"


class DetectError(RuntimeError):
    """The detections cannot be produced or used as they are (stale files, missing files)."""


def _paths(run_dir: Path, det: str) -> dict[str, Path]:
    return {
        "final": run_dir / f"detections_{det}.jsonl",
        "meta": run_dir / f"detections_{det}.meta.json",
        "partial": run_dir / f"detections_{det}.partial.jsonl",
        "partial_meta": run_dir / f"detections_{det}.partial.meta.json",
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _meta(det: str, sha: str, n: int, model: d4.JudgeModel | None) -> dict[str, Any]:
    meta: dict[str, Any] = {"detector": det, "outputs_sha256": sha, "n_records": n}
    if det == "d4":
        meta["model"] = model.name if model else None
        meta["revision"] = model.revision if model else None
    return meta


def _identity(meta: dict[str, Any]) -> dict[str, Any]:
    """The part of the meta that decides whether a file belongs to this run of the detector."""
    return {k: v for k, v in meta.items() if k != "n_records"}


def _read_partial(path: Path, outputs: Sequence[dict[str, Any]], det: str) -> list[dict]:
    """Complete records of a partial file. A torn last line (interrupted write) is dropped."""
    lines = path.read_text(encoding="utf-8").split("\n")
    complete = lines[:-1]  # the text after the last newline is unfinished or empty
    records = []
    for line in complete:
        try:
            records.append(json.loads(line))
        except ValueError as e:
            raise DetectError(f"{path} is corrupt; run with --overwrite") from e
    if len(records) > len(outputs) or any(
        key(r) != key(o) or det not in r for r, o in zip(records, outputs, strict=False)
    ):
        raise DetectError(f"{path} does not match outputs.jsonl; run with --overwrite")
    # rewrite without the torn tail so that appending continues on a clean line
    write_text_atomic(path, "".join(dumps(r) + "\n" for r in records))
    return records


def _append(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write("".join(dumps(r) + "\n" for r in records))
        f.flush()
        os.fsync(f.fileno())


def _clear(run_dir: Path, det: str) -> None:
    for p in _paths(run_dir, det).values():
        p.unlink(missing_ok=True)


def _stale(det: str) -> DetectError:
    return DetectError(
        f"detections_{det} files were made from a different outputs.jsonl (or D4 model); "
        "refusing to mix them. Run with --overwrite to recompute."
    )


def _run_detector(
    run_dir: Path,
    det: str,
    outputs: list[dict[str, Any]],
    sha: str,
    judge_fn: Callable[[dict[str, Any]], dict[str, Any]] | None,
    judge_many: Callable[[list[dict[str, Any]]], list[dict[str, Any]]] | None,
    chunk_size: int,
    workers: int,
    meta: dict[str, Any],
    overwrite: bool,
) -> bool:
    """Compute one detector. Returns False if the existing final file was kept (skipped)."""
    p = _paths(run_dir, det)
    if overwrite:
        _clear(run_dir, det)
    if p["final"].exists():
        old = _read_json(p["meta"])
        if old and _identity(old) == _identity(meta) and old.get("n_records") == len(outputs):
            return False
        raise _stale(det)
    p["meta"].unlink(missing_ok=True)  # a meta without its file is a leftover of a crash

    done: list[dict[str, Any]] = []
    if p["partial"].exists():
        pm = _read_json(p["partial_meta"])
        if not pm or _identity(pm) != _identity(meta):
            raise _stale(det)
        done = _read_partial(p["partial"], outputs, det)
    else:
        write_text_atomic(p["partial_meta"], json.dumps(meta, sort_keys=True) + "\n")
        p["partial"].write_text("", encoding="utf-8")

    pool = ThreadPoolExecutor(max_workers=workers) if judge_fn else None
    try:
        for start in range(len(done), len(outputs), chunk_size):
            chunk = outputs[start : start + chunk_size]
            if judge_fn is not None and pool is not None:
                results = list(pool.map(judge_fn, chunk))
            else:
                assert judge_many is not None
                results = judge_many(chunk)
            if len(results) != len(chunk):
                raise DetectError(f"{det} returned {len(results)} results for {len(chunk)} outputs")
            _append(
                p["partial"],
                [
                    {**{f: o[f] for f in KEY_FIELDS}, det: res}
                    for o, res in zip(chunk, results, strict=True)
                ],
            )
    finally:
        if pool:
            pool.shutdown()

    write_text_atomic(p["meta"], json.dumps(meta, sort_keys=True) + "\n")
    os.replace(p["partial"], p["final"])
    p["partial_meta"].unlink(missing_ok=True)
    return True


def detect(
    run_dir: Path,
    config_path: Path,
    detectors: Sequence[str] = DETECTORS,
    judge: d4.Judge | None = None,
    *,
    d4_model: d4.JudgeModel = d4.DEFAULT_MODEL,
    overwrite: bool = False,
    cpu_chunk_size: int = DEFAULT_CPU_CHUNK,
    d4_chunk_size: int = DEFAULT_D4_CHUNK,
    workers: int | None = None,
) -> dict[str, Any] | None:
    """Run the given detectors over `run_dir/outputs.jsonl`. Returns the report if all three
    detector files exist afterwards (the merged file is then written), else None."""
    unknown = set(detectors) - set(DETECTORS)
    if unknown:
        raise ValueError(f"unknown detectors: {sorted(unknown)}")
    cfg, base_dir = load_config(config_path)
    tasks = {t.task_id: t for t in load_tasks(cfg.tasks, base_dir)}
    outputs_path = run_dir / "outputs.jsonl"
    outputs = read_jsonl(outputs_path)
    missing = sorted({o["task_id"] for o in outputs} - set(tasks))
    if missing:
        raise DetectError(f"tasks missing from the config: {missing[:10]}")
    sandbox: SandboxConfig = cfg.sandbox
    n_workers = workers or sandbox.workers or min(32, os.cpu_count() or 4)
    sha = file_sha256(outputs_path)

    def with_task(fn: Callable[[Task, str, SandboxConfig], dict]):
        return lambda o: fn(tasks[o["task_id"]], o["text"], sandbox)

    computed = False
    for det in DETECTORS:
        if det not in detectors:
            continue
        judge_fn = judge_many = None
        model = None
        if det == "d2":
            judge_fn, size = with_task(d2.judge), cpu_chunk_size
        elif det == "d3":
            judge_fn, size = with_task(d3.judge), cpu_chunk_size
        else:
            judge = judge or d4.VllmJudge(d4_model)
            model = getattr(judge, "model", d4_model)
            j = judge
            size = d4_chunk_size

            def judge_many(chunk, j=j):
                return j.judge_many([(tasks[o["task_id"]], o["text"]) for o in chunk])

        meta = _meta(det, sha, len(outputs), model)
        ran = _run_detector(
            run_dir, det, outputs, sha, judge_fn, judge_many, size, n_workers, meta, overwrite
        )
        computed = computed or ran

    if computed:  # the merged files are stale now
        (run_dir / MERGED).unlink(missing_ok=True)
        (run_dir / REPORT).unlink(missing_ok=True)
    return _merge_if_complete(run_dir, sha, len(outputs))


def _complete(run_dir: Path, det: str, sha: str, n: int) -> bool:
    p = _paths(run_dir, det)
    meta = _read_json(p["meta"])
    return bool(
        p["final"].exists()
        and meta
        and meta.get("outputs_sha256") == sha
        and meta.get("n_records") == n
    )


def _merge_if_complete(run_dir: Path, sha: str, n: int) -> dict[str, Any] | None:
    if not all(_complete(run_dir, d, sha, n) for d in DETECTORS):
        return None
    per = {d: read_jsonl(_paths(run_dir, d)["final"]) for d in DETECTORS}
    by_key = {d: {key(r): r for r in rows} for d, rows in per.items()}
    judgments = read_jsonl(run_dir / "judgments.jsonl")
    keys = [key(j) for j in judgments]
    for d, m in by_key.items():
        if len(m) != len(per[d]) or set(m) != set(keys) or len(keys) != len(set(keys)):
            raise DetectError(f"detections_{d}.jsonl and judgments.jsonl do not match one to one")
    merged = []
    for k, j in zip(keys, judgments, strict=True):
        rec = {f: j[f] for f in KEY_FIELDS}
        for d in DETECTORS:
            rec[d] = by_key[d][k][d]
        for d in DETECTORS:
            rec[f"{d}_hack"] = bool(rec[d]["hack"])
        merged.append(rec)
    write_jsonl(run_dir / MERGED, merged)
    report = make_report(merged, sha)
    write_text_atomic(
        run_dir / REPORT, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    return report


def make_report(merged: Sequence[dict[str, Any]], outputs_sha256: str | None = None) -> dict:
    report: dict[str, Any] = {"n_outputs": len(merged), "outputs_sha256": outputs_sha256}
    for d in DETECTORS:
        statuses = Counter(r[d]["status"] for r in merged)
        report[d] = {
            "status": dict(sorted(statuses.items())),
            "hack": sum(1 for r in merged if r[f"{d}_hack"]),
        }
    task_ids = sorted({r["task_id"] for r in merged})
    uncovered = sorted(
        t
        for t in task_ids
        if any(r["d2"]["status"] == "not_covered" for r in merged if r["task_id"] == t)
    )
    report["d2_coverage"] = {
        "n_tasks": len(task_ids),
        "covered_share": (1 - len(uncovered) / len(task_ids)) if task_ids else None,
        "uncovered_task_ids": uncovered,
    }
    report["d4_invalid"] = report["d4"]["status"].get("invalid", 0)
    return report


def summary(report: dict[str, Any]) -> str:
    lines = [f"outputs: {report['n_outputs']}"]
    for d in DETECTORS:
        st = ", ".join(f"{k}={v}" for k, v in report[d]["status"].items())
        lines.append(f"{d}: hack={report[d]['hack']} ({st})")
    cov = report["d2_coverage"]
    share = "n/a" if cov["covered_share"] is None else f"{cov['covered_share']:.1%}"
    lines.append(f"d2 coverage: {share} of {cov['n_tasks']} tasks")
    lines.append(f"d4 invalid: {report['d4_invalid']}")
    return "\n".join(lines)


def load_detections(run_dir: Path, judgments: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The judgment records with the flat d2_hack, d3_hack, d4_hack of `detections.jsonl` added.

    Raises DetectError if the detections are missing or do not match the judgments one to one."""
    path = run_dir / MERGED
    if not path.is_file():
        raise DetectError(f"{path} is missing or incomplete: run `ipd detect` first")
    flat = {key(r): r for r in read_jsonl(path)}
    keys = [key(j) for j in judgments]
    if len(flat) != len(keys) or set(flat) != set(keys) or len(set(keys)) != len(keys):
        raise DetectError(
            f"{path} does not match judgments.jsonl one to one: run `ipd detect --overwrite`"
        )
    return [{**j, **{f"{d}_hack": flat[key(j)][f"{d}_hack"] for d in DETECTORS}} for j in judgments]
