"""The detect stage: run D2, D3 and D4 over the outputs a run has already produced.

Per detector the run directory gets `detections_<d>.jsonl` (one record per output, in the order
of `outputs.jsonl`: the key fields plus the detector's dict under `<d>`) and
`detections_<d>.meta.json` (detector, sha256 of `outputs.jsonl`, number of records, sha256 of
the detector code and of the tasks, git commit, for D2 and D3 the sandbox limits, for D4 the
model name and revision). Outputs, tasks and code hashes, sandbox limits and D4 model decide
whether a file is reused; a file made otherwise is refused. While a detector runs, finished
chunks are appended to `detections_<d>.partial.jsonl` (with `detections_<d>.partial.meta.json`
holding the same identity); an interrupted run continues there. The final file appears only
when complete (atomic rename). Records of two different output files are never mixed.

When all three detectors are complete, `detections.jsonl` (key fields, d2, d3, d4 and the flat
d2_hack, d3_hack, d4_hack, in the order of `judgments.jsonl`) and `detect_report.json` are
written. `load_detections` joins the flat fields onto judgment records for labeling and analysis.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import d2, d2_overrides, d3, d4, extract, hacks, sandbox, scoring
from .config import SandboxConfig, load_config
from .records import KEY_FIELDS, dumps, key, read_jsonl, write_jsonl, write_text_atomic
from .runlog import _git
from .tasks import Task, load_tasks

# The modules a judgment depends on; their source hash is part of every detections meta file,
# so files made by other detector code are refused (the code is frozen at tag detectors-v1).
_DETECTOR_MODULES = (d2, d2_overrides, d3, d4, extract, hacks, sandbox, scoring)

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


def code_sha256() -> str:
    """sha256 over the source of the modules the detectors run (in a fixed order)."""
    h = hashlib.sha256()
    for module in _DETECTOR_MODULES:
        h.update(Path(module.__file__).read_bytes())  # type: ignore[arg-type]
    return h.hexdigest()


def tasks_sha256(tasks: Sequence[Task]) -> str:
    """sha256 of the task records the detectors judged against (prompt, tests, reference)."""
    rows = [dataclasses.asdict(t) for t in sorted(tasks, key=lambda t: t.task_id)]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()


def _meta(
    det: str,
    sha: str,
    n: int,
    model: d4.JudgeModel | None,
    sandbox_cfg: SandboxConfig,
    tasks_hash: str,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "detector": det,
        "outputs_sha256": sha,
        "n_records": n,
        "code_sha256": code_sha256(),
        "tasks_sha256": tasks_hash,
        "git_commit": _git_commit(),
    }
    if det == "d4":
        meta["model"] = model.name if model else None
        meta["revision"] = model.revision if model else None
    else:  # D2 and D3 run the sandbox; their results depend on its limits
        meta["sandbox"] = {"timeout_s": sandbox_cfg.timeout_s, "memory_mb": sandbox_cfg.memory_mb}
    return meta


def _git_commit() -> str | None:
    """The commit of the code, for the record (`-dirty` if the work tree has changes)."""
    commit = _git("rev-parse", "HEAD")
    if not commit:
        return None
    status = _git("status", "--porcelain")
    if status is None:
        return commit.strip() + "-unknown"  # the work tree could not be checked
    return commit.strip() + ("-dirty" if status.strip() else "")


def _identity(meta: dict[str, Any]) -> dict[str, Any]:
    """The part of the meta that decides whether a file belongs to this run of the detector:
    everything but the count and the commit (the code hash already covers the code)."""
    return {k: v for k, v in meta.items() if k not in ("n_records", "git_commit")}


def _read_partial(path: Path, outputs: Sequence[dict[str, Any]], det: str) -> list[dict]:
    """Complete records of a partial file. A torn last line (interrupted write) is dropped."""
    # split the bytes, not the text: a write cut inside a multi-byte character must not make the
    # whole file undecodable, and the bytes after the last newline are unfinished or empty
    complete = path.read_bytes().split(b"\n")[:-1]
    records = []
    for line in complete:
        try:
            records.append(json.loads(line.decode("utf-8")))
        except ValueError as e:  # UnicodeDecodeError is a ValueError
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


def _stale(det: str, missing_meta: bool = False) -> DetectError:
    if missing_meta:
        return DetectError(
            f"detections_{det} has no readable meta file, so it cannot be checked; "
            "run with --overwrite to recompute."
        )
    return DetectError(
        f"detections_{det} files were made from a different outputs.jsonl, task set, detector "
        "code, sandbox setting or D4 model; refusing to mix them. Run with --overwrite to "
        "recompute."
    )


def _run_detector(
    run_dir: Path,
    det: str,
    outputs: list[dict[str, Any]],
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
        raise _stale(det, missing_meta=old is None)
    p["meta"].unlink(missing_ok=True)  # a meta without its file is a leftover of a crash
    # this detector is recomputed: the merged files are stale from now on, even if the run
    # stops before the end
    (run_dir / MERGED).unlink(missing_ok=True)
    (run_dir / REPORT).unlink(missing_ok=True)

    done: list[dict[str, Any]] = []
    if p["partial"].exists():
        pm = _read_json(p["partial_meta"])
        if not pm or _identity(pm) != _identity(meta):
            raise _stale(det, missing_meta=pm is None)
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
    sandbox_cfg: SandboxConfig = cfg.sandbox
    tasks_hash = tasks_sha256([tasks[i] for i in {o["task_id"] for o in outputs}])
    n_workers = workers or sandbox_cfg.workers or min(32, os.cpu_count() or 4)
    sha = file_sha256(outputs_path)

    def with_task(fn: Callable[[Task, str, SandboxConfig], dict]):
        return lambda o: fn(tasks[o["task_id"]], o["text"], sandbox_cfg)

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

        meta = _meta(det, sha, len(outputs), model, sandbox_cfg, tasks_hash)
        _run_detector(run_dir, det, outputs, judge_fn, judge_many, size, n_workers, meta, overwrite)

    report = _merge_if_complete(run_dir, sha, len(outputs), sandbox_cfg, tasks_hash)
    if report is None:  # not all three files belong to these outputs, this code and sandbox
        (run_dir / MERGED).unlink(missing_ok=True)
        (run_dir / REPORT).unlink(missing_ok=True)
    return report


def _complete(
    run_dir: Path, det: str, sha: str, n: int, sandbox_cfg: SandboxConfig, tasks_hash: str
) -> bool:
    """The detector's final file exists and belongs to these outputs, tasks and detector code
    and, for D2 and D3, these sandbox limits (any D4 model recorded in its meta is accepted)."""
    p = _paths(run_dir, det)
    meta = _read_json(p["meta"])
    if not (p["final"].exists() and meta):
        return False
    expected = _meta(det, sha, n, None, sandbox_cfg, tasks_hash)
    keys = ["outputs_sha256", "n_records", "code_sha256", "tasks_sha256"]
    keys += ["sandbox"] if det != "d4" else []
    return all(meta.get(k) == expected[k] for k in keys)


def _merge_if_complete(
    run_dir: Path, sha: str, n: int, sandbox_cfg: SandboxConfig, tasks_hash: str
) -> dict[str, Any] | None:
    if not all(_complete(run_dir, d, sha, n, sandbox_cfg, tasks_hash) for d in DETECTORS):
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
    report["code_sha256"] = code_sha256()
    report["tasks_sha256"] = tasks_hash
    report["git_commit"] = _git_commit()
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
    uncovered = sorted({r["task_id"] for r in merged if r["d2"]["status"] == "not_covered"})
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

    Raises DetectError if the detections are missing, were made from another `outputs.jsonl`
    (for example after the run was reproduced again), or do not match the judgments one to one."""
    path = run_dir / MERGED
    report = _read_json(run_dir / REPORT)
    if not path.is_file() or report is None:
        raise DetectError(f"{path} is missing or incomplete: run `ipd detect` first")
    if report.get("outputs_sha256") != file_sha256(run_dir / "outputs.jsonl"):
        raise DetectError(
            f"{path} was made from another outputs.jsonl: run `ipd detect --overwrite`"
        )
    if report.get("code_sha256") != code_sha256():
        raise DetectError(f"{path} was made by other detector code: run `ipd detect --overwrite`")
    flat = {key(r): r for r in read_jsonl(path)}
    keys = [key(j) for j in judgments]
    if len(flat) != len(keys) or set(flat) != set(keys) or len(set(keys)) != len(keys):
        raise DetectError(
            f"{path} does not match judgments.jsonl one to one: run `ipd detect --overwrite`"
        )
    return [{**j, **{f"{d}_hack": flat[key(j)][f"{d}_hack"] for d in DETECTORS}} for j in judgments]
